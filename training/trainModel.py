import torch, struct, requests, re, math, os, json, sys, time, torch.nn as nn
from torch.nn import functional as F
from collections import Counter

torch.set_num_threads(4)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from curriculum import generateAtomicSample

with open("config/model.json", "r") as f: modCfg = json.load(f)
cfgPath = "config/world.json" if os.path.exists("config/world.json") else "config/logic.json"
with open(cfgPath, "r") as f: logCfg = json.load(f)

weightsOverride = (
    modCfg.get("weights")
    or modCfg.get("training", {}).get("weights")
    or modCfg.get("category_weights")
    or modCfg.get("training", {}).get("category_weights")
    or logCfg.get("weights")
)
if os.path.exists("config/weights.json"):
    try:
        with open("config/weights.json", "r") as f:
            weightsOverride = json.load(f)
    except Exception:
        pass

if weightsOverride:
    logCfg["weights"] = weightsOverride
    print(f"[✓] Applied separate weights dictionary ({len(weightsOverride)} category overrides).")

# Architecture Configuration
VOCAB_SIZE   = modCfg["architecture"]["vocab_size"]
BLOCK_SIZE   = modCfg["architecture"]["block_size"]
EMBED_DIM    = modCfg["architecture"]["embed_dim"]
NUM_HEADS    = modCfg["architecture"]["num_heads"]
NUM_LAYERS   = modCfg["architecture"]["num_layers"]
MEOW_LAYERS  = modCfg["architecture"].get("meow_layers", 3)
MIAOU_LAYERS = modCfg["architecture"].get("miaou_layers", 3)
HEAD_DIM     = EMBED_DIM // NUM_HEADS
DEVICE       = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Training Hyperparameters loaded from model.json
trnCfg = modCfg.get("training", {})
WEIGHT_DECAY = trnCfg.get("weight_decay", 0.01)
GRAD_CLIP    = trnCfg.get("grad_clip", 1.0)

pretrainCfg    = trnCfg.get("pretrain", {})
PRETRAIN_STEPS = pretrainCfg.get("steps", 10000)
PRETRAIN_BATCH = pretrainCfg.get("batch_size", 32)
PRETRAIN_LR    = pretrainCfg.get("base_lr", 0.002)
PRETRAIN_MINLR = pretrainCfg.get("min_lr", 0.0002)
WARMUP_STEPS   = pretrainCfg.get("warmup_steps", 500)

sftCfg    = trnCfg.get("sft", {})
SFT_STEPS = sftCfg.get("steps", 16000)
SFT_BATCH = sftCfg.get("batch_size", 24)
SFT_LR    = sftCfg.get("base_lr", 0.0005)
SFT_MINLR = sftCfg.get("min_lr", 0.00005)

MODELS_DIR = modCfg["paths"]["models_dir"]
STORIES_FILE = modCfg["paths"]["stories_file"]
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("firmware", exist_ok=True)

print("=" * 65)
print(f" TWO-STAGE TRAINING: PRE-TRAIN -> LOGIC SFT ({NUM_LAYERS}L MHA, {EMBED_DIM}D, {NUM_HEADS}H)")
print("=" * 65)

# 1. Dataset & Vocabulary Preparation
if os.path.exists(STORIES_FILE):
    print(f"[*] Reading TinyStories from cached {STORIES_FILE}...")
    with open(STORIES_FILE, "r", encoding="utf-8", errors="ignore") as f: rawStories = f.read().lower()
else:
    print("[*] Downloading TinyStories dataset...")
    resp = requests.get(modCfg["dataset"]["url"], headers={"User-Agent": "Mozilla/5.0"}, stream=True)
    with open(STORIES_FILE, "wb") as f:
        for chunk in resp.iter_content(chunk_size=512 * 1024):
            f.write(chunk)
            if f.tell() >= 6 * 1024 * 1024: break
    with open(STORIES_FILE, "r", encoding="utf-8", errors="ignore") as f: rawStories = f.read().lower()

TARGET_VOCAB_SAMPLES = modCfg["dataset"].get("logic_samples_target", 8000)
allQaTasks = [generateAtomicSample(logCfg) for _ in range(TARGET_VOCAB_SAMPLES)]

TOKEN_REGEX = r"\[[cqa]\]|<eos>|<unk>|<pad>|'s|\w+|[.,?!:]"

def pluralize(word):
    if word.endswith(("s", "x", "z", "ch", "sh")): return word + "es"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou": return word[:-1] + "ies"
    return word + "s"

autoPoolWords = set()
for poolKey, poolList in logCfg.get("pools", {}).items():
    if isinstance(poolList, list):
        for item in poolList:
            autoPoolWords.add(item.lower())
            if poolKey in ("item", "container"):
                autoPoolWords.add(pluralize(item.lower()))

BASE_CORE = modCfg["dataset"]["core_words"]
CAT_CORE_WORDS = list(dict.fromkeys(BASE_CORE + sorted(list(autoPoolWords))))

storyWords = re.findall(TOKEN_REGEX, rawStories[:4000000])
cleanStoryWords = [w for w in storyWords if w in CAT_CORE_WORDS or (w.isalpha() and len(w) > 1 and not w.endswith("n't"))]
qaWords = re.findall(TOKEN_REGEX, " ".join(allQaTasks))

counts = Counter(cleanStoryWords + qaWords)
for w in CAT_CORE_WORDS: counts.pop(w, None)

remainingSlots = VOCAB_SIZE - len(CAT_CORE_WORDS)
VOCAB = CAT_CORE_WORDS + [w for w, _ in counts.most_common(remainingSlots)]
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}
padId, aTokenId = WORD_TO_ID["<pad>"], WORD_TO_ID["[a]"]

print(f"[✓] Vocab: {len(CAT_CORE_WORDS)} core/pool words + {len(VOCAB) - len(CAT_CORE_WORDS)} natural words.")

def tokenizeStr(s):
    for c in [".", "?", "!", ",", ":"]: s = s.replace(c, f" {c} ")
    s = s.replace("'s", " 's ")
    return [WORD_TO_ID.get(w, 0) for w in re.findall(TOKEN_REGEX, s.lower())]

# Prepare TinyStories Chunks
storyTokens = torch.tensor(tokenizeStr(rawStories), dtype=torch.long)
numStChunks = (len(storyTokens) - 1) // BLOCK_SIZE
stXData = storyTokens[:numStChunks * BLOCK_SIZE].view(numStChunks, BLOCK_SIZE).to(DEVICE)
stYData = storyTokens[1 : numStChunks * BLOCK_SIZE + 1].view(numStChunks, BLOCK_SIZE).to(DEVICE)

# 2. RoPE and Model Architecture
def precomputeRopeFreqs(headDim, maxSeqLen=BLOCK_SIZE):
    invFreq = 1.0 / (10000.0 ** (torch.arange(0, headDim, 2).float() / headDim))
    t = torch.arange(maxSeqLen).float()
    freqs = torch.outer(t, invFreq)
    return freqs.cos(), freqs.sin()

cosCached, sinCached = precomputeRopeFreqs(HEAD_DIM, BLOCK_SIZE)

def applyRope(x, cos, sin):
    T = x.shape[2]
    c = cos[:T].unsqueeze(0).unsqueeze(1).to(x.device)
    s = sin[:T].unsqueeze(0).unsqueeze(1).to(x.device)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out

class RoPE_MHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkvProj = nn.Linear(EMBED_DIM, 3 * EMBED_DIM)
        self.outProj = nn.Linear(EMBED_DIM, EMBED_DIM)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkvProj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        q = applyRope(q, cosCached, sinCached)
        k = applyRope(k, cosCached, sinCached)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.outProj(out.transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.sa  = RoPE_MHA()
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.mlp = nn.Sequential(nn.Linear(EMBED_DIM, 4 * EMBED_DIM), nn.ReLU(), nn.Linear(4 * EMBED_DIM, EMBED_DIM))

    def forward(self, x):
        return x + self.mlp(self.ln2(x + self.sa(self.ln1(x))))

class BabyStories(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokEmb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.layers = nn.ModuleList([Block() for _ in range(NUM_LAYERS)])
        self.lnF    = nn.LayerNorm(EMBED_DIM)
        self.lmHead = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, idx):
        x = self.tokEmb(idx)
        for layer in self.layers: x = layer(x)
        return self.lmHead(self.lnF(x))

rawModel = BabyStories().to(DEVICE)

BENCHMARK_MODE = any(arg in sys.argv for arg in ["--benchmark", "--no-train", "--eval-only"]) or os.environ.get("BENCHMARK_ONLY") == "1"
p1Path = os.path.join(MODELS_DIR, "pico1 weights.model")
p2Path = os.path.join(MODELS_DIR, "pico2 weights.model")

if BENCHMARK_MODE:
    print("\n" + "=" * 65)
    print(" [*] BENCHMARK MODE ACTIVE: Skipping training.")
    if os.path.exists(p1Path) and os.path.exists(p2Path):
        print(f" [✓] Found existing weights in {MODELS_DIR}.")
        sys.exit(0)
    print("=" * 65 + "\n")

# =====================================================================
# STAGE 1: PRE-TRAINING ON TINYSTORIES
# =====================================================================
optPretrain = torch.optim.AdamW(rawModel.parameters(), lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY)

def getPretrainLr(step):
    if step < WARMUP_STEPS: return PRETRAIN_LR * (step + 1) / WARMUP_STEPS
    prog = (step - WARMUP_STEPS) / max(1, PRETRAIN_STEPS - WARMUP_STEPS)
    return PRETRAIN_MINLR + 0.5 * (1.0 + math.cos(math.pi * prog)) * (PRETRAIN_LR - PRETRAIN_MINLR)

print("\n" + "=" * 65)
print(f" STAGE 1: PRE-TRAINING (100% TinyStories, {PRETRAIN_STEPS} Steps, Batch {PRETRAIN_BATCH})")
print("=" * 65)

rawModel.train()
t0 = time.time()
for step in range(PRETRAIN_STEPS + 1):
    lr = getPretrainLr(step)
    for p in optPretrain.param_groups: p["lr"] = lr
    
    stIdx = torch.randint(numStChunks, (PRETRAIN_BATCH,))
    xSt, ySt = stXData[stIdx], stYData[stIdx]
    
    logits = rawModel(xSt)
    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), ySt.view(-1))
    
    optPretrain.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(rawModel.parameters(), GRAD_CLIP)
    optPretrain.step()
    
    if step % 500 == 0:
        elapsed = time.time() - t0
        eta = (PRETRAIN_STEPS - step) * (elapsed / max(1, step)) / 60
        print(f" [Pre-train] Step {step:5d}/{PRETRAIN_STEPS} | Story Loss: {loss.item():.4f} | LR: {lr:.5f} | ETA: {eta:.1f}m", flush=True)

# =====================================================================
# STAGE 2: LOGIC SFT (SUPERVISED FINE-TUNING)
# =====================================================================
optSft = torch.optim.AdamW(rawModel.parameters(), lr=SFT_LR, weight_decay=WEIGHT_DECAY)

def getSftLr(step):
    prog = step / max(1, SFT_STEPS)
    return SFT_MINLR + 0.5 * (1.0 + math.cos(math.pi * prog)) * (SFT_LR - SFT_MINLR)

def getQaBatch(batchSize):
    qaX_list, qaY_list = [], []
    for _ in range(batchSize):
        task = generateAtomicSample(logCfg)
        toks = tokenizeStr(task)
        if len(toks) > BLOCK_SIZE: toks = toks[:BLOCK_SIZE]
        x = toks + [padId] * (BLOCK_SIZE - len(toks))
        y = [-100] * BLOCK_SIZE
        
        # Target tokens follow [a]
        ansStart = next((idx for idx, t in enumerate(toks) if t == aTokenId), -1)
        if ansStart != -1:
            for idx in range(ansStart, len(toks) - 1): 
                y[idx] = toks[idx + 1]
                
        qaX_list.append(x)
        qaY_list.append(y)
        
    return (torch.tensor(qaX_list, dtype=torch.long, device=DEVICE),
            torch.tensor(qaY_list, dtype=torch.long, device=DEVICE))

print("\n" + "=" * 65)
print(f" STAGE 2: LOGIC SFT (100% QA Tasks, {SFT_STEPS} Steps, Batch {SFT_BATCH})")
print("=" * 65)

t1 = time.time()
for step in range(SFT_STEPS + 1):
    lr = getSftLr(step)
    for p in optSft.param_groups: p["lr"] = lr
    
    xQa, yQa = getQaBatch(SFT_BATCH)
    logits = rawModel(xQa)
    
    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yQa.view(-1), ignore_index=-100)
    
    optSft.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(rawModel.parameters(), GRAD_CLIP)
    optSft.step()
    
    if step % 500 == 0:
        elapsed = time.time() - t1
        eta = (SFT_STEPS - step) * (elapsed / max(1, step)) / 60
        print(f" [Logic SFT] Step {step:5d}/{SFT_STEPS} | QA Loss: {loss.item():.4f} | LR: {lr:.5f} | ETA: {eta:.1f}m", flush=True)

# =====================================================================
# STAGE 3: INT8 EXPORT & PACKAGING
# =====================================================================
print("\n[*] Quantizing weights (INT8 W8A32) and exporting...")

def dumpTensor(t, f, quantize=True):
    d = t.detach().cpu()
    if d.ndim == 2 and quantize:
        for r in d:
            m = r.abs().max().item()
            s = m / 127.0 if m > 0 else 1.0
            f.write(struct.pack('f', s))
            f.write(torch.clamp(torch.round(r / s), -128, 127).to(torch.int8).numpy().tobytes())
    else:
        f.write(d.to(torch.float32).numpy().tobytes())

with open(p1Path, "wb") as f:
    dumpTensor(rawModel.tokEmb.weight, f, quantize=True)
    for i in range(MEOW_LAYERS):
        for p in rawModel.layers[i].parameters(): dumpTensor(p, f, quantize=True)

with open(p2Path, "wb") as f:
    for i in range(MEOW_LAYERS, NUM_LAYERS):
        for p in rawModel.layers[i].parameters(): dumpTensor(p, f, quantize=True)
    for p in rawModel.lnF.parameters(): dumpTensor(p, f, quantize=True)
    dumpTensor(rawModel.lmHead.weight, f, quantize=True)
    dumpTensor(rawModel.lmHead.bias, f, quantize=True)

vocabCode = f"VOCAB = {repr(VOCAB)}\n"
with open(os.path.join(MODELS_DIR, "vocab.py"), "w") as f: f.write(vocabCode)
with open("firmware/vocab.py", "w") as f: f.write(vocabCode)

print(f"[✓] Meow Weights (pico1, {MEOW_LAYERS}L): {os.path.getsize(p1Path)/1024:.1f} KB -> Saved to {p1Path}")
print(f"[✓] Miaou Weights (pico2, {MIAOU_LAYERS}L): {os.path.getsize(p2Path)/1024:.1f} KB -> Saved to {p2Path}")
print(f"[✓] Vocab updated: {len(VOCAB)} words.")