import torch, struct, requests, re, math, os, json, sys, time, torch.nn as nn
from torch.nn import functional as F
from collections import Counter

torch.set_num_threads(4)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from curriculum import generateAtomicSample

with open("config/model.json", "r") as f: modCfg = json.load(f)
cfgPath = "config/world.json" if os.path.exists("config/world.json") else "config/logic.json"
with open(cfgPath, "r") as f: logCfg = json.load(f)

VOCAB_SIZE   = modCfg["architecture"]["vocab_size"]
BLOCK_SIZE   = modCfg["architecture"]["block_size"]
EMBED_DIM    = modCfg["architecture"]["embed_dim"]
NUM_HEADS    = modCfg["architecture"]["num_heads"]
NUM_LAYERS   = modCfg["architecture"]["num_layers"]
MEOW_LAYERS  = modCfg["architecture"].get("meow_layers", 3)
MIAOU_LAYERS = modCfg["architecture"].get("miaou_layers", 3)
HEAD_DIM     = EMBED_DIM // NUM_HEADS
DEVICE       = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

MODELS_DIR = modCfg["paths"]["models_dir"]
STORIES_FILE = modCfg["paths"]["stories_file"]
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("firmware", exist_ok=True)

print("=" * 65)
print(f" TRAINING {NUM_LAYERS}-LAYER MHA RoPE ENGINE ({EMBED_DIM}D, {NUM_HEADS}H)")
print("=" * 65)

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

TARGET_LOGIC = modCfg["dataset"].get("logic_samples_target", 50000)
print(f"[*] Generating {TARGET_LOGIC} synthetic logic tasks...")
allQaTasks = [generateAtomicSample(logCfg) for _ in range(TARGET_LOGIC)]

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
qaWords = re.findall(TOKEN_REGEX, " ".join(allQaTasks[:8000]))

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

storyTokens = torch.tensor(tokenizeStr(rawStories), dtype=torch.long)
numStChunks = (len(storyTokens) - 1) // BLOCK_SIZE
stXData = storyTokens[:numStChunks * BLOCK_SIZE].view(numStChunks, BLOCK_SIZE).to(DEVICE)
stYData = storyTokens[1 : numStChunks * BLOCK_SIZE + 1].view(numStChunks, BLOCK_SIZE).to(DEVICE)

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
    print(" [*] BENCHMARK MODE ACTIVE: Skipping training loop.")
    if os.path.exists(p1Path) and os.path.exists(p2Path):
        print(f" [✓] Found existing weights in {MODELS_DIR}.")
        sys.exit(0)
    print("=" * 65 + "\n")

WARMUP      = modCfg["training"].get("warmup_steps", 1000)
TOTAL_STEPS = 0 if BENCHMARK_MODE else modCfg["training"].get("total_steps", 32000)
BASE_LR     = modCfg["training"].get("base_lr", 0.0015)
MIN_LR      = modCfg["training"].get("min_lr", 0.0001)
BATCH_QA    = modCfg["training"].get("batch_qa", 16)
BATCH_ST    = modCfg["training"].get("batch_st", 16)

optimizer = torch.optim.AdamW(rawModel.parameters(), lr=BASE_LR, weight_decay=modCfg["training"]["weight_decay"])

def getLr(step):
    if step < WARMUP: return BASE_LR * (step + 1) / WARMUP
    return MIN_LR + 0.5 * (1.0 + math.cos(math.pi * (step - WARMUP) / (TOTAL_STEPS - WARMUP))) * (BASE_LR - MIN_LR)

def getBatch():
    qaX_list, qaY_list = [], []
    for _ in range(BATCH_QA):
        task = generateAtomicSample(logCfg)
        toks = tokenizeStr(task)
        if len(toks) > BLOCK_SIZE: toks = toks[:BLOCK_SIZE]
        x = toks + [padId] * (BLOCK_SIZE - len(toks))
        y = [-100] * BLOCK_SIZE
        
        # Loss applies strictly to the direct answer tokens following [a]
        ansStart = next((idx for idx, t in enumerate(toks) if t == aTokenId), -1)
        if ansStart != -1:
            for idx in range(ansStart, len(toks) - 1): 
                y[idx] = toks[idx + 1]
                
        qaX_list.append(x)
        qaY_list.append(y)
        
    xQa = torch.tensor(qaX_list, dtype=torch.long, device=DEVICE)
    yQa = torch.tensor(qaY_list, dtype=torch.long, device=DEVICE)
    
    stIdx = torch.randint(numStChunks, (BATCH_ST,))
    xSt = stXData[stIdx]
    ySt = stYData[stIdx]
    
    return xQa, yQa, xSt, ySt

print(f"\n[*] Training configuration: {TOTAL_STEPS} steps | Batch: {BATCH_QA} QA + {BATCH_ST} Story | Equal 50/50 Loss")
rawModel.train()
tStart = time.time()

try:
    for step in range(TOTAL_STEPS + 1):
        for p in optimizer.param_groups: p["lr"] = getLr(step)
        xQa, yQa, xSt, ySt = getBatch()
        logitsAll = rawModel(torch.cat([xQa, xSt], dim=0))
        
        lossQa = F.cross_entropy(logitsAll[:BATCH_QA].view(-1, VOCAB_SIZE), yQa.view(-1), ignore_index=-100)
        lossSt = F.cross_entropy(logitsAll[BATCH_QA:].view(-1, VOCAB_SIZE), ySt.view(-1))
        
        # Exact 50/50 balance between QA reasoning and Story language modeling
        loss = (0.50 * lossQa) + (0.50 * lossSt)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rawModel.parameters(), modCfg["training"]["grad_clip"])
        optimizer.step()

        if step % 500 == 0:
            elapsed = time.time() - tStart
            etaMin = (TOTAL_STEPS - step) * (elapsed / max(step, 1)) / 60
            print(f"Step {step:5d}/{TOTAL_STEPS} | Total: {loss.item():.4f} | QA: {lossQa.item():.4f} | Story: {lossSt.item():.4f} | ETA: {etaMin:.1f}m", flush=True)
except KeyboardInterrupt:
    print("\n[!] Training paused. Saving current checkpoint...")

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