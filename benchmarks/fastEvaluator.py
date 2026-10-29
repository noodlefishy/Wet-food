import os, sys, json, re, time, struct, torch, numpy as np, torch.nn as nn
from torch.nn import functional as F

projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, projectRoot)
with open(os.path.join(projectRoot, "config", "model.json")) as f: modCfg = json.load(f)
from models.local import vocab
from chunks import preprocess_chunks, decode_token, TOKEN_REGEX

VOCAB = vocab.VOCAB
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}
VOCAB_SIZE   = modCfg["architecture"]["vocab_size"]
EMBED_DIM    = modCfg["architecture"]["embed_dim"]
BLOCK_SIZE   = modCfg["architecture"]["block_size"]
NUM_HEADS    = modCfg["architecture"]["num_heads"]
NUM_LAYERS   = modCfg["architecture"]["num_layers"]
MEOW_LAYERS  = modCfg["architecture"].get("meow_layers", NUM_LAYERS // 2)
MIAOU_LAYERS = modCfg["architecture"].get("miaou_layers", NUM_LAYERS - MEOW_LAYERS)
HEAD_DIM     = EMBED_DIM // NUM_HEADS

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def precomputeRope(headDim, maxLen=BLOCK_SIZE):
    inv = 1.0 / (10000.0 ** (torch.arange(0, headDim, 2).float() / headDim))
    f = torch.outer(torch.arange(maxLen).float(), inv)
    return f.cos(), f.sin()

cosC, sinC = precomputeRope(HEAD_DIM, BLOCK_SIZE)

def applyRope(x, cos, sin):
    T = x.shape[2]
    c = cos[:T].unsqueeze(0).unsqueeze(1).to(x.device)
    s = sin[:T].unsqueeze(0).unsqueeze(1).to(x.device)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out

class RoPE_MQA(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkvProj = nn.Linear(EMBED_DIM, EMBED_DIM + 2 * HEAD_DIM)
        self.outProj = nn.Linear(EMBED_DIM, EMBED_DIM)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkvProj(x)
        q = qkv[:, :, :C].view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = qkv[:, :, C:C+HEAD_DIM].view(B, T, 1, HEAD_DIM).transpose(1, 2)
        v = qkv[:, :, C+HEAD_DIM:].view(B, T, 1, HEAD_DIM).transpose(1, 2)
        q = applyRope(q, cosC, sinC)
        k = applyRope(k, cosC, sinC).expand(-1, NUM_HEADS, -1, -1)
        v = v.expand(-1, NUM_HEADS, -1, -1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.outProj(out.transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.sa  = RoPE_MQA()
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

model = BabyStories().to(device)

def loadParam(f, tensor, quantized=True):
    if tensor.ndim == 2 and quantized:
        rows, cols = tensor.shape
        data = np.empty((rows, cols), dtype=np.float32)
        for r in range(rows):
            s = struct.unpack('f', f.read(4))[0]
            data[r] = np.frombuffer(f.read(cols), dtype=np.int8).astype(np.float32) * s
        tensor.data.copy_(torch.from_numpy(data))
    else:
        rawBuf = np.frombuffer(f.read(tensor.numel() * 4), dtype=np.float32).copy()
        tensor.data.copy_(torch.from_numpy(rawBuf).view_as(tensor))

p1 = os.path.join(projectRoot, "models", "local", "pico1 weights.model")
p2 = os.path.join(projectRoot, "models", "local", "pico2 weights.model")

with open(p1, "rb") as f:
    loadParam(f, model.tokEmb.weight)
    for i in range(MEOW_LAYERS):
        for p in model.layers[i].parameters(): loadParam(f, p)

with open(p2, "rb") as f:
    for i in range(MEOW_LAYERS, NUM_LAYERS):
        for p in model.layers[i].parameters(): loadParam(f, p)
    for p in model.lnF.parameters(): loadParam(f, p)
    loadParam(f, model.lmHead.weight)
    loadParam(f, model.lmHead.bias)

model.eval()
print(f"[✓] Model loaded ({NUM_LAYERS}L RoPE) on {device.upper()}\n")

suitePath = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else os.path.join(projectRoot, "config", "stressTest.json")
if not os.path.exists(suitePath): suitePath = os.path.join(projectRoot, "config", "questions.json")

print(f"[*] Loading evaluation suite: {suitePath}")
with open(suitePath) as f: rawData = json.load(f)

aId, padId, dotId, eosId = WORD_TO_ID.get("[a]", 9), WORD_TO_ID.get("<pad>", 1), WORD_TO_ID.get(".", 3), WORD_TO_ID.get("<eos>", 2)

def tokenize(s):
    p = preprocess_chunks(s)
    return [WORD_TO_ID.get(w, 0) for w in re.findall(TOKEN_REGEX, p)]

allTests = []
for cat, items in rawData.items():
    for item in items:
        if item.get("target") is not None:
            allTests.append((cat, item["q"], str(item["target"])))

total = len(allTests)
print(f"[*] Evaluating {total:,} QA tests across {len(rawData)} categories with Micro-CoT validation...\n")

catStats, totalPassed, failures = {}, 0, []
t0 = time.time()
BATCH = 512

def extract_answer(text):
    t = preprocess_chunks(text)
    if ":" in t:
        return t.split(":")[-1].strip().rstrip(".").split()[0]
    words = t.strip().rstrip(".").split()
    return words[-1] if words else ""

with torch.no_grad():
    for i in range(0, total, BATCH):
        batch = allTests[i:i+BATCH]
        prompts = [tokenize(q) for _, q, _ in batch]
        
        # Determine max prompt length in this batch
        lens = [len(p) for p in prompts]
        maxLen = max(lens)
        
        padded = [p + [padId] * (BLOCK_SIZE - len(p)) for p in prompts]
        x = torch.tensor(padded, dtype=torch.long, device=device)
        
        # Autoregressive Micro-CoT generation (up to 5 tokens)
        genTokens = [[] for _ in batch]
        active = [True] * len(batch)
        
        for step in range(5):
            logits = model(x)
            for j in range(len(batch)):
                if not active[j]: continue
                pos = lens[j] + step - 1
                if pos >= BLOCK_SIZE:
                    active[j] = False
                    continue
                predId = logits[j, pos].argmax().item()
                genTokens[j].append(VOCAB[predId])
                if predId in (dotId, eosId): active[j] = False
                if pos + 1 < BLOCK_SIZE: x[j, pos + 1] = predId
            if not any(active): break

        for j, (cat, q, target) in enumerate(batch):
            genStr = " ".join(genTokens[j])
            predAns = extract_answer(genStr)
            expectedAns = extract_answer(target)
            
            ok = (predAns == expectedAns)
            if cat not in catStats: catStats[cat] = [0, 0]
            catStats[cat][1] += 1
            if ok:
                catStats[cat][0] += 1
                totalPassed += 1
            else:
                if len(failures) < 50:
                    failures.append((cat, q, target, genStr))

        done = min(i + BATCH, total)
        pct = (totalPassed / done * 100) if done else 0
        elapsed = time.time() - t0
        eta = (total - done) * (elapsed / done) if done else 0
        sys.stdout.write(f"\r  [Evaluator] Progress: {done:>7,}/{total:,} [{done/total*100:5.1f}%] | Acc: {pct:5.1f}% | Elapsed: {elapsed:4.1f}s | ETA: {eta:4.1f}s")
        sys.stdout.flush()

elapsed = time.time() - t0
print(f"\n\n{'='*65}")
print(f" FINAL MICRO-COT ACCURACY: {totalPassed:,}/{total:,} = {totalPassed/total*100:.1f}%  ({elapsed:.2f}s)")
print(f"{'='*65}")

for cat in sorted(catStats.keys()):
    passed, n = catStats[cat]
    pct = (passed / n * 100) if n else 0
    mark = "✓" if pct >= 90 else ("~" if pct >= 70 else "✗")
    print(f"  [{mark}] {cat:<38} {passed:>6}/{n:<6} {pct:5.1f}%")

if "--failures" in sys.argv and failures:
    print(f"\n{'='*65}")
    print(" SAMPLE MISPREDICTIONS (First 15)")
    print(f"{'='*65}")
    for cat, q, tgt, pred in failures[:15]:
        print(f"  [{cat}]\n    Q: {q}\n    Expected: {tgt!r}\n    Generated CoT: {pred!r}\n")