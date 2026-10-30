import os, sys, json, re, time, struct, torch, numpy as np, torch.nn as nn
from torch.nn import functional as F

projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, projectRoot)
with open(os.path.join(projectRoot, "config", "model.json")) as f: modCfg = json.load(f)
from models.local import vocab
VOCAB = vocab.VOCAB
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}

VOCAB_SIZE   = modCfg["architecture"]["vocab_size"]
EMBED_DIM    = modCfg["architecture"]["embed_dim"]
BLOCK_SIZE   = modCfg["architecture"]["block_size"]
NUM_HEADS    = modCfg["architecture"]["num_heads"]
NUM_LAYERS   = modCfg["architecture"]["num_layers"]
MEOW_LAYERS  = 3
MIAOU_LAYERS = 3
HEAD_DIM     = EMBED_DIM // NUM_HEADS
device       = "mps" if torch.backends.mps.is_available() else "cpu"

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

class RoPE_MHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkvProj = nn.Linear(EMBED_DIM, 3 * EMBED_DIM)
        self.outProj  = nn.Linear(EMBED_DIM, EMBED_DIM)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkvProj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        q = applyRope(q, cosC, sinC)
        k = applyRope(k, cosC, sinC)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.outProj(out.transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.sa  = RoPE_MHA()
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.mlp = nn.Sequential(nn.Linear(EMBED_DIM, 4*EMBED_DIM), nn.ReLU(), nn.Linear(4*EMBED_DIM, EMBED_DIM))
    def forward(self, x):
        return x + self.mlp(self.ln2(x + self.sa(self.ln1(x))))

class BabyStories(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokEmb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.layers  = nn.ModuleList([Block() for _ in range(NUM_LAYERS)])
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
        tensor.data.copy_(torch.from_numpy(np.frombuffer(f.read(tensor.numel() * 4), dtype=np.float32)).view_as(tensor))

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

aId, padId = WORD_TO_ID.get("[a]", 9), WORD_TO_ID.get("<pad>", 1)

def tokenize(s):
    return [WORD_TO_ID.get(w, 0) for w in re.findall(r"\[[cqa]\]|<eos>|<unk>|<pad>|\w+|[.?:]", s.lower())]

def evalSuite(tests, desc):
    total = len(tests)
    passed = 0
    t0 = time.time()
    for cat, q, target in tests:
        toks = tokenize(q)
        try: aPos = next(j for j in range(len(toks)-1, -1, -1) if toks[j] == aId)
        except StopIteration:
            toks.append(aId); aPos = len(toks) - 1
        prompt = toks[:aPos+1]
        padded = prompt + [padId] * (BLOCK_SIZE - len(prompt))
        x = torch.tensor([padded[:BLOCK_SIZE]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
            predId = logits[0, min(aPos, BLOCK_SIZE - 1)].argmax().item()
            pred = VOCAB[predId] if predId < len(VOCAB) else "<unk>"
            tgt = target.strip().rstrip(".").split()[0]
            if pred == tgt: passed += 1
    pct = (passed / total * 100) if total else 0
    print(f"  {desc:<45}: {passed:>4}/{total:<4} ({pct:5.1f}%) in {time.time()-t0:.2f}s")
    return passed, total

print("=" * 65)
print("     OUT-OF-DISTRIBUTION (OOD) GENERALIZATION BENCHMARK")
print("=" * 65)

# 1. Unseen Entities Probe
PROBE_ENTITIES = [
    ("Unseen Entity (Milo/Barn)", "[c] milo went to the barn . [q] where is milo ? [a]", "barn"),
    ("Unseen Entity (Cleo/Attic)", "[c] cleo went to the attic . [q] where is cleo ? [a]", "attic"),
    ("Unseen Entity (Jasper/Cellar)", "[c] jasper went to the cellar . [q] where is jasper ? [a]", "cellar"),
    ("Unseen Entity (Ziggy/Roof)", "[c] ziggy went to the roof . [q] where is ziggy ? [a]", "roof")
]
evalSuite([(cat, q, tgt) for cat, q, tgt in PROBE_ENTITIES], "1. Novel Entity Grounding")

# 2. Syntax Permutation
PROBE_SYNTAX = [
    ("Passive Voice", "[c] the red ball was taken by alice . [q] who has the red ball ? [a]", "alice"),
    ("Inverted Clause", "[c] to the kitchen went bob . [q] where is bob ? [a]", "kitchen"),
    ("Compound Subject", "[c] felix and luna were in the forest . [q] where is felix ? [a]", "forest")
]
evalSuite([(cat, q, tgt) for cat, q, tgt in PROBE_SYNTAX], "2. Novel Syntax Generalization")

# 3. Arithmetic Composition
PROBE_MATH = [
    ("3-Item Addition", "[c] alice has one cup . bob gave alice two cups . [q] how many items does alice have ? [a]", "three"),
    ("Zero Remainder", "[c] bob has two balls . bob gave two balls to felix . [q] how many items does bob have ? [a]", "none"),
    ("Single Transfer", "[c] felix has three books . felix gave one book to luna . [q] how many items does felix have ? [a]", "two")
]
evalSuite([(cat, q, tgt) for cat, q, tgt in PROBE_MATH], "3. Multi-hop Arithmetic Probe")
print("=" * 65)
