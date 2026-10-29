import os, sys, json, re, time, struct, math, torch, numpy as np, torch.nn as nn
from torch.nn import functional as F

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
with open(os.path.join(project_root, "config", "model.json")) as f: mod_cfg = json.load(f)
from models.local import vocab
VOCAB = vocab.VOCAB
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}

VOCAB_SIZE   = mod_cfg["architecture"]["vocab_size"]
EMBED_DIM    = mod_cfg["architecture"]["embed_dim"]
BLOCK_SIZE   = mod_cfg["architecture"]["block_size"]
NUM_HEADS    = mod_cfg["architecture"]["num_heads"]
NUM_LAYERS   = mod_cfg["architecture"]["num_layers"]
MEOW_LAYERS  = mod_cfg["architecture"]["meow_layers"]
MIAOU_LAYERS = mod_cfg["architecture"]["miaou_layers"]
HEAD_DIM     = EMBED_DIM // NUM_HEADS
device = "mps" if torch.backends.mps.is_available() else "cpu"

def precompute_rope(head_dim, max_len=BLOCK_SIZE):
    inv = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    f = torch.outer(t, inv)
    return f.cos(), f.sin()

cos_c, sin_c = precompute_rope(HEAD_DIM, BLOCK_SIZE)

def apply_rope(x, cos, sin):
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
        self.qkv_proj = nn.Linear(EMBED_DIM, EMBED_DIM + 2 * HEAD_DIM)
        self.out_proj  = nn.Linear(EMBED_DIM, EMBED_DIM)
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q = qkv[:, :, :C].view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = qkv[:, :, C:C+HEAD_DIM].view(B, T, 1, HEAD_DIM).transpose(1, 2)
        v = qkv[:, :, C+HEAD_DIM:].view(B, T, 1, HEAD_DIM).transpose(1, 2)
        q = apply_rope(q, cos_c, sin_c)
        k = apply_rope(k, cos_c, sin_c)
        k = k.expand(-1, NUM_HEADS, -1, -1)
        v = v.expand(-1, NUM_HEADS, -1, -1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.sa  = RoPE_MQA()
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.mlp = nn.Sequential(nn.Linear(EMBED_DIM, 4*EMBED_DIM), nn.ReLU(), nn.Linear(4*EMBED_DIM, EMBED_DIM))
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class BabyStories(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.layers  = nn.ModuleList([Block() for _ in range(NUM_LAYERS)])
        self.ln_f    = nn.LayerNorm(EMBED_DIM)
        self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE)
    def forward(self, idx):
        x = self.tok_emb(idx)
        for layer in self.layers: x = layer(x)
        return self.lm_head(self.ln_f(x))

model = BabyStories().to(device)

def load_param(f, tensor, quantized=True):
    if tensor.ndim == 2 and quantized:
        rows, cols = tensor.shape
        data = np.empty((rows, cols), dtype=np.float32)
        for r in range(rows):
            s = struct.unpack('f', f.read(4))[0]
            data[r] = np.frombuffer(f.read(cols), dtype=np.int8).astype(np.float32) * s
        tensor.data.copy_(torch.from_numpy(data))
    else:
        tensor.data.copy_(torch.from_numpy(np.frombuffer(f.read(tensor.numel() * 4), dtype=np.float32)).view_as(tensor))

p1 = os.path.join(project_root, "models", "local", "pico1 weights.model")
p2 = os.path.join(project_root, "models", "local", "pico2 weights.model")

print(f"[*] Loading Meow weights ({MEOW_LAYERS}L) from pico1...")
with open(p1, "rb") as f:
    load_param(f, model.tok_emb.weight)
    for i in range(MEOW_LAYERS):
        for p in model.layers[i].parameters(): load_param(f, p)

print(f"[*] Loading Miaou weights ({MIAOU_LAYERS}L) from pico2...")
with open(p2, "rb") as f:
    for i in range(MEOW_LAYERS, NUM_LAYERS):
        for p in model.layers[i].parameters(): load_param(f, p)
    for p in model.ln_f.parameters(): load_param(f, p)
    load_param(f, model.lm_head.weight)
    load_param(f, model.lm_head.bias)

model.eval()
print(f"[✓] Model loaded ({NUM_LAYERS}L RoPE, no pos_emb) on {device}\n")

suite_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(project_root, "config", "stress_test.json")
with open(suite_path) as f: raw_data = json.load(f)

a_id = WORD_TO_ID.get("[a]", 9)
pad_id = WORD_TO_ID.get("<pad>", 1)

def tokenize(s):
    return [WORD_TO_ID.get(w, 0) for w in re.findall(r"\[[cqa]\]|<eos>|<unk>|<pad>|\w+|[.?:]", s.lower())]

all_tests = [(cat, item["q"], item["target"]) for cat, items in raw_data.items() for item in items]
total = len(all_tests)
print(f"[*] Running {total:,} tests on {len(raw_data)} categories...\n")

cat_stats, total_passed, failures = {}, 0, []
t0 = time.time()

BATCH = 512
with torch.no_grad():
    for i in range(0, total, BATCH):
        batch = all_tests[i:i+BATCH]
        input_ids, answer_positions, metas = [], [], []
        for cat, q, target in batch:
            toks = tokenize(q)
            try:
                a_pos = next(j for j in range(len(toks)-1, -1, -1) if toks[j] == a_id)
            except StopIteration:
                toks.append(a_id)
                a_pos = len(toks) - 1
            prompt = toks[:a_pos+1]
            padded = prompt + [pad_id] * (BLOCK_SIZE - len(prompt))
            input_ids.append(padded[:BLOCK_SIZE])
            answer_positions.append(min(a_pos, BLOCK_SIZE - 1))
            metas.append((cat, target))

        x = torch.tensor(input_ids, dtype=torch.long, device=device)
        logits = model(x)

        for j, (cat, target) in enumerate(metas):
            pred_id = logits[j, answer_positions[j]].argmax().item()
            pred = VOCAB[pred_id] if pred_id < len(VOCAB) else "<unk>"
            tgt = target.strip().rstrip(".").split()[0]
            ok = pred == tgt
            if cat not in cat_stats: cat_stats[cat] = [0, 0]
            cat_stats[cat][1] += 1
            if ok:
                cat_stats[cat][0] += 1
                total_passed += 1
            else:
                failures.append((cat, batch[j][1], target, pred))

        done = min(i + BATCH, total)
        pct = total_passed / done * 100
        elapsed = time.time() - t0
        eta = (total - done) * (elapsed / done) if done else 0
        print(f"  {done:>7,}/{total:,}  acc={pct:5.1f}%  eta={eta/60:.1f}m", end="\r")

elapsed = time.time() - t0
print(f"\n\n{'='*60}")
print(f" RESULTS: {total_passed:,}/{total:,} = {total_passed/total*100:.1f}%  ({elapsed:.1f}s)")
print(f"{'='*60}")

perfect, struggling = [], []
for cat in sorted(cat_stats.keys()):
    passed, n = cat_stats[cat]
    pct = passed / n * 100
    mark = "✓" if pct == 100 else ("~" if pct >= 80 else "✗")
    print(f"  [{mark}] {cat:<35} {passed:>6}/{n:<6} {pct:5.1f}%")
    if pct == 100: perfect.append(cat)
    elif pct < 70:  struggling.append(cat)

print(f"\n[✓] Perfect (100%): {', '.join(perfect) or 'none'}")
print(f"[✗] Struggling (<70%): {', '.join(struggling) or 'none'}")

if "--failures" in sys.argv and failures:
    print(f"\n--- Sample Failures (first 20) ---")
    for cat, q, tgt, pred in failures[:20]:
        print(f"  [{cat}] Q: {q!r}  expected={tgt!r}  got={pred!r}")
