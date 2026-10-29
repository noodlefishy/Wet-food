"""
Out-of-Distribution (OOD) Generalization Evaluator for picoAI.
Tests the model on completely held-out entities, novel syntax, and reasoning depth
to rigorously verify if it has learned reasoning abstractions or memorized specific words.
"""
import os, sys, json, re, time, struct, torch, numpy as np, torch.nn as nn
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

with open(p1, "rb") as f:
    load_param(f, model.tok_emb.weight)
    for i in range(MEOW_LAYERS):
        for p in model.layers[i].parameters(): load_param(f, p)

with open(p2, "rb") as f:
    for i in range(MEOW_LAYERS, NUM_LAYERS):
        for p in model.layers[i].parameters(): load_param(f, p)
    for p in model.ln_f.parameters(): load_param(f, p)
    load_param(f, model.lm_head.weight)
    load_param(f, model.lm_head.bias)

model.eval()
print(f"[✓] Model loaded on {device} for Out-of-Distribution (OOD) Generalization Suite\n")

a_id = WORD_TO_ID.get("[a]", 9)
pad_id = WORD_TO_ID.get("<pad>", 1)

def tokenize(s):
    return [WORD_TO_ID.get(w, 0) for w in re.findall(r"\[[cqa]\]|<eos>|<unk>|<pad>|\w+|[.?:]", s.lower())]

# =====================================================================
# OOD TEST BATTERIES
# =====================================================================
# 1. Novel Entities from TinyStories vocabulary (never used in world.json logic templates!)
#    e.g. actors: mia, max, timmy, emma, daisy
#    locations: woods, forest, lake, school, beach
#    items: toy, fish, bread, flower, rock
ood_tests = {
    "OOD-1: Novel Entities Tracking (1-hop)": [
        {"q": "[c] mia went to the forest . [q] where is mia ? [a]", "target": "forest"},
        {"q": "[c] max went to the beach . [q] where is max ? [a]", "target": "beach"},
        {"q": "[c] timmy went to the lake . [q] where is timmy ? [a]", "target": "lake"},
        {"q": "[c] daisy moved to the woods . [q] where is daisy ? [a]", "target": "woods"},
        {"q": "[c] emma arrived at the school . [q] where is emma ? [a]", "target": "school"},
    ],
    "OOD-2: Novel Entities Multi-Hop (2-hop)": [
        {"q": "[c] mia went to the lake . max went to the forest . mia went to the beach . [q] where is mia ? [a]", "target": "beach"},
        {"q": "[c] timmy went to the woods . daisy went to the beach . timmy went to the school . [q] where is timmy ? [a]", "target": "school"},
        {"q": "[c] emma went to the forest . emma went to the lake . [q] where is emma ? [a]", "target": "lake"},
        {"q": "[c] max went to the school . max went to the woods . [q] where is max ? [a]", "target": "woods"},
    ],
    "OOD-3: Novel Entities Polar Yes/No": [
        {"q": "[c] mia went to the lake . [q] is mia in the lake ? [a]", "target": "yes"},
        {"q": "[c] mia went to the lake . [q] is mia in the forest ? [a]", "target": "no"},
        {"q": "[c] max went to the beach . [q] is max in the beach ? [a]", "target": "yes"},
        {"q": "[c] max went to the beach . [q] is max in the woods ? [a]", "target": "no"},
    ],
    "OOD-4: 4-Hop Depth Extrapolation (Longer than train)": [
        {"q": "[c] alice went to the kitchen . alice went to the garden . alice went to the park . alice went to the room . [q] where is alice ? [a]", "target": "room"},
        {"q": "[c] bob went to the park . bob went to the room . bob went to the kitchen . bob went to the garden . [q] where is bob ? [a]", "target": "garden"},
        {"q": "[c] felix went to the garden . felix went to the park . felix went to the room . felix went to the kitchen . [q] where is felix ? [a]", "target": "kitchen"},
    ],
    "OOD-5: Novel Syntax / Passive Voice": [
        {"q": "[c] the ball was taken by alice . [q] who got the ball ? [a]", "target": "alice"},
        {"q": "[c] the cup was given to bob . [q] who got the cup ? [a]", "target": "bob"},
        {"q": "[c] the box was found by felix . [q] who got the box ? [a]", "target": "felix"},
    ]
}

total_tests = sum(len(items) for items in ood_tests.values())
print(f"[*] Running {total_tests} Out-of-Distribution generalization probes...\n")

total_passed = 0
with torch.no_grad():
    for suite_name, tests in ood_tests.items():
        input_ids, a_positions, metas = [], [], []
        for item in tests:
            toks = tokenize(item["q"])
            try: a_pos = next(j for j in range(len(toks)-1, -1, -1) if toks[j] == a_id)
            except StopIteration: a_pos = len(toks) - 1
            padded = toks[:a_pos+1] + [pad_id] * (BLOCK_SIZE - len(toks[:a_pos+1]))
            input_ids.append(padded[:BLOCK_SIZE])
            a_positions.append(min(a_pos, BLOCK_SIZE - 1))
            metas.append((item["q"], item["target"]))

        x = torch.tensor(input_ids, dtype=torch.long, device=device)
        logits = model(x)
        
        passed_suite = 0
        print(f"--- {suite_name} ---")
        for j, (q_str, target) in enumerate(metas):
            pred_id = logits[j, a_positions[j]].argmax().item()
            pred = VOCAB[pred_id] if pred_id < len(VOCAB) else "<unk>"
            tgt = target.strip().rstrip(".").split()[0]
            ok = pred == tgt
            if ok:
                passed_suite += 1
                total_passed += 1
                status = "✓ PASS"
            else:
                status = f"✗ FAIL (got: '{pred}', expected: '{tgt}')"
            print(f"  [{status}] {q_str}")
        pct = passed_suite / len(tests) * 100
        print(f"  Score: {passed_suite}/{len(tests)} ({pct:.1f}%)\n")

print("=" * 60)
print(f"TOTAL OOD GENERALIZATION SCORE: {total_passed}/{total_tests} ({total_passed/total_tests*100:.1f}%)")
print("=" * 60)
