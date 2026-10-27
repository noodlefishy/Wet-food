import os, sys, json, re, time, torch, numpy as np, torch.nn as nn
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
with open(os.path.join(project_root, "config", "model.json"), "r") as f: mod_cfg = json.load(f)
from models.local import vocab
VOCAB = vocab.VOCAB
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}
EMBED_DIM = mod_cfg["architecture"]["embed_dim"]
device = "mps" if torch.backends.mps.is_available() else "cpu"

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(EMBED_DIM), nn.LayerNorm(EMBED_DIM)
        self.sa = nn.MultiheadAttention(EMBED_DIM, 4, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(EMBED_DIM, 4*EMBED_DIM), nn.ReLU(), nn.Linear(4*EMBED_DIM, EMBED_DIM))
    def forward(self, x):
        norm_x = self.ln1(x)
        attn_out, _ = self.sa(norm_x, norm_x, norm_x, attn_mask=torch.triu(torch.ones(x.size(1), x.size(1), device=x.device)*float("-inf"), 1), is_causal=True)
        return x + attn_out + self.mlp(self.ln2(x + attn_out))

class BabyStories(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb, self.pos_emb = nn.Embedding(1024, EMBED_DIM), nn.Embedding(64, EMBED_DIM)
        self.layers = nn.ModuleList([Block() for _ in range(6)])
        self.ln_f, self.lm_head = nn.LayerNorm(EMBED_DIM), nn.Linear(EMBED_DIM, 1024)
    def forward(self, idx):
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(idx.size(1), device=idx.device))
        for layer in self.layers: x = layer(x)
        return self.lm_head(self.ln_f(x))

model = BabyStories().to(device)
def load_param(f, tensor):
    tensor.data.copy_(torch.from_numpy(np.fromfile(f, dtype=np.float32, count=tensor.numel())).view_as(tensor))

with open(os.path.join(project_root, "models", "local", "pico1 weights.model"), "rb") as f1:
    load_param(f1, model.tok_emb.weight); load_param(f1, model.pos_emb.weight)
    for i in [0, 1, 2]:
        for p in model.layers[i].parameters(): load_param(f1, p)
with open(os.path.join(project_root, "models", "local", "pico2 weights.model"), "rb") as f2:
    for i in [3, 4, 5]:
        for p in model.layers[i].parameters(): load_param(f2, p)
    for p in model.ln_f.parameters(): load_param(f2, p)
    load_param(f2, model.lm_head.weight); load_param(f2, model.lm_head.bias)
model.eval()

with open(sys.argv[1] if len(sys.argv) > 1 else "config/stress_test.json", "r") as f:
    raw_data = json.load(f)
all_tests = [(cat, item["q"], item["target"]) for cat, items in raw_data.items() for item in items]

cat_stats, total_passed = {}, 0
with torch.no_grad():
    for i in range(0, len(all_tests), 512):
        batch = all_tests[i : i + 512]
        toks = [[WORD_TO_ID.get(w, 0) for w in re.findall(r"\[[cqa]\]|<eos>|<unk>|<pad>|\w+|[.]|[?]|[:]", (p + " [a]").lower())] for _, p, _ in batch]
        max_len = max(len(t) for t in toks)
        x_tensor = torch.tensor([t + [WORD_TO_ID["<pad>"]] * (max_len - len(t)) for t in toks], dtype=torch.long, device=device)
        logits = model(x_tensor)
        logits[:, :, WORD_TO_ID["<unk>"]] = logits[:, :, WORD_TO_ID["<pad>"]] = -1e9
        
        for b_idx, (cat, _, target) in enumerate(batch):
            if cat not in cat_stats: cat_stats[cat] = {"passed": 0, "total": 0}
            cat_stats[cat]["total"] += 1
            pred_word = VOCAB[logits[b_idx, len(toks[b_idx]) - 1].argmax().item()]
            expected = [w for w in re.findall(r"\w+", target.lower())]
            
            if len(expected) == 1: is_correct = (pred_word == expected[0])
            else:
                gen, curr = [pred_word], toks[b_idx] + [logits[b_idx, len(toks[b_idx]) - 1].argmax().item()]
                while len(gen) < len(expected) + 2:
                    out = model(torch.tensor([curr[-64:]], dtype=torch.long, device=device))[0, -1]
                    out[WORD_TO_ID["<unk>"]] = out[WORD_TO_ID["<pad>"]] = -1e9
                    nw = VOCAB[out.argmax().item()]
                    if nw in ("<eos>", "<pad>", "."): break
                    gen.append(nw); curr.append(out.argmax().item())
                is_correct = (gen == expected)
            if is_correct: cat_stats[cat]["passed"] += 1; total_passed += 1
        print(f"\rEvaluating: {min(i+512, len(all_tests))}/{len(all_tests)}", end="")

print("\n" + "="*76 + "\nSCORECARD\n" + "="*76)
for cat, s in cat_stats.items(): print(f"{cat:<34} : {s['passed']:05d}/{s['total']:05d} [{('#'*int((s['passed']/s['total'])*10)).ljust(10, '-')}] {(s['passed']/s['total'])*100:5.1f}%")