import os
import sys
import json
import re
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

print("=" * 76)
print("     PARALLEL BATCH EVALUATOR     ")
print("=" * 76)

# 1. Load Configurations & Vocab
with open("config/model.json", "r") as f:
    mod_cfg = json.load(f)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.local import vocab
VOCAB = vocab.VOCAB
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}

VOCAB_SIZE = mod_cfg["architecture"]["vocab_size"]
BLOCK_SIZE = mod_cfg["architecture"]["block_size"]
EMBED_DIM  = mod_cfg["architecture"]["embed_dim"]
NUM_HEADS  = mod_cfg["architecture"]["num_heads"]
NUM_LAYERS = mod_cfg["architecture"]["num_layers"]

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Accelerating evaluation on: {device.upper()}")

# 2. Define Identical Architecture
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.sa = nn.MultiheadAttention(EMBED_DIM, NUM_HEADS, batch_first=True)
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.mlp = nn.Sequential(nn.Linear(EMBED_DIM, 4 * EMBED_DIM), nn.ReLU(), nn.Linear(4 * EMBED_DIM, EMBED_DIM))

    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float("-inf"), diagonal=1)
        norm_x = self.ln1(x)
        attn_out, _ = self.sa(norm_x, norm_x, norm_x, attn_mask=mask, is_causal=True)
        return x + attn_out + self.mlp(self.ln2(x + attn_out))

class BabyStories(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.layers = nn.ModuleList([Block() for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_f(x))

model = BabyStories().to(device)

# 3. Load Raw Binary Weights Exported for Pico
def load_param(f, tensor):
    num = tensor.numel()
    data = np.fromfile(f, dtype=np.float32, count=num)
    tensor.data.copy_(torch.from_numpy(data).view_as(tensor))

p1_path = os.path.join("models", "local", "pico1 weights.model")
p2_path = os.path.join("models", "local", "pico2 weights.model")

print(f"Loading binary weights from:\n  {p1_path}\n  {p2_path}")

with open(p1_path, "rb") as f1:
    load_param(f1, model.tok_emb.weight)
    load_param(f1, model.pos_emb.weight)
    for i in [0, 1]:
        for p in model.layers[i].parameters():
            load_param(f1, p)

with open(p2_path, "rb") as f2:
    for i in [2, 3]:
        for p in model.layers[i].parameters():
            load_param(f2, p)
    for p in model.ln_f.parameters():
        load_param(f2, p)
    load_param(f2, model.lm_head.weight)
    load_param(f2, model.lm_head.bias)

model.eval()

# 4. Load the 94,248 Stress Test Dataset
stress_file = "config/stress_test.json"
if not os.path.exists(stress_file):
    print(f"Error: {stress_file} not found. Run 'python benchmarks/stress_test.py' first!")
    sys.exit(1)

with open(stress_file, "r") as f:
    data = json.load(f)

# Flatten dataset into structured list
all_tests = []
for cat, items in data.items():
    for item in items:
        all_tests.append((cat, item["q"], item["target"]))

total_questions = len(all_tests)
print(f"Loaded {total_questions} questions across {len(data)} categories.\n")

# 5. Tokenization & Fast Batch Processing
def tokenize_prompt(prompt_text):
    words = re.findall(r"\[[cqa]\]|<eos>|<unk>|<pad>|\w+|[.]|[?]|[:]", prompt_text.lower())
    return [WORD_TO_ID.get(w, 0) for w in words]

BATCH_SIZE = 512
pad_id = WORD_TO_ID["<pad>"]

cat_stats = {cat: {"passed": 0, "total": 0} for cat in data.keys()}
total_passed = 0

t_start = time.perf_counter()

with torch.no_grad():
    for i in range(0, total_questions, BATCH_SIZE):
        batch = all_tests[i : i + BATCH_SIZE]
        
        # Tokenize and find lengths
        tokenized_batch = [tokenize_prompt(item[1]) for item in batch]
        seq_lens = [len(toks) for toks in tokenized_batch]
        max_len = max(seq_lens)
        
        # Build tensor on GPU/Mac RAM
        batch_tensor = [toks + [pad_id] * (max_len - len(toks)) for toks in tokenized_batch]
        x_tensor = torch.tensor(batch_tensor, dtype=torch.long, device=device)
        
        # Single vectorized forward pass for all 512 questions
        logits = model(x_tensor)
        
        # Evaluate predictions
        for b_idx, (cat, prompt, target) in enumerate(batch):
            ans_pos = seq_lens[b_idx] - 1  # Position of [a]
            
            # Predict token
            pred_tok_id = logits[b_idx, ans_pos].argmax().item()
            pred_word = VOCAB[pred_tok_id]
            
            # Ground truth expected first word
            expected_word = target.strip().split()[0].lower()
            
            cat_stats[cat]["total"] += 1
            if pred_word == expected_word:
                cat_stats[cat]["passed"] += 1
                total_passed += 1

        # Live Progress
        processed = min(i + BATCH_SIZE, total_questions)
        pct = (processed / total_questions) * 100
        elapsed = time.perf_counter() - t_start
        qps = processed / (elapsed + 1e-5)
        print(f"\rEvaluating: {processed:5d}/{total_questions} ({pct:5.1f}%) | Speed: {qps:5.0f} tests/sec", end="", flush=True)

t_total = time.perf_counter() - t_start
print(f"\nCompleted in {t_total:.2f} seconds! ({total_questions / t_total:.1f} tests/sec)\n")

# 6. Print the Exhaustive Category Scorecard
print("=" * 76)
print("                   EXHAUSTIVE 94,248-QUESTION SCORECARD                ")
print("=" * 76)

for cat, s in cat_stats.items():
    if s["total"] > 0:
        pct = (s["passed"] / s["total"]) * 100
        bar = ("#" * int(pct // 10)).ljust(10, "-")
        print(f"{cat:<34} : {s['passed']:05d}/{s['total']:05d} [{bar}] {pct:5.1f}%")

final_pct = (total_passed / total_questions) * 100
final_bar = ("#" * int(final_pct // 10)).ljust(10, "-")
print("-" * 76)
print(f"TOTAL EXHAUSTIVE REASONING SCORE   : {total_passed:05d}/{total_questions:05d} [{final_bar}] {final_pct:5.1f}%")
print("=" * 76)