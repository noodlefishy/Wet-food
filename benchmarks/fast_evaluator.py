# =====================================================================
# WET FOOD: FAST PARALLEL BATCH EVALUATOR (MULTI-TOKEN & MPS ACCELERATED)
# =====================================================================
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
print("     PARALLEL BATCH EVALUATOR (EXHAUSTIVE HIGH-THROUGHPUT ENGINE)    ")
print("=" * 76)

# 1. Load Configurations & Vocab
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

config_path = os.path.join(project_root, "config", "model.json")
with open(config_path, "r") as f:
    mod_cfg = json.load(f)

from models.local import vocab
VOCAB = vocab.VOCAB
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}

VOCAB_SIZE = mod_cfg["architecture"]["vocab_size"]
BLOCK_SIZE = mod_cfg["architecture"]["block_size"]
EMBED_DIM  = mod_cfg["architecture"]["embed_dim"]
NUM_HEADS  = mod_cfg["architecture"]["num_heads"]
NUM_LAYERS = mod_cfg["architecture"]["num_layers"]

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Hardware Acceleration : {device.upper()}")
print(f"Architecture          : {NUM_LAYERS}L {EMBED_DIM}-D ({NUM_HEADS} Heads)")

# 2. Define Model Architecture
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

# 3. Load Binary Weights from Disk
def load_param(f, tensor):
    num = tensor.numel()
    data = np.fromfile(f, dtype=np.float32, count=num)
    tensor.data.copy_(torch.from_numpy(data).view_as(tensor))

p1_path = os.path.join(project_root, "models", "local", "pico1 weights.model")
p2_path = os.path.join(project_root, "models", "local", "pico2 weights.model")

if not os.path.exists(p1_path) or not os.path.exists(p2_path):
    print(f"Error: Missing binary files:\n  {p1_path}\n  {p2_path}")
    sys.exit(1)

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

# 4. Load Dataset (Accepts CLI argument or defaults to stress_test.json)
if len(sys.argv) > 1:
    dataset_file = sys.argv[1]
else:
    dataset_file = "config/stress_test.json" if os.path.exists("config/stress_test.json") else "config/questions.json"

if not os.path.exists(dataset_file):
    print(f"Error: Dataset '{dataset_file}' not found!")
    sys.exit(1)

with open(dataset_file, "r") as f:
    raw_data = json.load(f)

# Support both dictionary format and list format
all_tests = []
if isinstance(raw_data, dict):
    for cat, items in raw_data.items():
        for item in items:
            all_tests.append((cat, item["q"], item["target"]))
else:
    for item in raw_data:
        all_tests.append((item.get("cat", "Uncategorized"), item["q"], item["target"]))

total_questions = len(all_tests)
print(f"Evaluating Dataset    : {dataset_file} ({total_questions:,} tests)\n")

# 5. Tokenizer & Suffix Auto-Repair
def tokenize_prompt(prompt_text):
    clean_p = prompt_text.strip()
    # Guarantee prompt terminates with '[a]'
    if not clean_p.endswith("[a]"):
        clean_p = clean_p + " [a]"
    words = re.findall(r"\[[cqa]\]|<eos>|<unk>|<pad>|\w+|[.]|[?]|[:]", clean_p.lower())
    return [WORD_TO_ID.get(w, 0) for w in words]

BATCH_SIZE = 512
pad_id = WORD_TO_ID["<pad>"]
unk_id = WORD_TO_ID["<unk>"]

cat_stats = {}
total_passed = 0

t_start = time.perf_counter()

with torch.no_grad():
    for i in range(0, total_questions, BATCH_SIZE):
        batch = all_tests[i : i + BATCH_SIZE]
        
        tokenized_batch = [tokenize_prompt(item[1]) for item in batch]
        seq_lens = [len(toks) for toks in tokenized_batch]
        max_len = max(seq_lens)
        
        # Batch tensor on device
        batch_tensor = [toks + [pad_id] * (max_len - len(toks)) for toks in tokenized_batch]
        x_tensor = torch.tensor(batch_tensor, dtype=torch.long, device=device)
        
        # Forward pass
        logits = model(x_tensor)
        
        # Mask out special tokens across the entire batch
        logits[:, :, unk_id] = -1e9
        logits[:, :, pad_id] = -1e9
        
        for b_idx, (cat, prompt, target) in enumerate(batch):
            if cat not in cat_stats:
                cat_stats[cat] = {"passed": 0, "total": 0}
            cat_stats[cat]["total"] += 1
            
            ans_pos = seq_lens[b_idx] - 1  # Index of '[a]'
            pred_tok_id = logits[b_idx, ans_pos].argmax().item()
            pred_word = VOCAB[pred_tok_id]
            
            # Clean expected target words (e.g. ["kitchen"] or ["ball", "and", "box"])
            expected_words = [w for w in re.findall(r"\w+", target.lower())]
            
            # Single-token answer check (98% of tasks)
            if len(expected_words) == 1:
                is_correct = (pred_word == expected_words[0])
            else:
                # Multi-token answer check (Lists & Sets: e.g. "ball and box .")
                gen_words = [pred_word]
                curr_toks = tokenized_batch[b_idx] + [pred_tok_id]
                
                # Roll forward for remaining list tokens
                while len(gen_words) < len(expected_words) + 2:
                    x_in = torch.tensor([curr_toks[-BLOCK_SIZE:]], dtype=torch.long, device=device)
                    out_logits = model(x_in)[0, -1]
                    out_logits[unk_id] = -1e9
                    out_logits[pad_id] = -1e9
                    
                    next_id = out_logits.argmax().item()
                    next_word = VOCAB[next_id]
                    if next_word in ("<eos>", "<pad>", "."):
                        break
                    gen_words.append(next_word)
                    curr_toks.append(next_id)
                    
                is_correct = (gen_words == expected_words)
                
            if is_correct:
                cat_stats[cat]["passed"] += 1
                total_passed += 1

        # Progress monitor
        processed = min(i + BATCH_SIZE, total_questions)
        pct = (processed / total_questions) * 100
        elapsed = time.perf_counter() - t_start
        qps = processed / (elapsed + 1e-5)
        print(f"\rEvaluating: {processed:5d}/{total_questions} ({pct:5.1f}%) | Speed: {qps:5.0f} tests/sec", end="", flush=True)

t_total = time.perf_counter() - t_start
print(f"\nCompleted in {t_total:.2f} seconds! ({total_questions / t_total:.1f} tests/sec)\n")

# 6. Scorecard Display
print("=" * 76)
print("                   PARALLEL EVALUATION SCORECARD                       ")
print("=" * 76)

for cat, s in cat_stats.items():
    if s["total"] > 0:
        pct = (s["passed"] / s["total"]) * 100
        bar = ("#" * int(pct // 10)).ljust(10, "-")
        print(f"{cat:<34} : {s['passed']:05d}/{s['total']:05d} [{bar}] {pct:5.1f}%")

final_pct = (total_passed / total_questions) * 100
final_bar = ("#" * int(final_pct // 10)).ljust(10, "-")
print("-" * 76)
print(f"TOTAL REASONING SCORE              : {total_passed:05d}/{total_questions:05d} [{final_bar}] {final_pct:5.1f}%")
print("=" * 76)