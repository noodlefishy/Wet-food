# =====================================================================
# ZERO-DEPENDENCY OFFLINE EVALUATOR (PURE STANDARD PYTHON)
# Stitches Meow (L0-1) and Miaou (L2-3) directly in memory.
# =====================================================================
import os
import sys
import math
import struct
import json
import re

print("[EVALUATOR] Running Pure Standard-Library Engine (No PyTorch, No NumPy)...")

# 1. Load Vocab
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.local import vocab
VOCAB = vocab.VOCAB



with open("config/model.json", "r") as f:
    config = json.load(f)

WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}
VOCAB_SIZE = 102
VOCAB_SIZE = config["architecture"]["vocab_size"]
BLOCK_SIZE = config["architecture"]["block_size"]
EMBED_DIM  = config["architecture"]["embed_dim"]
NUM_HEADS  = config["architecture"]["num_heads"]
NUM_LAYERS = config["architecture"]["num_layers"]

HEAD_DIM   = EMBED_DIM // NUM_HEADS
LAYER0_OFFSET = (VOCAB_SIZE * EMBED_DIM + BLOCK_SIZE * EMBED_DIM) * 4


def find_model_file(filename):
    # Searches models/local/ first
    search_paths = [
        os.path.join("models", "local", filename),
        os.path.join("models", filename),
        filename
    ]
    for p in search_paths:
        if os.path.exists(p): return p
    return filename

p1_path = find_model_file("pico1 weights.model")
p2_path = find_model_file("pico2 weights.model")

if not os.path.exists(p1_path) or not os.path.exists(p2_path):
    print(f"Error: Missing '{p1_path}' or '{p2_path}'. Check your models/ folder.")
    sys.exit(1)

f1 = open(p1_path, "rb")
f2 = open(p2_path, "rb")
print(f"[EVALUATOR] Opened {p1_path} and {p2_path} successfully.\n")

# ---------------------------------------------------------------------
# EXACT MATH ROUTINES FROM MEOW & MIAOU
# ---------------------------------------------------------------------
def read_vec(f, count):
    return struct.unpack(f"{count}f", f.read(count * 4))

def stream_linear(f, rows, cols, x):
    out = [0.0] * rows
    row_bytes = cols * 4
    for r in range(rows):
        row = struct.unpack(f"{cols}f", f.read(row_bytes))
        out[r] = sum(row[c] * x[c] for c in range(cols))
    bias = struct.unpack(f"{rows}f", f.read(rows * 4))
    for r in range(rows):
        out[r] += bias[r]
    return out

def layer_norm(x, w, b):
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    std = math.sqrt(var + 1e-5)
    return [((x[i] - mean) / std) * w[i] + b[i] for i in range(len(x))]

def run_layer(f, layer_idx, x, kv_caches):
    ln1_w = read_vec(f, EMBED_DIM)
    ln1_b = read_vec(f, EMBED_DIM)
    norm_x = layer_norm(x, ln1_w, ln1_b)
    
    qkv = stream_linear(f, 3 * EMBED_DIM, EMBED_DIM, norm_x)
    q = qkv[:EMBED_DIM]
    k = qkv[EMBED_DIM:2*EMBED_DIM]
    v = qkv[2*EMBED_DIM:]

    cache = kv_caches[layer_idx]
    cache["k"].append(k)
    cache["v"].append(v)
    if len(cache["k"]) > BLOCK_SIZE:
        cache["k"].pop(0)
        cache["v"].pop(0)

    attn_out = [0.0] * EMBED_DIM
    scale = 1.0 / math.sqrt(HEAD_DIM)
    seq_len = len(cache["k"])

    for h in range(NUM_HEADS):
        h_off = h * HEAD_DIM
        q_h = q[h_off:h_off + HEAD_DIM]
        scores = [sum(q_h[i] * cache["k"][t][h_off + i] for i in range(HEAD_DIM)) * scale for t in range(seq_len)]
        max_s = max(scores)
        exp_s = [math.exp(s - max_s) for s in scores]
        sum_exp = sum(exp_s)
        weights = [e / sum_exp for e in exp_s]

        for t in range(seq_len):
            v_h = cache["v"][t][h_off:h_off + HEAD_DIM]
            w_t = weights[t]
            for i in range(HEAD_DIM):
                attn_out[h_off + i] += w_t * v_h[i]

    proj = stream_linear(f, EMBED_DIM, EMBED_DIM, attn_out)
    x = [x[i] + proj[i] for i in range(EMBED_DIM)]

    ln2_w = read_vec(f, EMBED_DIM)
    ln2_b = read_vec(f, EMBED_DIM)
    norm_x = layer_norm(x, ln2_w, ln2_b)
    h1 = stream_linear(f, 4 * EMBED_DIM, EMBED_DIM, norm_x)
    h1_relu = [max(0.0, val) for val in h1]
    h2 = stream_linear(f, EMBED_DIM, 4 * EMBED_DIM, h1_relu)
    return [x[i] + h2[i] for i in range(EMBED_DIM)]

# ---------------------------------------------------------------------
# TOP-5 LOGIT EXTRACTOR & SAMPLER
# ---------------------------------------------------------------------
def extract_top_5(logits):
    indexed = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)[:5]
    max_v = indexed[0][1]
    exps = [math.exp(v - max_v) for _, v in indexed]
    sum_e = sum(exps)
    return [{"word": VOCAB[idx], "prob": round(e / sum_e, 3)} for (idx, _), e in zip(indexed, exps)]

# ---------------------------------------------------------------------
# END-TO-END INFERENCE (MEOW + MIAOU IN ONE LOOP)
# ---------------------------------------------------------------------
def run_model_end_to_end(prompt_text, max_tokens=25, greedy=True):
    tokens = [WORD_TO_ID.get(w, 0) for w in re.findall(r"\[[cqa]\]|<eos>|<unk>|<pad>|\w+|[.]|[?]|[:]", prompt_text.lower())]
    if not tokens:
        tokens = [WORD_TO_ID["meow"]]

    # Reset KV caches for both simulated nodes
    meow_kv = [{"k": [], "v": []}, {"k": [], "v": []}]
    miaou_kv = [{"k": [], "v": []}, {"k": [], "v": []}]

    current_tok = tokens[-1]
    out_words = []
    step_candidates_log = []

    for step in range(len(tokens) + max_tokens):
        pos = min(step, BLOCK_SIZE - 1)
        tok_id = tokens[step] if step < len(tokens) else current_tok

        # --- MEOW'S HALF: EMBEDDINGS + LAYERS 0 & 1 ---
        f1.seek(tok_id * EMBED_DIM * 4)
        tok_v = read_vec(f1, EMBED_DIM)

        f1.seek((VOCAB_SIZE * EMBED_DIM + pos * EMBED_DIM) * 4)
        pos_v = read_vec(f1, EMBED_DIM)

        x = [tok_v[i] + pos_v[i] for i in range(EMBED_DIM)]
        f1.seek(LAYER0_OFFSET)
        x = run_layer(f1, 0, x, meow_kv)
        x = run_layer(f1, 1, x, meow_kv)

        # --- SIMULATED WI-FI HOP: x IS SENT TO MIAOU ---

        # --- MIAOU'S HALF: LAYERS 2 & 3 + LM HEAD ---
        f2.seek(0)
        x = run_layer(f2, 0, x, miaou_kv)
        x = run_layer(f2, 1, x, miaou_kv)

        ln_f_w = read_vec(f2, EMBED_DIM)
        ln_f_b = read_vec(f2, EMBED_DIM)
        norm_x = layer_norm(x, ln_f_w, ln_f_b)

        logits = stream_linear(f2, VOCAB_SIZE, EMBED_DIM, norm_x)

        # Suppress special tokens
        logits[WORD_TO_ID["<unk>"]] = -1e9
        logits[WORD_TO_ID["<pad>"]] = -1e9

        top_candidates = extract_top_5(logits)

        # Greedy choice (Rank 1)
        best_token = max(range(len(logits)), key=lambda i: logits[i])
        current_tok = best_token

        if step >= len(tokens) - 1:
            w = VOCAB[current_tok]
            if w in ("<eos>", "<pad>"):
                break
            out_words.append(w)
            step_candidates_log.append(top_candidates)
            if greedy and w == ".":
                break

    return " ".join(out_words), step_candidates_log


with open('config/questions.json','r') as f:
    BENCHMARKS = json.load(f)

# ---------------------------------------------------------------------
# EXECUTE BENCHMARK RUN
# ---------------------------------------------------------------------
# Load the newly formatted JSON
with open(config["paths"]["questions_file"], "r") as f:
    BENCHMARKS_DICT = json.load(f)

total_tests = sum(len(items) for items in BENCHMARKS_DICT.values())

print("=" * 76)
print(f"      RUNNING LOCAL EVALUATOR ({total_tests} TESTS)      ")
print("=" * 76)

results_json = {
    "specs": {
        "engine": "Pure Python Standard Library (Zero PyTorch, Zero NumPy)",
        "layers": NUM_LAYERS,
        "embedding_dim": EMBED_DIM,
        "heads": NUM_HEADS,
        "vocab_size": VOCAB_SIZE,
        "block_size": BLOCK_SIZE
    },
    "summary": {},
    "details": []
}

cat_stats = {}
passed_count = 0
exact_count = 0
idx = 1

# Iterate through the dictionary!
for cat, items in BENCHMARKS_DICT.items():
    if cat not in cat_stats:
        cat_stats[cat] = {"passed": 0, "total": 0}

    for item in items:
        prompt = item["q"]
        target = item["target"]
        is_story = target is None

        out, candidates = run_model_end_to_end(prompt, max_tokens=25, greedy=not is_story)
        clean_out = out.strip().lower()
        clean_target = target.strip().lower() if target else ""

        record = {
            "id": idx,
            "category": cat,
            "prompt": prompt,
            "output": clean_out,
            "expected": clean_target if not is_story else None,
            "status": None,
            "top_candidates_step_1": candidates[0] if candidates else []
        }

        if not is_story:
            exact_count += 1
            cat_stats[cat]["total"] += 1
            matched = (clean_out == clean_target)
            if matched:
                passed_count += 1
                cat_stats[cat]["passed"] += 1
                record["status"] = "PASS"
                print(f"[{idx:03d}] PASS | {cat:<34} | {clean_out}")
            else:
                record["status"] = "FAIL"
                print(f"[{idx:03d}] FAIL | {cat:<34} | {clean_out} (Expected: {clean_target})")
        else:
            has_leak = any(tag in clean_out for tag in ("[q]", "[a]", "[c]"))
            status = "LEAK" if has_leak else "STORY"
            record["status"] = status
            print(f"[{idx:03d}] {status:<4} | {cat:<33} | {clean_out}")

        results_json["details"].append(record)
        idx += 1

print("\n" + "=" * 76)
print("                           CATEGORY SCORECARD                          ")
print("=" * 76)

for cat, stats in cat_stats.items():
    if stats["total"] > 0:
        pct = (stats["passed"] / stats["total"]) * 100
        bar = ("#" * int(pct // 10)).ljust(10, "-")
        print(f"{cat:<34} : {stats['passed']:02d}/{stats['total']:02d} [{bar}] {pct:5.1f}%")
        results_json["summary"][cat] = {
            "passed": stats["passed"],
            "total": stats["total"],
            "accuracy": round(pct, 2)
        }

total_pct = (passed_count / exact_count) * 100
total_bar = ("#" * int(total_pct // 10)).ljust(10, "-")
print("-" * 76)
print(f"TOTAL REASONING SCORE        : {passed_count:02d}/{exact_count:02d} [{total_bar}] {total_pct:5.1f}%")
print("=" * 76)

# Save results
out_file = "benchmark_results.json"
with open(out_file, "w") as f:
    json.dump(results_json, f, indent=2)

print(f"\n[EVALUATOR] Full JSON results saved to: {out_file}")

f1.close()
f2.close()