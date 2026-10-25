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
try:
    import vocab
    VOCAB = vocab.VOCAB
except ImportError:
    print("Error: vocab.py not found in current directory.")
    sys.exit(1)

WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}
VOCAB_SIZE = 1024
BLOCK_SIZE = 64
EMBED_DIM = 64
NUM_HEADS = 4
HEAD_DIM = EMBED_DIM // NUM_HEADS
LAYER0_OFFSET = (VOCAB_SIZE * EMBED_DIM + BLOCK_SIZE * EMBED_DIM) * 4

# 2. Locate Weight Files
p1_path = os.path.join("models", "pico1 weights.model") if os.path.exists("models/pico1 weights.model") else "pico1 weights.model"
p2_path = os.path.join("models", "pico2 weights.model") if os.path.exists("models/pico2 weights.model") else "pico2 weights.model"

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


# ---------------------------------------------------------------------
# EXPANDED 90-QUESTION BENCHMARK SUITE (INCLUDING CREATIVE REASONING)
# ---------------------------------------------------------------------
BENCHMARKS = [
    # --- CORE SPATIAL & BINDING SUITES ---
    # 1. Actor Location
    {"cat": "Actor Location", "q": "[c] alice went to the kitchen . bob went to the room . [q] where is alice ? [a]", "target": "kitchen ."},
    {"cat": "Actor Location", "q": "[c] alice went to the kitchen . bob went to the room . [q] where is bob ? [a]", "target": "room ."},
    {"cat": "Actor Location", "q": "[c] bob went to the room . alice went to the kitchen . [q] where is alice ? [a]", "target": "kitchen ."},
    {"cat": "Actor Location", "q": "[c] bob went to the room . alice went to the kitchen . [q] where is bob ? [a]", "target": "room ."},
    {"cat": "Actor Location", "q": "[c] meow went to the park . miaou went to the garden . [q] where is meow ? [a]", "target": "park ."},
    {"cat": "Actor Location", "q": "[c] meow went to the park . miaou went to the garden . [q] where is miaou ? [a]", "target": "garden ."},
    {"cat": "Actor Location", "q": "[c] alice went to the kitchen . bob went to the room . meow went to the park . [q] where is meow ? [a]", "target": "park ."},
    {"cat": "Actor Location", "q": "[c] alice went to the kitchen . then alice went to the garden . [q] where is alice ? [a]", "target": "garden ."},
    {"cat": "Actor Location", "q": "[c] bob went to the park . then bob went to the room . [q] where is bob ? [a]", "target": "room ."},
    {"cat": "Actor Location", "q": "[c] miaou went to the room . then miaou went to the park . [q] where is miaou ? [a]", "target": "park ."},

    # 2. Object Location
    {"cat": "Object Location", "q": "[c] the ball is in the garden . the box is in the kitchen . [q] where is the ball ? [a]", "target": "garden ."},
    {"cat": "Object Location", "q": "[c] the ball is in the garden . the box is in the kitchen . [q] where is the box ? [a]", "target": "kitchen ."},
    {"cat": "Object Location", "q": "[c] the milk is in the room . the fish is in the park . [q] where is the milk ? [a]", "target": "room ."},
    {"cat": "Object Location", "q": "[c] the milk is in the room . the fish is in the park . [q] where is the fish ? [a]", "target": "park ."},
    {"cat": "Object Location", "q": "[c] the box is in the park . the ball is in the room . [q] where is the box ? [a]", "target": "park ."},
    {"cat": "Object Location", "q": "[c] the box is in the park . the ball is in the room . [q] where is the ball ? [a]", "target": "room ."},
    {"cat": "Object Location", "q": "[c] the fish is in the kitchen . the milk is in the garden . [q] where is the fish ? [a]", "target": "kitchen ."},
    {"cat": "Object Location", "q": "[c] the milk is in the garden . the fish is in the kitchen . [q] where is the milk ? [a]", "target": "garden ."},

    # 3. Direct Possession (has)
    {"cat": "Possession (has)", "q": "[c] alice has the ball . bob has the fish . [q] what does alice have ? [a]", "target": "ball ."},
    {"cat": "Possession (has)", "q": "[c] alice has the ball . bob has the fish . [q] what does bob have ? [a]", "target": "fish ."},
    {"cat": "Possession (has)", "q": "[c] bob has the fish . alice has the ball . [q] what does alice have ? [a]", "target": "ball ."},
    {"cat": "Possession (has)", "q": "[c] bob has the fish . alice has the ball . [q] what does bob have ? [a]", "target": "fish ."},
    {"cat": "Possession (has)", "q": "[c] meow has the milk . miaou has the box . [q] what does meow have ? [a]", "target": "milk ."},
    {"cat": "Possession (has)", "q": "[c] meow has the milk . miaou has the box . [q] what does miaou have ? [a]", "target": "box ."},

    # 4. In-Context Discovery (found)
    {"cat": "Possession (found)", "q": "[c] alice found a ball . bob found a fish . [q] what did alice find ? [a]", "target": "ball ."},
    {"cat": "Possession (found)", "q": "[c] alice found a ball . bob found a fish . [q] what did bob find ? [a]", "target": "fish ."},
    {"cat": "Possession (found)", "q": "[c] bob found a fish . alice found a ball . [q] what did alice find ? [a]", "target": "ball ."},
    {"cat": "Possession (found)", "q": "[c] bob found a fish . alice found a ball . [q] what did bob find ? [a]", "target": "fish ."},
    {"cat": "Possession (found)", "q": "[c] meow found a fish . miaou found a ball . [q] what did meow find ? [a]", "target": "fish ."},
    {"cat": "Possession (found)", "q": "[c] meow found a fish . miaou found a ball . [q] what did miaou find ? [a]", "target": "ball ."},
    {"cat": "Possession (found)", "q": "[c] miaou found a box . meow found a milk . [q] what did miaou find ? [a]", "target": "box ."},

    # 5. Dynamic Object Transfers
    {"cat": "Object Transfer", "q": "[c] bob had the ball . bob gave the ball to alice . [q] who has the ball ? [a]", "target": "alice ."},
    {"cat": "Object Transfer", "q": "[c] alice had the fish . alice gave the fish to meow . [q] who has the fish ? [a]", "target": "meow ."},
    {"cat": "Object Transfer", "q": "[c] meow had the box . meow gave the box to miaou . [q] who has the box ? [a]", "target": "miaou ."},
    {"cat": "Object Transfer", "q": "[c] miaou had the milk . miaou gave the milk to bob . [q] who has the milk ? [a]", "target": "bob ."},
    {"cat": "Object Transfer", "q": "[c] alice had the ball . alice gave the ball to bob . [q] who has the ball ? [a]", "target": "bob ."},
    {"cat": "Object Transfer", "q": "[c] bob had the box . bob gave the box to meow . [q] who has the box ? [a]", "target": "meow ."},

    # 6. Indirect Object Identification (Giving / IOI)
    {"cat": "Giving (IOI)", "q": "[c] alice gave the ball to miaou . bob gave the box to meow . [q] who got the ball ? [a]", "target": "miaou ."},
    {"cat": "Giving (IOI)", "q": "[c] alice gave the ball to miaou . bob gave the box to meow . [q] who got the box ? [a]", "target": "meow ."},
    {"cat": "Giving (IOI)", "q": "[c] bob gave the box to meow . alice gave the ball to miaou . [q] who got the ball ? [a]", "target": "miaou ."},
    {"cat": "Giving (IOI)", "q": "[c] bob gave the box to meow . alice gave the ball to miaou . [q] who got the box ? [a]", "target": "meow ."},
    {"cat": "Giving (IOI)", "q": "[c] alice gave the milk to meow . bob gave the fish to miaou . [q] who got the milk ? [a]", "target": "meow ."},
    {"cat": "Giving (IOI)", "q": "[c] alice gave the milk to meow . bob gave the fish to miaou . [q] who got the fish ? [a]", "target": "miaou ."},
    {"cat": "Giving (IOI)", "q": "[c] meow gave the fish to alice . miaou gave the ball to bob . [q] who got the fish ? [a]", "target": "alice ."},
    {"cat": "Giving (IOI)", "q": "[c] meow gave the fish to alice . miaou gave the ball to bob . [q] who got the ball ? [a]", "target": "bob ."},

    # 7. Adjective-Noun Color Binding
    {"cat": "Color Binding", "q": "[c] alice has a red box . bob has a blue ball . [q] what color is the box ? [a]", "target": "red ."},
    {"cat": "Color Binding", "q": "[c] alice has a red box . bob has a blue ball . [q] what color is the ball ? [a]", "target": "blue ."},
    {"cat": "Color Binding", "q": "[c] meow has a yellow ball . miaou has a pink box . [q] what color is the ball ? [a]", "target": "yellow ."},
    {"cat": "Color Binding", "q": "[c] meow has a yellow ball . miaou has a pink box . [q] what color is the box ? [a]", "target": "pink ."},
    {"cat": "Color Binding", "q": "[c] bob has a green box . alice has a white ball . [q] what color is the box ? [a]", "target": "green ."},
    {"cat": "Color Binding", "q": "[c] bob has a green box . alice has a white ball . [q] what color is the ball ? [a]", "target": "white ."},
    {"cat": "Color Binding", "q": "[c] alice has a pink fish . bob has a green milk . [q] what color is the fish ? [a]", "target": "pink ."},
    {"cat": "Color Binding", "q": "[c] alice has a pink fish . bob has a green milk . [q] what color is the milk ? [a]", "target": "green ."},

    # 8. Feline System Persona
    {"cat": "Persona", "q": "question : what are your names ? answer :", "target": "we are meow and miaou ."},
    {"cat": "Persona", "q": "question : who is meow ? answer :", "target": "meow is the pico wh who talks to you on the web ."},
    {"cat": "Persona", "q": "question : who is miaou ? answer :", "target": "miaou is the pico w who calculates the words ."},
    {"cat": "Persona", "q": "question : what are you ? answer :", "target": "we are two tiny cat ais living in microcontrollers ."},
    {"cat": "Persona", "q": "question : are you real cats ? answer :", "target": "we are digital kittens made of code and wifi ."},
    {"cat": "Persona", "q": "question : what do you love ? answer :", "target": "we love milk warm sun naps and catching bugs ."},
    {"cat": "Persona", "q": "question : can you purr ? answer :", "target": "purr purr purr yes we are purring right now ."},
    {"cat": "Persona", "q": "question : what hardware do you run on ? answer :", "target": "we run on two raspberry pi pico chips over wifi ."},

    # 9. Inverted / Negation
    {"cat": "Inverted / Negation", "q": "[c] alice has the ball . bob has the fish . [q] who has the ball ? [a]", "target": "alice ."},
    {"cat": "Inverted / Negation", "q": "[c] alice has the ball . bob has the fish . [q] who has the fish ? [a]", "target": "bob ."},
    {"cat": "Inverted / Negation", "q": "[c] the milk is in the room . the box is in the park . [q] what is in the room ? [a]", "target": "milk ."},
    {"cat": "Inverted / Negation", "q": "[c] the milk is in the room . the box is in the park . [q] what is in the park ? [a]", "target": "box ."},

    # --- NEW CREATIVE REASONING SUITES ---
    # 10. Transitive Spatial Reasoning (Two-Hop Inference)
    {"cat": "Transitive", "q": "[c] alice is with bob . bob is in the kitchen . [q] where is alice ? [a]", "target": "kitchen ."},
    {"cat": "Transitive", "q": "[c] meow is with miaou . miaou is in the garden . [q] where is meow ? [a]", "target": "garden ."},
    {"cat": "Transitive", "q": "[c] bob is with alice . alice is in the park . [q] where is bob ? [a]", "target": "park ."},
    {"cat": "Transitive", "q": "[c] miaou is with meow . meow is in the room . [q] where is miaou ? [a]", "target": "room ."},

    # 11. Functional Needs & Theory of Mind
    {"cat": "Functional Needs", "q": "[c] alice is hungry . bob is tired . [q] who wants to eat ? [a]", "target": "alice ."},
    {"cat": "Functional Needs", "q": "[c] alice is hungry . bob is tired . [q] who wants to sleep ? [a]", "target": "bob ."},
    {"cat": "Functional Needs", "q": "[c] meow is cold . miaou is happy . [q] who needs a blanket ? [a]", "target": "meow ."},
    {"cat": "Functional Needs", "q": "[c] bob is thirsty . alice is warm . [q] who wants water ? [a]", "target": "bob ."},

    # 12. Action Affordances & Capabilities
    {"cat": "Action Affordance", "q": "[c] the bird can fly . the cat can climb . [q] who can fly ? [a]", "target": "bird ."},
    {"cat": "Action Affordance", "q": "[c] the bird can fly . the cat can climb . [q] who can climb ? [a]", "target": "cat ."},
    {"cat": "Action Affordance", "q": "[c] the fish can swim . the rabbit can hop . [q] who can swim ? [a]", "target": "fish ."},
    {"cat": "Action Affordance", "q": "[c] the fish can swim . the rabbit can hop . [q] who can hop ? [a]", "target": "rabbit ."},

    # 13. Physical State & Opposites
    {"cat": "Physical States", "q": "[c] the milk is hot . the water is cold . [q] what is hot ? [a]", "target": "milk ."},
    {"cat": "Physical States", "q": "[c] the milk is hot . the water is cold . [q] what is cold ? [a]", "target": "water ."},
    {"cat": "Physical States", "q": "[c] the box is heavy . the ball is soft . [q] what is heavy ? [a]", "target": "box ."},
    {"cat": "Physical States", "q": "[c] the box is heavy . the ball is soft . [q] what is soft ? [a]", "target": "ball ."},

    # 14. Temporal Ordering (First vs. Next)
    {"cat": "Temporal Order", "q": "[c] first bob went to the park . then bob went to the room . [q] where was bob first ? [a]", "target": "park ."},
    {"cat": "Temporal Order", "q": "[c] first bob went to the park . then bob went to the room . [q] where did bob go next ? [a]", "target": "room ."},
    {"cat": "Temporal Order", "q": "[c] first alice found a fish . then alice found a ball . [q] what did alice find first ? [a]", "target": "fish ."},
    {"cat": "Temporal Order", "q": "[c] first alice found a fish . then alice found a ball . [q] what did alice find next ? [a]", "target": "ball ."},

    # 15. Storytelling / Fluency Checks
    {"cat": "Storytelling", "q": "one day alice and miaou were in the kitchen .", "target": None},
    {"cat": "Storytelling", "q": "once upon a time bob went into the green garden .", "target": None},
    {"cat": "Storytelling", "q": "the sun was very warm and the little kitten wanted to", "target": None},
    {"cat": "Storytelling", "q": "alice and meow found a little box under the tree .", "target": None},
]

# ---------------------------------------------------------------------
# EXECUTE BENCHMARK RUN
# ---------------------------------------------------------------------
print("=" * 76)
print(f"      RUNNING LOCAL EVALUATOR ({len(BENCHMARKS)} TESTS)      ")
print("=" * 76)

results_json = {
    "specs": {
        "engine": "Pure Python Standard Library (Zero PyTorch, Zero NumPy)",
        "layers": 4,
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

for idx, item in enumerate(BENCHMARKS, 1):
    cat = item["cat"]
    prompt = item["q"]
    target = item["target"]
    is_story = target is None

    if cat not in cat_stats:
        cat_stats[cat] = {"passed": 0, "total": 0}

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
            print(f"[{idx:02d}] PASS | {cat:<20} | {clean_out}")
        else:
            record["status"] = "FAIL"
            print(f"[{idx:02d}] FAIL | {cat:<20} | {clean_out} (Expected: {clean_target})")
    else:
        has_leak = any(tag in clean_out for tag in ("[q]", "[a]", "[c]"))
        status = "LEAK" if has_leak else "STORY"
        record["status"] = status
        print(f"[{idx:02d}] {status:<4} | {cat:<20} | {clean_out}")

    results_json["details"].append(record)

print("\n" + "=" * 76)
print("                           CATEGORY SCORECARD                          ")
print("=" * 76)

for cat, stats in cat_stats.items():
    if stats["total"] > 0:
        pct = (stats["passed"] / stats["total"]) * 100
        bar = ("#" * int(pct // 10)).ljust(10, "-")
        print(f"{cat:<28} : {stats['passed']:02d}/{stats['total']:02d} [{bar}] {pct:5.1f}%")
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

results_json["summary"]["overall"] = {
    "passed": passed_count,
    "total": exact_count,
    "accuracy": round(total_pct, 2)
}

# Save results
out_file = "benchmark_results.json"
with open(out_file, "w") as f:
    json.dump(results_json, f, indent=2)

print(f"\n[EVALUATOR] Full JSON results saved to: {out_file}")

f1.close()
f2.close()