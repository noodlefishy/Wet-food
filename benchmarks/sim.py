# =====================================================================
# WET FOOD: CLEAN LOCAL INTERACTIVE SIMULATOR (DYNAMIC ARCHITECTURE)
# =====================================================================
import os
import sys
import math
import struct
import json
import re
import atexit

# Enable arrow keys & command history in terminal
try:
    import readline
    hist_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".sim_history")
    try:
        readline.read_history_file(hist_file)
        readline.set_history_length(1000)
    except (FileNotFoundError, OSError):
        pass
    atexit.register(lambda: readline.write_history_file(hist_file))
except ImportError:
    pass

# 1. Dynamically Load Config & Vocab (Matches Any Dimension Automatically!)
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
HEAD_DIM   = EMBED_DIM // NUM_HEADS
LAYER0_OFFSET = (VOCAB_SIZE * EMBED_DIM + BLOCK_SIZE * EMBED_DIM) * 4

# 2. Locate Weight Files
p1_path = os.path.join(project_root, "models", "local", "pico1 weights.model")
p2_path = os.path.join(project_root, "models", "local", "pico2 weights.model")

if not os.path.exists(p1_path) or not os.path.exists(p2_path):
    print(f"Error: Missing model files:\n  {p1_path}\n  {p2_path}")
    sys.exit(1)

f1 = open(p1_path, "rb")
f2 = open(p2_path, "rb")

# 3. Exact Hardware Math Functions (Pure Python)
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

def generate(prompt_text, max_tokens=25):
    # Ensure standard prompt tagging without mangling whitespace
    clean_p = prompt_text.strip()
    if not clean_p.startswith("[c]") and not clean_p.startswith("question"):
        if "[q]" in clean_p:
            clean_p = f"[c] {clean_p}"
            
    if "[q]" in clean_p and not clean_p.endswith("[a]"):
        clean_p = clean_p.rstrip("?.: ") + " ? [a]"

    words = re.findall(r"\[[cqa]\]|<eos>|<unk>|<pad>|\w+|[.]|[?]|[:]", clean_p.lower())
    tokens = [WORD_TO_ID.get(w, 0) for w in words]
    if not tokens:
        tokens = [WORD_TO_ID["felix"]]

    is_logic = "[a]" in clean_p or "answer" in clean_p

    # Fresh KV caches on EVERY forward pass
    meow_kv = [{"k": [], "v": []}, {"k": [], "v": []}]
    miaou_kv = [{"k": [], "v": []}, {"k": [], "v": []}]

    current_tok = tokens[-1]
    out_words = []

    for step in range(len(tokens) + max_tokens):
        pos = min(step, BLOCK_SIZE - 1)
        tok_id = tokens[step] if step < len(tokens) else current_tok

        # Meow (L0, L1)
        f1.seek(tok_id * EMBED_DIM * 4)
        tok_v = read_vec(f1, EMBED_DIM)

        f1.seek((VOCAB_SIZE * EMBED_DIM + pos * EMBED_DIM) * 4)
        pos_v = read_vec(f1, EMBED_DIM)

        x = [tok_v[i] + pos_v[i] for i in range(EMBED_DIM)]
        f1.seek(LAYER0_OFFSET)
        x = run_layer(f1, 0, x, meow_kv)
        x = run_layer(f1, 1, x, meow_kv)

        # Miaou (L2, L3 + LM Head)
        f2.seek(0)
        x = run_layer(f2, 0, x, miaou_kv)
        x = run_layer(f2, 1, x, miaou_kv)

        ln_f_w = read_vec(f2, EMBED_DIM)
        ln_f_b = read_vec(f2, EMBED_DIM)
        norm_x = layer_norm(x, ln_f_w, ln_f_b)

        logits = stream_linear(f2, VOCAB_SIZE, EMBED_DIM, norm_x)
        logits[WORD_TO_ID["<unk>"]] = -1e9
        logits[WORD_TO_ID["<pad>"]] = -1e9

        # Strict greedy argmax for logic queries
        best_token = max(range(len(logits)), key=lambda i: logits[i])
        current_tok = best_token

        if step >= len(tokens) - 1:
            w = VOCAB[current_tok]
            if w in ("<eos>", "<pad>"):
                break
            out_words.append(w)
            if is_logic and w == ".":
                break

    return " ".join(out_words)

# 4. Interactive Terminal REPL
if __name__ == "__main__":
    print("=" * 68)
    print("      MEOW & MIAOU DISTRIBUTED NEURAL ENGINE (LOCAL SIMULATOR)     ")
    print("=" * 68)
    print(f"Architecture : 4-Layer {EMBED_DIM}-D Transformer ({NUM_HEADS} Heads, Exact Math)")
    print(f"Vocabulary   : {VOCAB_SIZE} words | Context: {BLOCK_SIZE} tokens")
    print(f"Model Files  :\n  Pico 1 (Meow)  -> {p1_path}\n  Pico 2 (Miaou) -> {p2_path}")
    print("Navigation   : Left/Right arrows move cursor, Up/Down recalls history")
    print("=" * 68)
    print("Type a prompt or 'exit' to quit.\n")

    while True:
        try:
            prompt = input("> ")
            if prompt.strip().lower() in ["exit", "quit", "q"]:
                break
            if not prompt.strip():
                continue
            answer = generate(prompt)
            print(f"  {answer}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

    f1.close()
    f2.close()