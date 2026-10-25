# =====================================================================
# WET FOOD: ZERO-DEPENDENCY LOCAL RUNNER (64-D, 4-LAYER)
# Simulates Meow & Miaou pipeline directly in terminal.
# Supports Left/Right cursor editing and Up/Down prompt history.
# =====================================================================
import os
import sys
import math
import struct
import random
import re
import atexit

# ---------------------------------------------------------
# 0. READLINE CONFIGURATION (ARROW KEYS & PROMPT HISTORY)
# ---------------------------------------------------------
try:
    import readline
    # Persistent history file in the benchmarks/ directory
    hist_dir = os.path.dirname(os.path.abspath(__file__))
    hist_file = os.path.join(hist_dir, ".local_history")
    
    try:
        readline.read_history_file(hist_file)
        readline.set_history_length(1000)
    except (FileNotFoundError, OSError):
        pass

    def save_history():
        try:
            readline.write_history_file(hist_file)
        except (OSError, IOError):
            pass

    atexit.register(save_history)
except ImportError:
    # Fallback for environments lacking readline
    pass

# ---------------------------------------------------------
# 1. PATH RESOLUTION & VOCABULARY IMPORT
# ---------------------------------------------------------
def resolve_path(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    candidates = [
        filename,
        os.path.join(project_root, filename),
        os.path.join(project_root, "models", filename),
        os.path.join(script_dir, filename),
        os.path.join(script_dir, "models", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return filename

# Ensure project root is in sys.path to import single root vocab.py
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import vocab
    VOCAB = vocab.VOCAB
except ImportError:
    print("Error: Could not import root vocab.py.")
    sys.exit(1)

WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}

# ---------------------------------------------------------
# 2. 64-D MODEL SPECIFICATIONS
# ---------------------------------------------------------
EMBED_DIM = 64
BLOCK_SIZE = 64
NUM_HEADS = 4
HEAD_DIM = EMBED_DIM // NUM_HEADS
VOCAB_SIZE = 1024
LAYER0_OFFSET = (VOCAB_SIZE * EMBED_DIM + BLOCK_SIZE * EMBED_DIM) * 4

PATH_PICO1 = resolve_path("pico1 weights.model")
PATH_PICO2 = resolve_path("pico2 weights.model")

if not os.path.exists(PATH_PICO1) or not os.path.exists(PATH_PICO2):
    print(f"Error: Missing model files:\n  {PATH_PICO1}\n  {PATH_PICO2}")
    sys.exit(1)

# ---------------------------------------------------------
# 3. PURE PYTHON HARDWARE MATH
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 4. END-TO-END PIPELINE (MEOW + MIAOU SIMULATOR)
# ---------------------------------------------------------
def generate(prompt_text, max_words=50):
    t_clean = prompt_text.strip().lower()
    
    # Auto-format natural queries if missing syntax
    q_starters = ("who", "what", "where", "why", "how", "when", "is", "are", "can", "do")
    if "[c]" in t_clean and "[q]" in t_clean and "[a]" not in t_clean:
        t_clean = t_clean.rstrip("?.: ") + " ? [a]"
    elif not t_clean.startswith("question") and not t_clean.startswith("[c]"):
        if t_clean.endswith("?") or any(t_clean.startswith(w) for w in q_starters):
            core = t_clean.rstrip("?.: ")
            t_clean = f"question : {core} ? answer :"

    # Punctuation spacing
    t_clean = t_clean.replace(":", " : ").replace("?", " ? ").replace(".", " . ")
    words = re.findall(r"\[[cqa]\]|<eos>|<unk>|<pad>|\w+|[.]|[?]|[:]", t_clean)
    tokens = [WORD_TO_ID.get(w, 0) for w in words]
    if not tokens:
        tokens = [WORD_TO_ID["meow"]]

    is_comprehension = "[a]" in t_clean or "answer" in t_clean
    temp = 0.05 if is_comprehension else 0.7

    f1 = open(PATH_PICO1, "rb")
    f2 = open(PATH_PICO2, "rb")

    meow_kv = [{"k": [], "v": []}, {"k": [], "v": []}]
    miaou_kv = [{"k": [], "v": []}, {"k": [], "v": []}]

    current_tok = tokens[-1]
    generated = []
    content_started = False
    suppress = {":", "?", "answer", "question", "<unk>", "<pad>"}

    for step in range(len(tokens) + max_words):
        pos = min(step, BLOCK_SIZE - 1)
        tok_id = tokens[step] if step < len(tokens) else current_tok

        # Meow: Embedding + Layers 0 & 1
        f1.seek(tok_id * EMBED_DIM * 4)
        tok_v = read_vec(f1, EMBED_DIM)

        f1.seek((VOCAB_SIZE * EMBED_DIM + pos * EMBED_DIM) * 4)
        pos_v = read_vec(f1, EMBED_DIM)

        x = [tok_v[i] + pos_v[i] for i in range(EMBED_DIM)]
        f1.seek(LAYER0_OFFSET)
        x = run_layer(f1, 0, x, meow_kv)
        x = run_layer(f1, 1, x, meow_kv)

        # Miaou: Layers 2 & 3 + LM Head
        f2.seek(0)
        x = run_layer(f2, 0, x, miaou_kv)
        x = run_layer(f2, 1, x, miaou_kv)

        ln_f_w = read_vec(f2, EMBED_DIM)
        ln_f_b = read_vec(f2, EMBED_DIM)
        norm_x = layer_norm(x, ln_f_w, ln_f_b)

        logits = stream_linear(f2, VOCAB_SIZE, EMBED_DIM, norm_x)
        logits[WORD_TO_ID["<unk>"]] = -1e9
        logits[WORD_TO_ID["<pad>"]] = -1e9

        # Sampling
        if temp < 0.1:
            current_tok = max(range(len(logits)), key=lambda i: logits[i])
        else:
            indexed = sorted(enumerate(logits), key=lambda it: it[1], reverse=True)[:5]
            max_v = indexed[0][1] / temp
            exps = [math.exp((val / temp) - max_v) for _, val in indexed]
            sum_e = sum(exps)
            probs = [e / sum_e for e in exps]
            r = random.random()
            acc = 0.0
            current_tok = indexed[0][0]
            for i, p in enumerate(probs):
                acc += p
                if r <= acc:
                    current_tok = indexed[i][0]
                    break

        if step >= len(tokens) - 1:
            word = VOCAB[current_tok]
            
            if not content_started:
                if word in suppress:
                    continue
                content_started = True

            if word in ["<eos>", "[c]", "[q]"]:
                break

            generated.append(word)

            if is_comprehension and word == ".":
                break

            if not is_comprehension and word == ".":
                if generated.count(".") >= 2 or len(generated) >= 20:
                    break

    f1.close()
    f2.close()

    res = " ".join(generated).replace(" .", ".")
    return res

# ---------------------------------------------------------
# 5. INTERACTIVE CLI REPL
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=" * 68)
    print("      MEOW & MIAOU DISTRIBUTED NEURAL ENGINE (LOCAL SIMULATOR)     ")
    print("=" * 68)
    print(f"Architecture : 4-Layer 64-D Transformer (Exact Hardware Math)")
    print(f"Vocabulary   : {VOCAB_SIZE} words | Context: {BLOCK_SIZE} tokens")
    print(f"Model Files  :\n  Pico 1 (Meow)  -> {PATH_PICO1}\n  Pico 2 (Miaou) -> {PATH_PICO2}")
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