import struct
import math
import random
import re
import os
import sys

# ---------------------------------------------------------
# 1. FILE RESOLUTION & VOCABULARY SETUP
# ---------------------------------------------------------
def find_model_file(filename):
    """Search for model files in current dir, script dir, or ~/Downloads."""
    candidates = [
        filename,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
        os.path.expanduser(f"~/Downloads/{filename}"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return filename

try:
    from vocab import VOCAB
except ImportError:
    # Try importing from script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from vocab import VOCAB

WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}

EMBED_DIM = 48
BLOCK_SIZE = 64       # 64-token extended context
NUM_HEADS = 4
HEAD_DIM = EMBED_DIM // NUM_HEADS
VOCAB_SIZE = 1024
LAYER0_OFFSET = (VOCAB_SIZE * 48 + BLOCK_SIZE * 48) * 4

PATH_PICO1 = find_model_file("pico1 weights.model")
PATH_PICO2 = find_model_file("pico2 weights.model")

# ---------------------------------------------------------
# 2. TRANSFORMER TENSOR OPERATIONS
# ---------------------------------------------------------
def read_vec(f, count):
    return struct.unpack(f'{count}f', f.read(count * 4))

def linear_layer(f, rows, cols, x):
    out = [0.0] * rows
    for r in range(rows):
        row = struct.unpack(f'{cols}f', f.read(cols * 4))
        out[r] = sum(row[c] * x[c] for c in range(cols))
    
    # Read bias vector
    bias = struct.unpack(f'{rows}f', f.read(rows * 4))
    for r in range(rows):
        out[r] += bias[r]
    return out

def layer_norm(x, w, b):
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    std = math.sqrt(var + 1e-5)
    return [((x[i] - mean) / std) * w[i] + b[i] for i in range(len(x))]

def run_layer(f, x, cache):
    ln1_w = read_vec(f, EMBED_DIM)
    ln1_b = read_vec(f, EMBED_DIM)
    norm_x = layer_norm(x, ln1_w, ln1_b)
    
    qkv = linear_layer(f, 3 * EMBED_DIM, EMBED_DIM, norm_x)
    q, k, v = qkv[:EMBED_DIM], qkv[EMBED_DIM:2*EMBED_DIM], qkv[2*EMBED_DIM:]

    cache['k'].append(k)
    cache['v'].append(v)
    if len(cache['k']) > BLOCK_SIZE:
        cache['k'].pop(0)
        cache['v'].pop(0)

    attn_out = [0.0] * EMBED_DIM
    scale = 1.0 / math.sqrt(HEAD_DIM)
    seq_len = len(cache['k'])

    for h in range(NUM_HEADS):
        h_off = h * HEAD_DIM
        q_h = q[h_off:h_off + HEAD_DIM]
        scores = [sum(q_h[i] * cache['k'][t][h_off + i] for i in range(HEAD_DIM)) * scale for t in range(seq_len)]
        max_s = max(scores)
        exp_s = [math.exp(s - max_s) for s in scores]
        sum_exp = sum(exp_s)
        weights = [e / sum_exp for e in exp_s]

        for t in range(seq_len):
            v_h = cache['v'][t][h_off:h_off + HEAD_DIM]
            for i in range(HEAD_DIM):
                attn_out[h_off + i] += weights[t] * v_h[i]

    proj = linear_layer(f, EMBED_DIM, EMBED_DIM, attn_out)
    x = [x[i] + proj[i] for i in range(EMBED_DIM)]

    ln2_w = read_vec(f, EMBED_DIM)
    ln2_b = read_vec(f, EMBED_DIM)
    norm_x = layer_norm(x, ln2_w, ln2_b)
    
    h1 = linear_layer(f, 4 * EMBED_DIM, EMBED_DIM, norm_x)
    h1_relu = [max(0.0, val) for val in h1]
    h2 = linear_layer(f, EMBED_DIM, 4 * EMBED_DIM, h1_relu)
    return [x[i] + h2[i] for i in range(EMBED_DIM)]

def sample_top_k(logits, k=5, temp=0.7):
    indexed = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)[:k]
    if temp < 0.1:
        return indexed[0][0], [VOCAB[i] for i, _ in indexed]
    
    vals = [l / temp for _, l in indexed]
    max_v = max(vals)
    exps = [math.exp(v - max_v) for v in vals]
    sum_exp = sum(exps)
    probs = [e / sum_exp for e in exps]
    
    r = random.random()
    acc = 0.0
    chosen = indexed[0][0]
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            chosen = indexed[i][0]
            break
    return chosen, [VOCAB[i] for i, _ in indexed]

# ---------------------------------------------------------
# 3. GENERATION ENGINE
# ---------------------------------------------------------
def generate(prompt_text, max_words=64, temp=0.7):
    words = re.findall(r"\[[cqa]\]|<eos>|<unk>|<pad>|\w+|[.]|[?]|[:]", prompt_text.lower())
    tokens = [WORD_TO_ID.get(w, 0) for w in words]
    if not tokens:
        tokens = [WORD_TO_ID["meow"]]

    is_comprehension = "[a]" in prompt_text
    if is_comprehension:
        temp = 0.05  # Greedy precision for Q&A

    f1 = open(PATH_PICO1, "rb")
    f2 = open(PATH_PICO2, "rb")
    
    caches_pico1 = [{'k': [], 'v': []}, {'k': [], 'v': []}]
    caches_pico2 = [{'k': [], 'v': []}, {'k': [], 'v': []}]
    
    current_tok = tokens[-1]
    generated = []

    for step in range(len(tokens) + max_words):
        pos = min(step, BLOCK_SIZE - 1)
        tok_id = tokens[step] if step < len(tokens) else current_tok

        # --- MEOW (Pico 1: Layers 0 & 1) ---
        f1.seek(tok_id * 48 * 4)
        tok_v = read_vec(f1, EMBED_DIM)
        f1.seek((VOCAB_SIZE * 48 + pos * 48) * 4)
        pos_v = read_vec(f1, EMBED_DIM)
        x = [tok_v[i] + pos_v[i] for i in range(EMBED_DIM)]

        f1.seek(LAYER0_OFFSET)
        x = run_layer(f1, x, caches_pico1[0])
        x = run_layer(f1, x, caches_pico1[1])

        # --- MIAOU (Pico 2: Layers 2 & 3 + LM Head) ---
        f2.seek(0)
        x = run_layer(f2, x, caches_pico2[0])
        x = run_layer(f2, x, caches_pico2[1])

        ln_f_w = read_vec(f2, EMBED_DIM)
        ln_f_b = read_vec(f2, EMBED_DIM)
        norm_x = layer_norm(x, ln_f_w, ln_f_b)

        logits = linear_layer(f2, VOCAB_SIZE, EMBED_DIM, norm_x)

        # Banish <unk> and <pad> from generation
        logits[WORD_TO_ID["<unk>"]] = -1e9
        logits[WORD_TO_ID["<pad>"]] = -1e9

        current_tok, _ = sample_top_k(logits, k=5, temp=temp)

        # --- STOPPING CONDITIONS ---
        if step >= len(tokens) - 1:
            word = VOCAB[current_tok]
            
            # 1. Natural Termination: LM emitted <eos> or a control tag
            if word in ["<eos>", "[c]", "[q]"]:
                break
                
            generated.append(word)
            
            # 2. Q&A Mode: Stop immediately after the 1st period
            if is_comprehension and word == ".":
                break

            # 3. Story Mode: Conclude cleanly on a period
            if not is_comprehension and word == ".":
                sentence_count = generated.count(".")
                if sentence_count >= 2 or len(generated) >= 18:
                    break

            # 4. Context Buffer Safety
            if len(tokens) + len(generated) >= BLOCK_SIZE - 2:
                break

    f1.close()
    f2.close()

    # --- POST-PROCESSING: TRIM TRAILING FRAGMENTS ---
    # Ensure stories never end on trailing words like 'let s' or 'at the'
    if not is_comprehension and "." in generated:
        last_dot_idx = len(generated) - 1 - generated[::-1].index(".")
        generated = generated[:last_dot_idx + 1]

    return " ".join(generated)

# ---------------------------------------------------------
# 4. INTERACTIVE REPL
# ---------------------------------------------------------
if __name__ == "__main__":
    print("🐾 Meow & Miaou Distributed Simulator (Local)")
    print(f"Loaded: '{PATH_PICO1}' and '{PATH_PICO2}'")
    print(f"Context: {BLOCK_SIZE} tokens | Embedding: {EMBED_DIM}-D | Vocab: {VOCAB_SIZE} words")
    print("Type a prompt or 'exit' to quit.\n")
    
    while True:
        try:
            p = input("> ")
            if p.strip().lower() in ["exit", "quit"]:
                break
            if not p.strip():
                continue
            res = generate(p)
            print(f"  {res}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break