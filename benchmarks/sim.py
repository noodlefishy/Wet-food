import os, sys, math, struct, json, re, random
try:
    import readline
except ImportError:
    pass

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
with open(os.path.join(project_root, "config", "model.json"), "r") as f: mod_cfg = json.load(f)
from models.local import vocab
VOCAB = vocab.VOCAB
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}
EMBED_DIM = mod_cfg["architecture"]["embed_dim"]
BLOCK_SIZE = mod_cfg["architecture"]["block_size"]
NUM_HEADS = mod_cfg["architecture"]["num_heads"]
HEAD_DIM = EMBED_DIM // NUM_HEADS
SINK_SIZE = 4
WINDOW_SIZE = 32

ROW_B = 4 + EMBED_DIM 
L0_OFF = 1024 * ROW_B  # No absolute pos_emb!

p1_path = os.path.join(project_root, "models", "local", "pico1 weights.model")
p2_path = os.path.join(project_root, "models", "local", "pico2 weights.model")

if not os.path.exists(p1_path) or not os.path.exists(p2_path):
    print("[!] Model weights not found. Please run training first!")
    sys.exit(1)

f1, f2 = open(p1_path, "rb"), open(p2_path, "rb")

def rv(f, c): return struct.unpack(f"{c}f", f.read(c * 4))

def read_emb(f, idx):
    f.seek(idx * ROW_B)
    s = struct.unpack('f', f.read(4))[0]
    return [v * s for v in struct.unpack(f'{EMBED_DIM}b', f.read(EMBED_DIM))]

def sl(f, r, c, x):
    out = [0.0] * r
    for i in range(r):
        s = struct.unpack('f', f.read(4))[0]
        row = struct.unpack(f"{c}b", f.read(c))
        out[i] = sum(row[j] * x[j] for j in range(c)) * s
    b = struct.unpack(f"{r}f", f.read(r * 4))
    return [out[i] + b[i] for i in range(r)]

def ln(x, w, b):
    m = sum(x) / len(x)
    v = sum((i - m)**2 for i in x) / len(x)
    s = math.sqrt(v + 1e-5)
    return [((x[i] - m) / s) * w[i] + b[i] for i in range(len(x))]

inv_freq = [1.0 / (10000.0 ** (2.0 * i / HEAD_DIM)) for i in range(HEAD_DIM // 2)]
cos_tab = [[math.cos(p * inv_freq[i]) for i in range(HEAD_DIM // 2)] for p in range(BLOCK_SIZE)]
sin_tab = [[math.sin(p * inv_freq[i]) for i in range(HEAD_DIM // 2)] for p in range(BLOCK_SIZE)]

def apply_rope(vec, pos):
    c, s = cos_tab[min(pos, BLOCK_SIZE - 1)], sin_tab[min(pos, BLOCK_SIZE - 1)]
    for i in range(10):
        i2 = i * 2
        x0, x1 = vec[i2], vec[i2 + 1]
        vec[i2] = x0 * c[i] - x1 * s[i]
        vec[i2 + 1] = x0 * s[i] + x1 * c[i]

def rl(f, x, kv, pos):
    nx = ln(x, rv(f, EMBED_DIM), rv(f, EMBED_DIM))
    qkv = sl(f, EMBED_DIM + 2 * HEAD_DIM, EMBED_DIM, nx)
    q = qkv[:EMBED_DIM]
    k = qkv[EMBED_DIM : EMBED_DIM + HEAD_DIM]
    v = qkv[EMBED_DIM + HEAD_DIM :]
    
    for h in range(NUM_HEADS):
        ho = h * HEAD_DIM
        apply_rope(q[ho : ho + HEAD_DIM], pos)
    apply_rope(k, pos)
    
    kv["k"].append(k)
    kv["v"].append(v)
    
    # SWA window
    if len(kv["k"]) > SINK_SIZE + WINDOW_SIZE:
        kv["k"].pop(SINK_SIZE)
        kv["v"].pop(SINK_SIZE)
        
    ao, sc = [0.0] * EMBED_DIM, 1.0 / math.sqrt(HEAD_DIM)
    for h in range(NUM_HEADS):
        ho = h * HEAD_DIM; qh = q[ho : ho + HEAD_DIM]
        scs = [sum(qh[i] * kv["k"][t][i] for i in range(HEAD_DIM)) * sc for t in range(len(kv["k"]))]
        ms = max(scs); wts = [math.exp(s - ms) for s in scs]; s_wts = sum(wts); wts = [e / s_wts for e in wts]
        for t in range(len(wts)):
            for i in range(HEAD_DIM): ao[ho + i] += wts[t] * kv["v"][t][i]
            
    proj = sl(f, EMBED_DIM, EMBED_DIM, ao)
    x = [x[i] + proj[i] for i in range(EMBED_DIM)]
    nx = ln(x, rv(f, EMBED_DIM), rv(f, EMBED_DIM))
    h1 = sl(f, 4 * EMBED_DIM, EMBED_DIM, nx)
    h2 = sl(f, EMBED_DIM, 4 * EMBED_DIM, [max(0.0, val) for val in h1])
    return [x[i] + h2[i] for i in range(EMBED_DIM)]

def generate(p_txt, max_t=30):
    is_log = any(t in p_txt for t in ("[c]", "[q]", "[a]")) or any(p_txt.lower().strip().startswith(w) for w in ("who", "what", "where", "is", "does", "why", "how", "question"))
    p = p_txt.strip()
    if is_log:
        if "[c]" not in p and not p.startswith("question"): p = f"[c] {p}"
        if "[q]" in p and not p.endswith("[a]"): p = p.rstrip("?.: ") + " ? [a]"
        
    toks = [WORD_TO_ID.get(w, 0) for w in re.findall(r"\[[cqa]\]|<eos>|<unk>|<pad>|\w+|[.]|[?]|[:]", p.lower())] or [WORD_TO_ID["alice"]]
    m_kv = [{"k":[], "v":[]} for _ in range(3)]  # 3 Meow layers
    mi_kv = [{"k":[], "v":[]} for _ in range(5)] # 5 Miaou layers
    curr, out = toks[-1], []

    for step in range(len(toks) + max_t):
        tid = toks[step] if step < len(toks) else curr
        x = read_emb(f1, tid)
        
        f1.seek(L0_OFF)
        for l in range(3): x = rl(f1, x, m_kv[l], step)
        
        f2.seek(0)
        for l in range(5): x = rl(f2, x, mi_kv[l], step)
        
        nx = ln(x, rv(f2, EMBED_DIM), rv(f2, EMBED_DIM))
        lg = sl(f2, 1024, EMBED_DIM, nx)
        lg[WORD_TO_ID["<unk>"]] = lg[WORD_TO_ID["<pad>"]] = -1e9
        
        if is_log: 
            curr = max(range(1024), key=lambda i: lg[i])
        else:
            scaled = [v / 0.7 for v in lg]; ms = max(scaled); exps = [math.exp(v - ms) for v in scaled]
            probs = [e / sum(exps) for e in exps]; r, acc, curr = random.random(), 0.0, len(probs) - 1
            for i, pr in enumerate(probs):
                acc += pr
                if r <= acc: curr = i; break

        if step >= len(toks) - 1:
            w = VOCAB[curr]
            if w in ("<eos>", "<pad>"): break
            out.append(w)
            if is_log and w == ".": break
            
    return " ".join(out)

if __name__ == "__main__":
    print("="*65)
    print("       WET FOOD LOCAL SIMULATOR (8-LAYER RoPE ENGINE)         ")
    print("="*65)
    print("Architecture: 8 Layers (3 Meow + 5 Miaou), RoPE, INT8 W8A32")
    print("Type a prompt or 'exit' to quit.\n")

    while True:
        try:
            line = input("> ")
            if not line.strip(): continue
            if line.strip().lower() in ["exit", "quit"]: break
            res = generate(line)
            print(f"  {res}\n")
        except (KeyboardInterrupt, EOFError): 
            break
