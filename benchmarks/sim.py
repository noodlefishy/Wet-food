import os, sys, math, struct, json, re
import readline  # <-- Fixes arrow keys and terminal history!

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
with open(os.path.join(project_root, "config", "model.json"), "r") as f: mod_cfg = json.load(f)
from models.local import vocab
VOCAB = vocab.VOCAB; WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}

EMBED_DIM = mod_cfg["architecture"]["embed_dim"]
BLOCK_SIZE = mod_cfg["architecture"]["block_size"]
HEAD_DIM = EMBED_DIM // 4

# Quantized row byte sizes
ROW_B = 4 + EMBED_DIM 
L0_OFF = (1024 * ROW_B) + (BLOCK_SIZE * ROW_B)

f1 = open(os.path.join(project_root, "models", "local", "pico1 weights.model"), "rb")
f2 = open(os.path.join(project_root, "models", "local", "pico2 weights.model"), "rb")

def rv(f, c): 
    return struct.unpack(f"{c}f", f.read(c * 4))

def read_emb(f, idx, off=0):
    f.seek(off + idx * ROW_B)
    scale = struct.unpack('f', f.read(4))[0]
    row = struct.unpack(f'{EMBED_DIM}b', f.read(EMBED_DIM))
    return [v * scale for v in row]

def sl(f, r, c, x):
    out = [0.0]*r
    for i in range(r):
        scale = struct.unpack('f', f.read(4))[0]
        row = struct.unpack(f"{c}b", f.read(c))
        out[i] = sum(row[j] * x[j] for j in range(c)) * scale
    b = struct.unpack(f"{r}f", f.read(r * 4))
    return [out[i] + b[i] for i in range(r)]

def ln(x, w, b):
    m = sum(x)/len(x); v = sum((i-m)**2 for i in x)/len(x); s = math.sqrt(v + 1e-5)
    return [((x[i]-m)/s)*w[i] + b[i] for i in range(len(x))]

def rl(f, x, kv):
    nx = ln(x, rv(f, EMBED_DIM), rv(f, EMBED_DIM))
    qkv = sl(f, EMBED_DIM + 2*HEAD_DIM, EMBED_DIM, nx)
    q, k, v = qkv[:EMBED_DIM], qkv[EMBED_DIM:EMBED_DIM+HEAD_DIM], qkv[EMBED_DIM+HEAD_DIM:]
    
    kv["k"].append(k); kv["v"].append(v)
    if len(kv["k"]) > BLOCK_SIZE: kv["k"].pop(0); kv["v"].pop(0)
    
    ao, sc = [0.0]*EMBED_DIM, 1.0/math.sqrt(HEAD_DIM)
    for h in range(4):
        ho = h * HEAD_DIM; qh = q[ho:ho+HEAD_DIM]
        scs = [sum(qh[i]*kv["k"][t][i] for i in range(HEAD_DIM))*sc for t in range(len(kv["k"]))]
        ms = max(scs); exps = [math.exp(s-ms) for s in scs]; wts = [e/sum(exps) for e in exps]
        for t in range(len(kv["k"])):
            for i in range(HEAD_DIM): ao[ho+i] += wts[t] * kv["v"][t][i]
            
    # FIXED: Compute matrix multiplication ONCE before unpacking it
    proj = sl(f, EMBED_DIM, EMBED_DIM, ao)
    x = [x[i] + proj[i] for i in range(EMBED_DIM)]
    
    nx = ln(x, rv(f, EMBED_DIM), rv(f, EMBED_DIM))
    
    # FIXED: Compute MLP ONCE before unpacking it
    h1 = sl(f, 4*EMBED_DIM, EMBED_DIM, nx)
    h2 = sl(f, EMBED_DIM, 4*EMBED_DIM, [max(0.0, v) for v in h1])
    
    return [x[i] + h2[i] for i in range(EMBED_DIM)]

def generate(p_txt, max_t=30):
    is_log = any(t in p_txt for t in ("[c]", "[q]", "[a]")) or any(p_txt.lower().strip().startswith(w) for w in ("who", "what", "where", "is", "does", "why", "how", "question"))
    p = p_txt.strip()
    if is_log:
        if "[c]" not in p and not p.startswith("question"): p = f"[c] {p}"
        if "[q]" in p and not p.endswith("[a]"): p = p.rstrip("?.: ") + " ? [a]"
        
    toks = [WORD_TO_ID.get(w, 0) for w in re.findall(r"\[[cqa]\]|<eos>|<unk>|<pad>|\w+|[.]|[?]|[:]", p.lower())] or [WORD_TO_ID["alice"]]
    m_kv, mi_kv = [{"k":[], "v":[]} for _ in range(3)], [{"k":[], "v":[]} for _ in range(3)]
    curr, out = toks[-1], []
    
    for step in range(len(toks) + max_t):
        tid = toks[step] if step < len(toks) else curr
        tv = read_emb(f1, tid, 0)
        pv = read_emb(f1, min(step, BLOCK_SIZE - 1), 1024 * ROW_B)
        x = [tv[i] + pv[i] for i in range(EMBED_DIM)]
        
        f1.seek(L0_OFF)
        for l in range(3): x = rl(f1, x, m_kv[l])
        
        f2.seek(0)
        for l in range(3): x = rl(f2, x, mi_kv[l])
        
        nx = ln(x, rv(f2, EMBED_DIM), rv(f2, EMBED_DIM))
        lg = sl(f2, 1024, EMBED_DIM, nx)
        lg[WORD_TO_ID["<unk>"]] = lg[WORD_TO_ID["<pad>"]] = -1e9
        
        import random
        if is_log: 
            curr = max(range(1024), key=lambda i: lg[i])
        else:
            scaled = [v/0.7 for v in lg]; ms = max(scaled); exps = [math.exp(v-ms) for v in scaled]
            probs = [e/sum(exps) for e in exps]; r, acc, curr = random.random(), 0.0, len(probs)-1
            for i, pr in enumerate(probs):
                acc += pr
                if r <= acc: curr = i; break
                
        if step >= len(toks) - 1:
            w = VOCAB[curr]
            if w in ("<eos>", "<pad>"): break
            out.append(w)
            if is_log and w == ".": break
            
    return " ".join(out)

while True:
    try:
        line = input("> ")
        if line.strip().lower() in ["exit", "quit"]: break
        if line.strip(): print(f"  {generate(line)}\n")
    except (KeyboardInterrupt, EOFError): 
        break