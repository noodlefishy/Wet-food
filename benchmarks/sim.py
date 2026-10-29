import os, sys, math, struct, json, re, random
try: import readline
except ImportError: pass

projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, projectRoot)
with open(os.path.join(projectRoot, "config", "model.json"), "r") as f: modCfg = json.load(f)
from models.local import vocab
from chunks import preprocess_chunks, decode_token, TOKEN_REGEX

VOCAB = vocab.VOCAB
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}
EMBED_DIM = modCfg["architecture"]["embed_dim"]
BLOCK_SIZE = modCfg["architecture"]["block_size"]
NUM_HEADS = modCfg["architecture"]["num_heads"]
NUM_LAYERS = modCfg["architecture"]["num_layers"]
MEOW_LAYERS = modCfg["architecture"].get("meow_layers", NUM_LAYERS // 2)
MIAOU_LAYERS = modCfg["architecture"].get("miaou_layers", NUM_LAYERS - MEOW_LAYERS)
HEAD_DIM = EMBED_DIM // NUM_HEADS
SINK_SIZE = 4; WINDOW_SIZE = 32
ROW_B = 4 + EMBED_DIM; L0_OFF = 1024 * ROW_B

p1Path = os.path.join(projectRoot, "models", "local", "pico1 weights.model")
p2Path = os.path.join(projectRoot, "models", "local", "pico2 weights.model")

if not os.path.exists(p1Path) or not os.path.exists(p2Path):
    print("[!] Model weights not found. Please run training first!")
    sys.exit(1)

f1, f2 = open(p1Path, "rb"), open(p2Path, "rb")

def readVec(f, c): return struct.unpack(f"{c}f", f.read(c * 4))

def readEmb(f, idx):
    f.seek(idx * ROW_B)
    s = struct.unpack('f', f.read(4))[0]
    return [v * s for v in struct.unpack(f'{EMBED_DIM}b', f.read(EMBED_DIM))]

def streamLinear(f, r, c, x):
    out = [0.0] * r
    for i in range(r):
        s = struct.unpack('f', f.read(4))[0]
        row = struct.unpack(f"{c}b", f.read(c))
        out[i] = sum(row[j] * x[j] for j in range(c)) * s
    b = struct.unpack(f"{r}f", f.read(r * 4))
    return [out[i] + b[i] for i in range(r)]

def layerNorm(x, w, b):
    m = sum(x) / len(x)
    v = sum((i - m)**2 for i in x) / len(x)
    s = math.sqrt(v + 1e-5)
    return [((x[i] - m) / s) * w[i] + b[i] for i in range(len(x))]

invFreq = [1.0 / (10000.0 ** (2.0 * i / HEAD_DIM)) for i in range(HEAD_DIM // 2)]
cosTab = [[math.cos(p * invFreq[i]) for i in range(HEAD_DIM // 2)] for p in range(BLOCK_SIZE)]
sinTab = [[math.sin(p * invFreq[i]) for i in range(HEAD_DIM // 2)] for p in range(BLOCK_SIZE)]

def applyRope(vec, pos):
    c, s = cosTab[min(pos, BLOCK_SIZE - 1)], sinTab[min(pos, BLOCK_SIZE - 1)]
    for i in range(HEAD_DIM // 2):
        i2 = i * 2
        x0, x1 = vec[i2], vec[i2 + 1]
        vec[i2]     = x0 * c[i] - x1 * s[i]
        vec[i2 + 1] = x0 * s[i] + x1 * c[i]

def runLayer(f, x, kv, pos):
    nx = layerNorm(x, readVec(f, EMBED_DIM), readVec(f, EMBED_DIM))
    qkv = streamLinear(f, EMBED_DIM + 2 * HEAD_DIM, EMBED_DIM, nx)
    q = qkv[:EMBED_DIM]
    k = qkv[EMBED_DIM : EMBED_DIM + HEAD_DIM]
    v = qkv[EMBED_DIM + HEAD_DIM :]
    
    cTab, sTab = cosTab[min(pos, BLOCK_SIZE - 1)], sinTab[min(pos, BLOCK_SIZE - 1)]
    for h in range(NUM_HEADS):
        ho = h * HEAD_DIM
        for i in range(HEAD_DIM // 2):
            i2 = ho + i * 2
            x0, x1 = q[i2], q[i2 + 1]
            q[i2]     = x0 * cTab[i] - x1 * sTab[i]
            q[i2 + 1] = x0 * sTab[i] + x1 * cTab[i]
    applyRope(k, pos)

    kv["k"].append(k); kv["v"].append(v)
    if len(kv["k"]) > SINK_SIZE + WINDOW_SIZE:
        kv["k"].pop(SINK_SIZE); kv["v"].pop(SINK_SIZE)

    ao, sc = [0.0] * EMBED_DIM, 1.0 / math.sqrt(HEAD_DIM)
    seqLen = len(kv["k"])
    for h in range(NUM_HEADS):
        ho = h * HEAD_DIM; qh = q[ho : ho + HEAD_DIM]
        scs = [sum(qh[i] * kv["k"][t][i] for i in range(HEAD_DIM)) * sc for t in range(seqLen)]
        ms = max(scs); wts = [math.exp(s - ms) for s in scs]; sWts = sum(wts); wts = [e / sWts for e in wts]
        for t in range(seqLen):
            for i in range(HEAD_DIM): ao[ho + i] += wts[t] * kv["v"][t][i]

    proj = streamLinear(f, EMBED_DIM, EMBED_DIM, ao)
    x = [x[i] + proj[i] for i in range(EMBED_DIM)]
    nx = layerNorm(x, readVec(f, EMBED_DIM), readVec(f, EMBED_DIM))
    h1 = streamLinear(f, 4 * EMBED_DIM, EMBED_DIM, nx)
    h2 = streamLinear(f, EMBED_DIM, 4 * EMBED_DIM, [max(0.0, val) for val in h1])
    return [x[i] + h2[i] for i in range(EMBED_DIM)]

def generate(promptTxt, maxTokens=25):
    isLog = any(t in promptTxt for t in ("[c]", "[q]", "[a]")) or any(promptTxt.lower().strip().startswith(w) for w in ("who", "what", "where", "is", "does", "why", "how", "question"))
    p = promptTxt.strip()
    if isLog:
        if "[c]" not in p and not p.startswith("question"): p = f"[c] {p}"
        if "[q]" in p and not p.endswith("[a]"): p = p.rstrip("?.: ") + " ? [a]"

    pClean = preprocess_chunks(p)
    toks = [WORD_TO_ID.get(w, 0) for w in re.findall(TOKEN_REGEX, pClean)]
    if not toks: return ""

    mKv = [{"k":[], "v":[]} for _ in range(MEOW_LAYERS)]
    miKv = [{"k":[], "v":[]} for _ in range(MIAOU_LAYERS)]
    curr, out = toks[-1], []
    
    for step in range(len(toks) + maxTokens):
        tid = toks[step] if step < len(toks) else curr
        x = readEmb(f1, tid)
        f1.seek(L0_OFF)
        for l in range(MEOW_LAYERS): x = runLayer(f1, x, mKv[l], step)
        f2.seek(0)
        for l in range(MIAOU_LAYERS): x = runLayer(f2, x, miKv[l], step)
        nx = layerNorm(x, readVec(f2, EMBED_DIM), readVec(f2, EMBED_DIM))
        lg = streamLinear(f2, 1024, EMBED_DIM, nx)
        lg[WORD_TO_ID["<unk>"]] = lg[WORD_TO_ID["<pad>"]] = -1e9
        
        if isLog:
            curr = max(range(1024), key=lambda i: lg[i])
        else:
            for pastTok in set(toks + out[-8:]): lg[pastTok] -= 1.8
            scaled = [v / 0.7 for v in lg]; ms = max(scaled); exps = [math.exp(v - ms) for v in scaled]
            probs = [e / sum(exps) for e in exps]; r, acc, curr = random.random(), 0.0, len(probs) - 1
            for i, pr in enumerate(probs):
                acc += pr
                if r <= acc: curr = i; break
                
        if step >= len(toks) - 1:
            w = VOCAB[curr]
            if w in ("<eos>", "<pad>"): break
            out.append(decode_token(w))
            if isLog and w == ".": break
            
    return " ".join(out)

if __name__ == "__main__":
    print("=" * 65)
    print("      WET FOOD LOCAL SIMULATOR (MICRO-COT + CHUNKS ACTIVE)     ")
    print("=" * 65)
    print("Type a prompt or 'exit' to quit.\n")
    while True:
        try:
            line = input("> ")
            if not line.strip(): continue
            if line.strip().lower() in ["exit", "quit"]: break
            print(f"  {generate(line)}\n")
        except (KeyboardInterrupt, EOFError): break