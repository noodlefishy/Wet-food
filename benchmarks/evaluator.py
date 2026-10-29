import os, sys, math, struct, json, re

print("[EVALUATOR] Running Pure Standard-Library Engine (No PyTorch, No NumPy)...")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.local import vocab
VOCAB = vocab.VOCAB

with open("config/model.json", "r") as f: config = json.load(f)

WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}
VOCAB_SIZE = config["architecture"]["vocab_size"]
BLOCK_SIZE = config["architecture"]["block_size"]
EMBED_DIM  = config["architecture"]["embed_dim"]
NUM_HEADS  = config["architecture"]["num_heads"]
NUM_LAYERS = config["architecture"]["num_layers"]
HEAD_DIM   = EMBED_DIM // NUM_HEADS

def findModelFile(filename):
    for p in [os.path.join("models", "local", filename), os.path.join("models", filename), filename]:
        if os.path.exists(p): return p
    return filename

p1Path = findModelFile("pico1 weights.model")
p2Path = findModelFile("pico2 weights.model")

if not os.path.exists(p1Path) or not os.path.exists(p2Path):
    print(f"Error: Missing '{p1Path}' or '{p2Path}'. Check your models/ folder.")
    sys.exit(1)

f1, f2 = open(p1Path, "rb"), open(p2Path, "rb")

def readVec(f, count): return struct.unpack(f"{count}f", f.read(count * 4))

def streamLinear(f, rows, cols, x):
    out = [0.0] * rows
    rowBytes = cols * 4
    for r in range(rows):
        row = struct.unpack(f"{cols}f", f.read(rowBytes))
        out[r] = sum(row[c] * x[c] for c in range(cols))
    bias = struct.unpack(f"{rows}f", f.read(rows * 4))
    for r in range(rows): out[r] += bias[r]
    return out

def layerNorm(x, w, b):
    m = sum(x) / len(x)
    s = math.sqrt(sum((v - m) ** 2 for v in x) / len(x) + 1e-5)
    return [((x[i] - m) / s) * w[i] + b[i] for i in range(len(x))]

def runLayer(f, layerIdx, x, kvCaches):
    normX = layerNorm(x, readVec(f, EMBED_DIM), readVec(f, EMBED_DIM))
    qkv = streamLinear(f, 3 * EMBED_DIM, EMBED_DIM, normX)
    q, k, v = qkv[:EMBED_DIM], qkv[EMBED_DIM:2*EMBED_DIM], qkv[2*EMBED_DIM:]

    cache = kvCaches[layerIdx]
    cache["k"].append(k); cache["v"].append(v)
    if len(cache["k"]) > BLOCK_SIZE:
        cache["k"].pop(0); cache["v"].pop(0)

    attnOut = [0.0] * EMBED_DIM
    scale = 1.0 / math.sqrt(HEAD_DIM)
    seqLen = len(cache["k"])

    for h in range(NUM_HEADS):
        hOff = h * HEAD_DIM
        qH = q[hOff:hOff + HEAD_DIM]
        scores = [sum(qH[i] * cache["k"][t][hOff + i] for i in range(HEAD_DIM)) * scale for t in range(seqLen)]
        maxS = max(scores)
        expS = [math.exp(s - maxS) for s in scores]
        sumExp = sum(expS)
        weights = [e / sumExp for e in expS]
        for t in range(seqLen):
            vH = cache["v"][t][hOff:hOff + HEAD_DIM]
            for i in range(HEAD_DIM): attnOut[hOff + i] += weights[t] * vH[i]

    proj = streamLinear(f, EMBED_DIM, EMBED_DIM, attnOut)
    x = [x[i] + proj[i] for i in range(EMBED_DIM)]
    normX = layerNorm(x, readVec(f, EMBED_DIM), readVec(f, EMBED_DIM))
    h1 = streamLinear(f, 4 * EMBED_DIM, EMBED_DIM, normX)
    h2 = streamLinear(f, EMBED_DIM, 4 * EMBED_DIM, [max(0.0, val) for val in h1])
    return [x[i] + h2[i] for i in range(EMBED_DIM)]

def runModelEndToEnd(promptText, maxTokens=25):
    tokens = [WORD_TO_ID.get(w, 0) for w in re.findall(r"\[[cqa]\]|<eos>|<unk>|<pad>|\w+|[.]|[?]|[:]", promptText.lower())] or [WORD_TO_ID["meow"]]
    meowKv = [{"k": [], "v": []}, {"k": [], "v": []}]
    miaouKv = [{"k": [], "v": []}, {"k": [], "v": []}]
    currentTok = tokens[-1]
    outWords = []

    for step in range(len(tokens) + maxTokens):
        tokId = tokens[step] if step < len(tokens) else currentTok
        f1.seek(tokId * EMBED_DIM * 4)
        x = readVec(f1, EMBED_DIM)
        for l in range(2): x = runLayer(f1, l, x, meowKv)
        for l in range(2): x = runLayer(f2, l, x, miaouKv)
        normX = layerNorm(x, readVec(f2, EMBED_DIM), readVec(f2, EMBED_DIM))
        logits = streamLinear(f2, VOCAB_SIZE, EMBED_DIM, normX)
        currentTok = max(range(len(logits)), key=lambda idx: logits[idx])
        if step >= len(tokens) - 1:
            w = VOCAB[currentTok]
            if w in ("<eos>", "<pad>"): break
            outWords.append(w)
            if w == ".": break
    return " ".join(outWords)
