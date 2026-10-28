import os, struct, math, json

try:
    from firmware.vocab import VOCAB
except ImportError:
    from models.local.vocab import VOCAB

WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}
EMBED_DIM, BLOCK_SIZE, NUM_HEADS = 80, 64, 4
HEAD_DIM = EMBED_DIM // NUM_HEADS
ROW_B = 4 + EMBED_DIM
L0_OFF = (1024 * ROW_B) + (BLOCK_SIZE * ROW_B)

model_path = os.path.join(os.path.dirname(__file__), "models", "local", "pico1 weights.model")
if not os.path.exists(model_path):
    model_path = "models/local/pico1 weights.model"

f1 = open(model_path, "rb")

def rv(f, c): return struct.unpack(f"{c}f", f.read(c * 4))

def read_emb(f, idx, off=0):
    f.seek(off + idx * ROW_B)
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

def get_thought_vector(word):
    tid = WORD_TO_ID.get(word, 0)
    tv = read_emb(f1, tid, 0)
    pv = read_emb(f1, 0, 1024 * ROW_B)
    x = [tv[i] + pv[i] for i in range(EMBED_DIM)]
    
    f1.seek(L0_OFF)
    for l in range(3):
        nx = ln(x, rv(f1, EMBED_DIM), rv(f1, EMBED_DIM))
        qkv = sl(f1, EMBED_DIM + 2 * HEAD_DIM, EMBED_DIM, nx)
        v = qkv[EMBED_DIM + HEAD_DIM :]
        ao = v * NUM_HEADS
        proj = sl(f1, EMBED_DIM, EMBED_DIM, ao)
        x = [x[i] + proj[i] for i in range(EMBED_DIM)]
        
        nx = ln(x, rv(f1, EMBED_DIM), rv(f1, EMBED_DIM))
        h1 = sl(f1, 4 * EMBED_DIM, EMBED_DIM, nx)
        h2 = sl(f1, EMBED_DIM, 4 * EMBED_DIM, [max(0.0, val) for val in h1])
        x = [x[i] + h2[i] for i in range(EMBED_DIM)]
        
    return x

print("[*] Profiling Layer 2 thought vectors for all 1,024 tokens...")
vocab_vectors = [get_thought_vector(w) for w in VOCAB]

global_mean = [sum(vocab_vectors[w][d] for w in range(1024)) / 1024.0 for d in range(EMBED_DIM)]
global_std = [
    math.sqrt(sum((vocab_vectors[w][d] - global_mean[d])**2 for w in range(1024)) / 1024.0) + 1e-5
    for d in range(EMBED_DIM)
]

CIRCUITS = {
    "Mythic": (["dragon", "castle", "magic", "magical", "treasure", "monster", "giant", "king"], "Castles, dragons, magic, beasts"),
    "Cozy": (["honey", "cookies", "cake", "soft", "fluffy", "warm", "blanket", "tea", "nap"], "Comfort, warmth, food, rest"),
    "Action": (["jump", "climb", "flew", "fly", "run", "swam", "chased", "grabbed"], "Kinetic motion & pacing"),
    "Nature": (["forest", "woods", "river", "garden", "lake", "ocean", "beach", "cave"], "Outdoor world & settings"),
    "Needs": (["hungry", "tired", "thirsty", "food", "sleep", "water", "eat", "drink"], "Hunger, thirst, sleep drives"),
    "Scale": (["bigger", "smaller", "big", "small", "huge", "tiny", "tall", "large"], "Relative size & comparisons"),
    "Possession": (["has", "had", "have", "gave", "got", "found", "find", "put", "picked"], "Item ownership & transfer"),
    "Feline": (["cat", "cats", "kitten", "purr", "paws", "fur", "tail", "nap"], "Cat lore, fur, paws, purrs"),
    "Agents": (["alice", "bob", "felix", "luna", "he", "she", "they"], "Character identity & pronouns"),
    "Negation": (["not", "no", "never", "nobody", "nothing", "nowhere", "wasn"], "Logical refusal & negation")
}

dim_dictionary = {}
for d in range(EMBED_DIM):
    best_circuit = "Language Subsystem"
    best_desc = "Sentence syntax & grammar"
    max_z = 0.35
    
    for circuit_name, (tokens, desc) in CIRCUITS.items():
        c_ids = [WORD_TO_ID[t] for t in tokens if t in WORD_TO_ID]
        if not c_ids: continue
        c_mean = sum(vocab_vectors[tid][d] for tid in c_ids) / len(c_ids)
        z_score = (c_mean - global_mean[d]) / global_std[d]
        if z_score > max_z:
            max_z = z_score
            best_circuit = circuit_name
            best_desc = desc
            
    dim_dictionary[str(d)] = f"{best_circuit} | {best_desc}"

def get_theme_centroid(words):
    vec = [0.0] * EMBED_DIM
    count = 0
    for w in words:
        if w in WORD_TO_ID:
            v = vocab_vectors[WORD_TO_ID[w]]
            for i in range(EMBED_DIM): vec[i] += v[i]
            count += 1
    if count: vec = [x / count for x in vec]
    centered = [vec[i] - global_mean[i] for i in range(EMBED_DIM)]
    norm = math.sqrt(sum(x * x for x in centered)) or 1.0
    return [round(x / norm, 4) for x in centered]

theme_radar = {
    "myth": get_theme_centroid(["dragon", "castle", "magic", "monster", "giant", "king"]),
    "cozy": get_theme_centroid(["honey", "cookies", "cake", "soft", "fluffy", "warm", "blanket"]),
    "action": get_theme_centroid(["jump", "climb", "flew", "fly", "run", "swam", "chased"]),
    "nature": get_theme_centroid(["forest", "woods", "river", "garden", "lake", "ocean"])
}

CATALYSTS = {
    "Mythical & Royalty": ["dragon", "castle", "magic", "magical", "treasure", "monster", "giant", "king"],
    "Creatures & Wildlife": ["wolf", "owl", "fox", "bear", "lion", "elephant", "snake", "frog", "duck", "rabbit"],
    "Sensory & Comfort": ["honey", "cookies", "cake", "soft", "fluffy", "warm", "blanket", "tea", "milk"],
    "Atmosphere & Settings": ["garden", "kitchen", "park", "forest", "woods", "river", "lake", "ocean"],
    "Mischief & Action": ["mud", "messy", "silly", "broke", "bite", "ugly", "rude", "trouble", "chased"],
    "Action Verbs": ["jump", "climb", "flew", "fly", "run", "swam", "chased", "grabbed"]
}

cortex_payload = {
    "neurons": dim_dictionary,
    "radar": theme_radar,
    "mean": [round(m, 3) for m in global_mean],
    "catalysts": CATALYSTS,
    "vectors": {}
}

for cat, words in CATALYSTS.items():
    for w in words:
        if w in WORD_TO_ID:
            cortex_payload["vectors"][w] = [round(v, 3) for v in vocab_vectors[WORD_TO_ID[w]]]

out_path = os.path.join(os.path.dirname(__file__), "firmware", "cortex.json")
with open(out_path, "w") as f:
    json.dump(cortex_payload, f)

print(f"[✓] Generated clean Layer 2 cortex dictionary & theme radar -> {out_path}")