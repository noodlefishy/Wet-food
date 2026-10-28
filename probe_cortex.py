import os, struct, math, json
from collections import defaultdict

from models.local import vocab
VOCAB = vocab.VOCAB
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}

EMBED_DIM, BLOCK_SIZE = 80, 64
ROW_B = 4 + EMBED_DIM
L0_OFF = (1024 * ROW_B) + (BLOCK_SIZE * ROW_B)

f1 = open("models/local/pico1 weights.model", "rb")

def rv(f, c): return struct.unpack(f"{c}f", f.read(c * 4))
def read_emb(f, idx, off=0):
    f.seek(off + idx * ROW_B)
    s = struct.unpack('f', f.read(4))[0]
    return [v * s for v in struct.unpack(f'{EMBED_DIM}b', f.read(EMBED_DIM))]

def sl(f, r, c, x):
    out = [0.0]*r
    for i in range(r):
        s = struct.unpack('f', f.read(4))[0]
        row = struct.unpack(f"{c}b", f.read(c))
        out[i] = sum(row[j] * x[j] for j in range(c)) * s
    b = struct.unpack(f"{r}f", f.read(r * 4))
    return [out[i] + b[i] for i in range(r)]

def ln(x, w, b):
    m = sum(x)/len(x); v = sum((i-m)**2 for i in x)/len(x); s = math.sqrt(v + 1e-5)
    return [((x[i]-m)/s)*w[i] + b[i] for i in range(len(x))]

def get_thought_vector(word):
    tid = WORD_TO_ID.get(word, 0)
    tv = read_emb(f1, tid, 0)
    pv = read_emb(f1, 0, 1024 * ROW_B)
    x = [tv[i] + pv[i] for i in range(EMBED_DIM)]
    f1.seek(L0_OFF)
    for l in range(3):
        nx = ln(x, rv(f1, EMBED_DIM), rv(f1, EMBED_DIM))
        qkv = sl(f1, EMBED_DIM + 40, EMBED_DIM, nx)
        proj = sl(f1, EMBED_DIM, EMBED_DIM, qkv[:EMBED_DIM])
        x = [x[i] + proj[i] for i in range(EMBED_DIM)]
        nx = ln(x, rv(f1, EMBED_DIM), rv(f1, EMBED_DIM))
        h1 = sl(f1, 4*EMBED_DIM, EMBED_DIM, nx)
        h2 = sl(f1, EMBED_DIM, 4*EMBED_DIM, [max(0.0, v) for v in h1])
        x = [x[i] + h2[i] for i in range(EMBED_DIM)]
    return x

print("[*] Profiling all 1,024 vocabulary tokens across Layer 2...")
vocab_vectors = [get_thought_vector(w) for w in VOCAB]

# 14 Fine-Grained Specialized Subsystems
CIRCUITS = {
    "Negation Flag": ["not", "no", "never", "nobody", "nothing", "nowhere", "wasn", "didn", "couldn", "wouldn"],
    "Needs & Drives": ["hungry", "tired", "thirsty", "food", "sleep", "water", "eat", "drink"],
    "Relative Size": ["bigger", "smaller", "big", "small", "huge", "tiny", "tall", "large", "than"],
    "Compass / Spatial": ["north", "south", "left", "right"],
    "Prepositions": ["in", "to", "from", "into", "inside", "at", "by", "near", "under", "behind"],
    "Ownership / Transfer": ["has", "had", "have", "gave", "got", "found", "find", "put", "left", "picked", "brought"],
    "Temporal Flow": ["first", "then", "now", "before", "after", "soon", "later", "eventually"],
    "Cardinality / Count": ["one", "two", "three", "none", "many", "few", "lot", "lots"],
    "Reasoning Delimiters": ["[c]", "[q]", "[a]", ".", "?", ":"],
    "Query Interrogative": ["who", "what", "where", "why", "how", "does", "did", "is", "was", "were"],
    "Truth Value": ["yes", "no", "ok", "okay"],
    "Agent Identity": ["alice", "bob", "felix", "luna", "he", "she", "they", "him", "her"],
    "Physical Item": ["ball", "box", "book", "cup", "toy", "stick", "key", "basket", "bag"],
    "Spatial Room": ["garden", "kitchen", "park", "room", "house", "forest", "field"],
    "Feline Lore": ["cat", "cats", "kitten", "purr", "paws", "fur", "tail", "whiskers", "nap"]
}

# 1. Monosemantic Neuron Dictionary with Top Trigger Words
dim_dictionary = {}
for d in range(EMBED_DIM):
    # Find top 3 activating words across entire vocabulary for dimension d
    top_word_indices = sorted(range(len(VOCAB)), key=lambda w: vocab_vectors[w][d], reverse=True)[:3]
    top_words = [VOCAB[w] for w in top_word_indices]
    
    # Identify which circuit has highest average alignment on this neuron
    best_circuit = "Latent Feature"
    max_score = 0.0
    for circuit_name, tokens in CIRCUITS.items():
        circuit_ids = [WORD_TO_ID[t] for t in tokens if t in WORD_TO_ID]
        if not circuit_ids: continue
        avg_act = sum(vocab_vectors[t_id][d] for t_id in circuit_ids) / len(circuit_ids)
        if abs(avg_act) > max_score:
            max_score = abs(avg_act)
            best_circuit = circuit_name
            
    # Combine circuit specialization with the exact top trigger tokens
    dim_dictionary[str(d)] = f"{best_circuit} [{', '.join(top_words[:2])}]"

# 2. Contrastive Steering Probes (Normalized Difference Vectors: Concept_A - Concept_B)
def get_contrastive_probe(set_a, set_b):
    vec_a = [0.0]*EMBED_DIM
    for w in set_a:
        if w in WORD_TO_ID:
            v = vocab_vectors[WORD_TO_ID[w]]
            for i in range(EMBED_DIM): vec_a[i] += v[i] / len(set_a)
            
    vec_b = [0.0]*EMBED_DIM
    for w in set_b:
        if w in WORD_TO_ID:
            v = vocab_vectors[WORD_TO_ID[w]]
            for i in range(EMBED_DIM): vec_b[i] += v[i] / len(set_b)
            
    diff = [vec_a[i] - vec_b[i] for i in range(EMBED_DIM)]
    norm = math.sqrt(sum(x*x for x in diff)) or 1.0
    return [round(x / norm, 3) for x in diff]

probe_entity = get_contrastive_probe(["alice"], ["bob"])
probe_category = get_contrastive_probe(CIRCUITS["Spatial Room"], CIRCUITS["Physical Item"])

print("\n" + "="*70)
print("             ADVANCED CORTEX DICTIONARY & PROBES              ")
print("="*70)
print(f"const NEURON_DICT = {json.dumps(dim_dictionary)};\n")
print(f"const PROBE_ENTITY = {probe_entity};\n")
print(f"const PROBE_CATEGORY = {probe_category};\n")