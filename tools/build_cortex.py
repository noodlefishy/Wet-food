import os, sys, json, math, struct

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try: from firmware.vocab import VOCAB
except: from models.local.vocab import VOCAB
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}

EMBED_DIM, BLOCK_SIZE, ROW_B = 80, 64, 4 + 80
L0_OFF = (1024 * ROW_B) + (BLOCK_SIZE * ROW_B)
f1 = open(os.path.join(os.path.dirname(__file__), "..", "models", "local", "pico1 weights.model"), "rb")

def rv(f, c): return struct.unpack(f"{c}f", f.read(c * 4))
def read_emb(f, idx):
    f.seek(idx * ROW_B)
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
    if word not in WORD_TO_ID: return [0.0]*EMBED_DIM
    tv = read_emb(f1, WORD_TO_ID[word])
    pv = read_emb(f1, 1024) # Position 0
    x = [tv[i] + pv[i] for i in range(EMBED_DIM)]
    f1.seek(L0_OFF)
    for l in range(3):
        nx = ln(x, rv(f1, EMBED_DIM), rv(f1, EMBED_DIM))
        qkv = sl(f1, 100, EMBED_DIM, nx)
        
        # Expanded cleanly to avoid the 80x stream-read bug
        proj = sl(f1, EMBED_DIM, EMBED_DIM, qkv[:EMBED_DIM])
        x = [x[i] + proj[i] for i in range(EMBED_DIM)]
        
        nx = ln(x, rv(f1, EMBED_DIM), rv(f1, EMBED_DIM))
        h1 = sl(f1, 320, EMBED_DIM, nx)
        h2 = sl(f1, EMBED_DIM, 320, [max(0.0, v) for v in h1])
        x = [x[i] + h2[i] for i in range(EMBED_DIM)]
    return [round(val, 3) for val in x]

print("[*] Building external Cortex JSON...")

CATALYSTS = {
    "Mythical & Royalty": ["dragon", "castle", "magic", "magical", "treasure", "monster", "giant", "fairy", "creature", "tower", "king", "prince", "princess", "wings", "rainbow"],
    "Creatures & Wildlife": ["wolf", "owl", "fox", "bear", "lion", "elephant", "snake", "frog", "duck", "rabbit", "bunny", "squirrel", "butterfly", "swan", "sheep"],
    "Sensory, Food & Comfort": ["honey", "cookies", "cake", "candy", "cheese", "apple", "soup", "carrot", "jam", "berries", "bread", "milk", "blanket", "bubbles", "tea"],
    "Atmosphere & Settings": ["garden", "kitchen", "park", "forest", "woods", "river", "lake", "ocean", "beach", "cave", "jungle", "storm", "sunshine", "clouds"],
    "Mischief & Chaos": ["clown", "screamed", "smelly", "naughty", "mud", "messy", "silly", "broke", "bite", "ugly", "rude", "spicy", "spider", "danger", "trouble"],
    "Props, Gear & Toys": ["wagon", "kite", "balloon", "boat", "train", "truck", "bike", "umbrella", "drum", "doll", "basket", "bottle", "rope", "key", "box"],
    "Action Verbs": ["jump", "climb", "flew", "fly", "run", "swam", "dance", "hop", "sing", "shouted", "chased", "fell", "grabbed", "pushed", "explore"],
    "Sensory Adjectives": ["sparkly", "shiny", "bright", "dark", "huge", "giant", "tiny", "soft", "fluffy", "sharp", "spicy", "smelly", "warm", "wet", "delicious"]
}

NEURON_DICT = {"0": "Reasoning Delimiters [hopped, remember]", "1": "Compass / Spatial [skipped, want]", "2": "Cardinality / Count [off, catch]", "3": "Compass / Spatial [by, other]", "4": "Compass / Spatial [wanted, move]", "5": "Agent Identity [pot, peter]", "6": "Reasoning Delimiters [think, but]", "7": "Reasoning Delimiters [something, cars]", "8": "Compass / Spatial [bob, ride]", "9": "Compass / Spatial [for, don]", "10": "Compass / Spatial [t, same]", "11": "Feline Lore [nodded, t]", "12": "Negation Flag [snowman, smelly]", "13": "Compass / Spatial [drum, hopped]", "14": "Prepositions [worry, help]", "15": "Agent Identity [be, she]", "16": "Prepositions [amazed, polite]", "17": "Agent Identity [angry, himself]", "18": "Negation Flag [scared, hours]", "19": "Relative Size [made, are]", "20": "Reasoning Delimiters [plan, go]", "21": "Cardinality / Count [embarrassed, became]", "22": "Ownership / Transfer [laugh, sound]", "23": "Feline Lore [swing, thank]", "24": "Truth Value [rude, remember]", "25": "Reasoning Delimiters [reached, ?]", "26": "Negation Flag [can, couldn]", "27": "Needs & Drives [called, she]", "28": "Truth Value [why, there]", "29": "Prepositions [loved, able]", "30": "Prepositions [under, but]", "31": "Prepositions [near, longer]", "32": "Truth Value [find, he]", "33": "Needs & Drives [the, fake]", "34": "Negation Flag [you, arrived]", "35": "Agent Identity [they, amy]", "36": "Relative Size [liked, went]", "37": "Feline Lore [endoftext, bite]", "38": "Query Interrogative [pink, apples]", "39": "Negation Flag [singing, lots]", "40": "Temporal Flow [time, mary]", "41": "Prepositions [to, mia]", "42": "Prepositions [making, bought]", "43": "Truth Value [keep, everyone]", "44": "Temporal Flow [wished, mia]", "45": "Agent Identity [his, worry]", "46": "Agent Identity [but, stepped]", "47": "Negation Flag [purple, be]", "48": "Temporal Flow [upon, went]", "49": "Prepositions [different, clothes]", "50": "Negation Flag [smelly, ducks]", "51": "Agent Identity [he, wouldn]", "52": "Negation Flag [re, are]", "53": "Agent Identity [luna, bush]", "54": "Agent Identity [little, toys]", "55": "Cardinality / Count [an, children]", "56": "Agent Identity [feel, .]", "57": "Compass / Spatial [thanked, ?]", "58": "Negation Flag [back, been]", "59": "Ownership / Transfer [had, clothes]", "60": "Prepositions [when, perfect]", "61": "Query Interrogative [smart, moment]", "62": "Cardinality / Count [hi, not]", "63": "Reasoning Delimiters [had, hop]", "64": "Query Interrogative [far, broke]", "65": "Reasoning Delimiters [hi, mess]", "66": "Needs & Drives [it, dance]", "67": "Truth Value [animal, owl]", "68": "Reasoning Delimiters [delicious, from]", "69": "Ownership / Transfer [like, made]", "70": "Negation Flag [with, thank]", "71": "Compass / Spatial [couldn, won]", "72": "Needs & Drives [never, bigger]", "73": "Feline Lore [a, say]", "74": "Reasoning Delimiters [smelly, upon]", "75": "Needs & Drives [that, both]", "76": "Relative Size [helped, his]", "77": "Prepositions [they, plan]", "78": "Agent Identity [at, around]", "79": "Prepositions [alice, breath]"}
PROBE_ENTITY = [-0.015, -0.199, 0.152, -0.143, -0.054, 0.178, 0.054, -0.14, -0.206, 0.001, -0.113, -0.073, 0.07, -0.035, 0.051, -0.076, 0.038, 0.014, -0.004, -0.055, -0.01, -0.028, 0.016, -0.172, 0.034, 0.027, 0.045, -0.07, -0.069, 0.111, -0.238, -0.074, 0.117, -0.101, 0.214, 0.003, -0.035, -0.029, 0.065, 0.164, -0.059, 0.285, -0.013, 0.068, 0.084, 0.045, -0.142, 0.064, -0.047, 0.269, -0.125, -0.054, 0.22, 0.07, -0.144, -0.082, -0.081, -0.125, 0.141, -0.035, 0.016, -0.057, 0.169, -0.063, 0.013, -0.02, -0.041, 0.165, -0.132, -0.01, -0.01, 0.097, -0.129, 0.029, -0.165, 0.148, 0.09, 0.186, -0.004, 0.126]
PROBE_CATEGORY = [-0.132, 0.07, -0.242, 0.053, -0.052, 0.258, 0.159, -0.008, -0.111, -0.08, 0.043, -0.216, 0.095, 0.013, -0.056, -0.043, -0.151, 0.154, 0.031, 0.027, 0.004, -0.031, -0.076, -0.07, 0.17, -0.1, -0.057, -0.028, 0.099, -0.028, -0.039, 0.027, -0.034, 0.069, 0.071, 0.053, -0.027, 0.052, 0.019, 0.027, -0.152, 0.058, 0.031, -0.007, 0.027, 0.292, -0.272, 0.0, 0.15, -0.057, 0.049, -0.147, 0.02, 0.069, 0.073, 0.164, -0.069, -0.057, 0.098, 0.076, -0.07, -0.18, 0.011, -0.044, 0.063, -0.008, 0.046, 0.351, 0.018, -0.179, -0.057, -0.065, 0.226, -0.192, 0.036, 0.007, 0.009, 0.102, 0.02, 0.003]

cortex_payload = {
    "neurons": NEURON_DICT,
    "probes": {"entity": PROBE_ENTITY, "category": PROBE_CATEGORY},
    "catalysts": CATALYSTS,
    "vectors": {}
}

for cat, words in CATALYSTS.items():
    for w in words:
        if w not in cortex_payload["vectors"]:
            cortex_payload["vectors"][w] = get_thought_vector(w)

out_path = os.path.join(os.path.dirname(__file__), "..", "firmware", "cortex.json")
with open(out_path, "w") as f: json.dump(cortex_payload, f)
print(f"[✓] Successfully generated {out_path}!")