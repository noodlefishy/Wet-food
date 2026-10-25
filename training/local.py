# =====================================================================
# WET FOOD: BLAZING FAST LOCAL MAC TRAINING (CPU OPTIMIZED)
# =====================================================================
import torch
import torch.nn as nn
from torch.nn import functional as F
import requests
import re
import math
import random
from collections import Counter
import os

# For a 138k parameter model, Apple's M-Series CPU cache is actually 
# faster than sending tiny 64-D matrices back and forth to the GPU.
MODELS_DIR = "models/local"
os.makedirs(MODELS_DIR, exist_ok=True)

device = "cpu"

print("==================================================")
print(f" Training Verified 4L 64-D Engine on: {device.upper()}")
print("==================================================")

os.makedirs("models", exist_ok=True)

# ---------------------------------------------------------
# 1. FETCH FAIRY TALE FOUNDATION
# ---------------------------------------------------------
if os.path.exists("stories.txt"):
    print("Using cached stories.txt...")
    with open("stories.txt", "r", encoding="utf-8", errors="ignore") as f:
        raw_stories = f.read().lower()
else:
    print("Downloading TinyStories...")
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, stream=True)
    with open("stories.txt", "wb") as f:
        for chunk in response.iter_content(chunk_size=512 * 1024):
            f.write(chunk)
            if f.tell() >= 6 * 1024 * 1024: break
    with open("stories.txt", "r", encoding="utf-8", errors="ignore") as f:
        raw_stories = f.read().lower()

raw_stories = re.sub(r"\b(timmy|tom|jack|sam|billy)\b", "bob", raw_stories)
raw_stories = re.sub(r"\b(lily|anna|sarah|jane|lucy|emma)\b", "alice", raw_stories)

# ---------------------------------------------------------
# 2. COMPILE REASONING TASKS
# ---------------------------------------------------------
print("Compiling Strictly Balanced Reasoning Curriculum...")
actors = ["alice", "bob", "meow", "miaou"]
locations = ["garden", "kitchen", "park", "room"]
items = ["milk", "fish", "ball", "box"]
colors = ["red", "blue", "yellow", "green", "pink", "white"]

logic_tasks = []

# Location
for _ in range(25):
    for a in actors:
        for l in locations:
            logic_tasks.append(f"[c] {a} went to the {l} . [q] where is {a} ? [a] {l} . <eos>")
            logic_tasks.append(f"[c] {a} is in the {l} . [q] where is {a} ? [a] {l} . <eos>")
for _ in range(350):
    a1, a2 = random.sample(actors, 2)
    l1, l2 = random.sample(locations, 2)
    logic_tasks.append(f"[c] {a1} went to the {l1} . {a2} went to the {l2} . [q] where is {a1} ? [a] {l1} . <eos>")
    logic_tasks.append(f"[c] {a1} went to the {l1} . {a2} went to the {l2} . [q] where is {a2} ? [a] {l2} . <eos>")
    logic_tasks.append(f"[c] {a1} is in the {l1} . {a2} is in the {l2} . [q] where is {a1} ? [a] {l1} . <eos>")
    logic_tasks.append(f"[c] {a1} is in the {l1} . {a2} is in the {l2} . [q] where is {a2} ? [a] {l2} . <eos>")
for _ in range(300):
    l1, l2 = random.sample(locations, 2)
    logic_tasks.append(f"[c] alice went to the {l1} . bob went to the {l2} . [q] where is alice ? [a] {l1} . <eos>")
    logic_tasks.append(f"[c] alice went to the {l1} . bob went to the {l2} . [q] where is bob ? [a] {l2} . <eos>")
    logic_tasks.append(f"[c] bob went to the {l2} . alice went to the {l1} . [q] where is alice ? [a] {l1} . <eos>")
    logic_tasks.append(f"[c] bob went to the {l2} . alice went to the {l1} . [q] where is bob ? [a] {l2} . <eos>")
for _ in range(250):
    a1, a2, a3 = random.sample(actors, 3)
    l1, l2, l3 = random.sample(locations, 3)
    t_a, t_l = random.choice([(a1, l1), (a2, l2), (a3, l3)])
    logic_tasks.append(f"[c] {a1} went to the {l1} . {a2} went to the {l2} . {a3} went to the {l3} . [q] where is {t_a} ? [a] {t_l} . <eos>")
for _ in range(250):
    a = random.choice(actors)
    l1, l2 = random.sample(locations, 2)
    logic_tasks.append(f"[c] {a} went to the {l1} . then {a} went to the {l2} . [q] where is {a} ? [a] {l2} . <eos>")

# Object Location
for _ in range(20):
    for i in items:
        for l in locations:
            logic_tasks.append(f"[c] the {i} is in the {l} . [q] where is the {i} ? [a] {l} . <eos>")
for _ in range(350):
    i1, i2 = random.sample(items, 2)
    l1, l2 = random.sample(locations, 2)
    logic_tasks.append(f"[c] the {i1} is in the {l1} . the {i2} is in the {l2} . [q] where is the {i1} ? [a] {l1} . <eos>")
    logic_tasks.append(f"[c] the {i1} is in the {l1} . the {i2} is in the {l2} . [q] where is the {i2} ? [a] {l2} . <eos>")
    logic_tasks.append(f"[c] the {i1} is in the {l1} and the {i2} is in the {l2} . [q] where is the {i1} ? [a] {l1} . <eos>")
    logic_tasks.append(f"[c] the {i1} is in the {l1} and the {i2} is in the {l2} . [q] where is the {i2} ? [a] {l2} . <eos>")
for _ in range(200):
    i1, i2, i3 = random.sample(items, 3)
    l1, l2, l3 = random.sample(locations, 3)
    t_i, t_l = random.choice([(i1, l1), (i2, l2), (i3, l3)])
    logic_tasks.append(f"[c] the {i1} is in the {l1} . the {i2} is in the {l2} . the {i3} is in the {l3} . [q] where is the {t_i} ? [a] {t_l} . <eos>")

# Possession
for _ in range(20):
    for a in actors:
        for i in items:
            logic_tasks.append(f"[c] {a} has the {i} . [q] what does {a} have ? [a] {i} . <eos>")
            logic_tasks.append(f"[c] {a} found a {i} . [q] what did {a} find ? [a] {i} . <eos>")
for _ in range(300):
    a1, a2 = random.sample(actors, 2)
    i1, i2 = random.sample(items, 2)
    logic_tasks.append(f"[c] {a1} has the {i1} . {a2} has the {i2} . [q] what does {a1} have ? [a] {i1} . <eos>")
    logic_tasks.append(f"[c] {a1} has the {i1} . {a2} has the {i2} . [q] what does {a2} have ? [a] {i2} . <eos>")
    logic_tasks.append(f"[c] {a1} found a {i1} . {a2} found a {i2} . [q] what did {a1} find ? [a] {i1} . <eos>")
    logic_tasks.append(f"[c] {a1} found a {i1} . {a2} found a {i2} . [q] what did {a2} find ? [a] {i2} . <eos>")
    logic_tasks.append(f"[c] {a1} has the {i1} . {a2} has the {i2} . [q] who has the {i1} ? [a] {a1} . <eos>")
for _ in range(300):
    i1, i2 = random.sample(items, 2)
    logic_tasks.append(f"[c] alice has the {i1} . bob has the {i2} . [q] what does alice have ? [a] {i1} . <eos>")
    logic_tasks.append(f"[c] alice has the {i1} . bob has the {i2} . [q] what does bob have ? [a] {i2} . <eos>")
    logic_tasks.append(f"[c] bob has the {i2} . alice has the {i1} . [q] what does alice have ? [a] {i1} . <eos>")
    logic_tasks.append(f"[c] bob has the {i2} . alice has the {i1} . [q] what does bob have ? [a] {i2} . <eos>")
    logic_tasks.append(f"[c] alice found a {i1} . bob found a {i2} . [q] what did alice find ? [a] {i1} . <eos>")
    logic_tasks.append(f"[c] alice found a {i1} . bob found a {i2} . [q] what did bob find ? [a] {i2} . <eos>")

# Transfers
for _ in range(300):
    a1, a2 = random.sample(actors, 2)
    item = random.choice(items)
    logic_tasks.append(f"[c] {a1} had the {item} . {a1} gave the {item} to {a2} . [q] who has the {item} ? [a] {a2} . <eos>")
for _ in range(350):
    g1, g2 = random.sample(["alice", "bob"], 2)
    r1, r2 = random.sample(["meow", "miaou"], 2)
    i1, i2 = random.sample(items, 2)
    logic_tasks.append(f"[c] {g1} gave the {i1} to {r1} . {g2} gave the {i2} to {r2} . [q] who got the {i1} ? [a] {r1} . <eos>")
    logic_tasks.append(f"[c] {g1} gave the {i1} to {r1} . {g2} gave the {i2} to {r2} . [q] who got the {i2} ? [a] {r2} . <eos>")

# Colors
for a in actors:
    for c in colors:
        for i in items:
            logic_tasks.append(f"[c] {a} has a {c} {i} . [q] what color is the {i} ? [a] {c} . <eos>")
for _ in range(450):
    a1, a2 = random.sample(actors, 2)
    c1, c2 = random.sample(colors, 2)
    i1, i2 = random.sample(items, 2)
    logic_tasks.append(f"[c] {a1} has a {c1} {i1} . {a2} has a {c2} {i2} . [q] what color is the {i1} ? [a] {c1} . <eos>")
    logic_tasks.append(f"[c] {a1} has a {c1} {i1} . {a2} has a {c2} {i2} . [q] what color is the {i2} ? [a] {c2} . <eos>")
    logic_tasks.append(f"[c] {a1} has a {c1} {i1} and {a2} has a {c2} {i2} . [q] what color is the {i1} ? [a] {c1} . <eos>")
    logic_tasks.append(f"[c] {a1} has a {c1} {i1} and {a2} has a {c2} {i2} . [q] what color is the {i2} ? [a] {c2} . <eos>")


# --- 8. CREATIVE REASONING TASKS ---

# A. Transitive Reasoning (Alice with Bob -> Bob in Kitchen -> Alice in Kitchen)
for _ in range(300):
    a1, a2 = random.sample(actors, 2)
    l = random.choice(locations)
    logic_tasks.append(f"[c] {a1} is with {a2} . {a2} is in the {l} . [q] where is {a1} ? [a] {l} . <eos>")

# B. Functional Needs (Hungry -> Eat, Tired -> Sleep, Thirsty -> Water)
needs = [
    ("hungry", "wants to eat"),
    ("tired", "wants to sleep"),
    ("cold", "needs a blanket"),
    ("thirsty", "wants water")
]
for _ in range(300):
    a1, a2 = random.sample(actors, 2)
    (s1, n1), (s2, n2) = random.sample(needs, 2)
    logic_tasks.append(f"[c] {a1} is {s1} . {a2} is {s2} . [q] who {n1} ? [a] {a1} . <eos>")
    logic_tasks.append(f"[c] {a1} is {s1} . {a2} is {s2} . [q] who {n2} ? [a] {a2} . <eos>")

# C. Action Affordances (Capabilities of entities)
actions = [
    ("bird", "fly"),
    ("cat", "climb"),
    ("fish", "swim"),
    ("rabbit", "hop")
]
for _ in range(300):
    (e1, act1), (e2, act2) = random.sample(actions, 2)
    logic_tasks.append(f"[c] the {e1} can {act1} . the {e2} can {act2} . [q] who can {act1} ? [a] {e1} . <eos>")
    logic_tasks.append(f"[c] the {e1} can {act1} . the {e2} can {act2} . [q] who can {act2} ? [a] {e2} . <eos>")

# D. Physical States & Opposites
states = [
    ("milk", "hot"),
    ("water", "cold"),
    ("box", "heavy"),
    ("ball", "soft")
]
for _ in range(300):
    (i1, p1), (i2, p2) = random.sample(states, 2)
    logic_tasks.append(f"[c] the {i1} is {p1} . the {i2} is {p2} . [q] what is {p1} ? [a] {i1} . <eos>")
    logic_tasks.append(f"[c] the {i1} is {p1} . the {i2} is {p2} . [q] what is {p2} ? [a] {i2} . <eos>")

# E. Temporal Ordering (First vs. Next)
for _ in range(300):
    a = random.choice(actors)
    l1, l2 = random.sample(locations, 2)
    logic_tasks.append(f"[c] first {a} went to the {l1} . then {a} went to the {l2} . [q] where was {a} first ? [a] {l1} . <eos>")
    logic_tasks.append(f"[c] first {a} went to the {l1} . then {a} went to the {l2} . [q] where did {a} go next ? [a] {l2} . <eos>")

###

# Persona
cat_identities = [
    "question : what are your names ? answer : we are meow and miaou . <eos>",
    "question : who is meow ? answer : meow is the pico wh who talks to you on the web . <eos>",
    "question : who is miaou ? answer : miaou is the pico w who calculates the words . <eos>",
    "question : what are you ? answer : we are two tiny cat ais living in microcontrollers . <eos>",
    "question : are you real cats ? answer : we are digital kittens made of code and wifi . <eos>",
    "question : what do you love ? answer : we love milk warm sun naps and catching bugs . <eos>",
    "question : can you purr ? answer : purr purr purr yes we are purring right now . <eos>",
    "question : what hardware do you run on ? answer : we run on two raspberry pi pico chips over wifi . <eos>",
]
all_qa_tasks = logic_tasks + (cat_identities * 120)
random.seed(42)
random.shuffle(all_qa_tasks)

# ---------------------------------------------------------
# 3. VOCABULARY GENERATION
# ---------------------------------------------------------
VOCAB_SIZE = 1024
BLOCK_SIZE = 64
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 4

CAT_CORE_WORDS = [
    "<unk>", "<pad>", "<eos>", ".", "?", ":", "[c]", "[q]", "[a]",
    "question", "answer",
    "the", "in", "to", "a", "and", "who", "what", "where", "when", "is", "are", "was", "were", "then", "does", "did", "got",
    "alice", "bob", "meow", "miaou", "cat", "cats", "kitten", "kittens",
    "garden", "kitchen", "park", "room",
    "milk", "fish", "ball", "box", "rabbit",
    "purr", "paws", "fur", "tail", "whiskers", "nap", "sun", "warm", "fluffy",
    "pico", "wifi",
    "went", "gave", "has", "had", "have", "found", "find",
    "red", "blue", "yellow", "green", "pink", "white",
]

all_words = re.findall(r"\[[cqa]\]|<eos>|<unk>|<pad>|\w+|[.]|[?]|[:]", raw_stories + " " + " ".join(all_qa_tasks))
counts = Counter(all_words)
for w in CAT_CORE_WORDS:
    counts.pop(w, None)
remaining = VOCAB_SIZE - len(CAT_CORE_WORDS)
top_words = [w for w, _ in counts.most_common(remaining)]
VOCAB = CAT_CORE_WORDS + top_words
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}

# ---------------------------------------------------------
# 4. DATASET TOKENIZATION (STRICT CPU PROCESSING)
# ---------------------------------------------------------

print("Tokenizing and constructing masked QA tensors...")
def tokenize_str(s):
    return [WORD_TO_ID.get(w, 0) for w in re.findall(r"\[[cqa]\]|<eos>|<unk>|<pad>|\w+|[.]|[?]|[:]", s.lower())]

qa_tensor_x, qa_tensor_y = [], []
pad_id = WORD_TO_ID["<pad>"]
a_token_id = WORD_TO_ID["[a]"]
ans_token_id = WORD_TO_ID["answer"]

for task in all_qa_tasks:
    toks = tokenize_str(task)
    if len(toks) > BLOCK_SIZE: continue
    x = toks + [pad_id] * (BLOCK_SIZE - len(toks))
    y = [-100] * BLOCK_SIZE
    ans_start = -1
    for idx, t in enumerate(toks):
        if t in (a_token_id, ans_token_id):
            ans_start = idx
            break
    if ans_start != -1:
        for idx in range(ans_start, len(toks) - 1):
            y[idx] = toks[idx + 1]
    qa_tensor_x.append(torch.tensor(x, dtype=torch.long))
    qa_tensor_y.append(torch.tensor(y, dtype=torch.long))

# Data stays safely on CPU to prevent indexing latency
qa_x_data = torch.stack(qa_tensor_x)
qa_y_data = torch.stack(qa_tensor_y)

story_tokens = tokenize_str(raw_stories)
story_data = torch.tensor(story_tokens, dtype=torch.long)

# ---------------------------------------------------------
# 5. MODEL ARCHITECTURE & BALANCED TRAINING
# ---------------------------------------------------------
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.sa = nn.MultiheadAttention(EMBED_DIM, NUM_HEADS, batch_first=True)
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.mlp = nn.Sequential(nn.Linear(EMBED_DIM, 4 * EMBED_DIM), nn.ReLU(), nn.Linear(4 * EMBED_DIM, EMBED_DIM))

    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float("-inf"), diagonal=1)
        norm_x = self.ln1(x)
        attn_out, _ = self.sa(norm_x, norm_x, norm_x, attn_mask=mask, is_causal=True)
        return x + attn_out + self.mlp(self.ln2(x + attn_out))

class BabyStories(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.layers = nn.ModuleList([Block() for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=device))
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_f(x))

model = BabyStories().to(device)

TOTAL_STEPS = 8000
WARMUP = 200
BASE_LR, MIN_LR = 2e-3, 2e-4
HALF_BATCH = 16

optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-2)

def get_lr(step):
    if step < WARMUP: return BASE_LR * (step + 1) / WARMUP
    ratio = (step - WARMUP) / (TOTAL_STEPS - WARMUP)
    return MIN_LR + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (BASE_LR - MIN_LR)

def get_balanced_batch():
    # Random selection and slicing happens purely in CPU RAM (Instantaneous)
    qa_idx = torch.randint(len(qa_x_data), (HALF_BATCH,))
    x_qa = qa_x_data[qa_idx]
    y_qa = qa_y_data[qa_idx]

    st_idx = torch.randint(len(story_data) - BLOCK_SIZE - 1, (HALF_BATCH,))
    x_st = torch.stack([story_data[i : i + BLOCK_SIZE] for i in st_idx])
    y_st = torch.stack([story_data[i + 1 : i + BLOCK_SIZE + 1] for i in st_idx])
    
    # Only push the tiny final batch to device
    return x_qa.to(device), y_qa.to(device), x_st.to(device), y_st.to(device)

print(f"\nTraining for {TOTAL_STEPS} steps...")
model.train()
for step in range(TOTAL_STEPS + 1):
    lr = get_lr(step)
    for p in optimizer.param_groups: p["lr"] = lr

    x_qa, y_qa, x_st, y_st = get_balanced_batch()
    x_all = torch.cat([x_qa, x_st], dim=0)
    logits_all = model(x_all)
    
    logits_qa = logits_all[:HALF_BATCH]
    logits_st = logits_all[HALF_BATCH:]

    # SURGICAL LOSS GRADIENT WEIGHTING
    loss_qa = F.cross_entropy(logits_qa.view(-1, VOCAB_SIZE), y_qa.view(-1), ignore_index=-100)
    loss_st = F.cross_entropy(logits_st.view(-1, VOCAB_SIZE), y_st.view(-1))
    loss = 2.5 * loss_qa + 0.3 * loss_st

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 500 == 0:
        print(f"Step {step:5d}/{TOTAL_STEPS} | Total: {loss.item():.4f} | QA: {loss_qa.item():.4f} | Story: {loss_st.item():.4f}")

# ---------------------------------------------------------
# 6. EXPORT BINARIES DIRECTLY TO MODELS FOLDER
# ---------------------------------------------------------
def dump_t(t, f): 
    f.write(t.detach().cpu().to(torch.float32).numpy().tobytes())

print("\nSaving binary files to ./models/")
path1 = os.path.join(MODELS_DIR, "pico1 weights.model")
path2 = os.path.join(MODELS_DIR, "pico2 weights.model")

with open(path1, "wb") as f:
    dump_t(model.tok_emb.weight, f)
    dump_t(model.pos_emb.weight, f)
    for i in [0, 1]:
        for p in model.layers[i].parameters(): dump_t(p, f)

with open(path2, "wb") as f:
    for i in [2, 3]:
        for p in model.layers[i].parameters(): dump_t(p, f)
    for p in model.ln_f.parameters(): dump_t(p, f)
    dump_t(model.lm_head.weight, f)
    dump_t(model.lm_head.bias, f)

with open(os.path.join(MODELS_DIR, "vocab.py"), "w") as f: 
    f.write(f"VOCAB = {repr(VOCAB)}\n")

print(f"Meow Weights (pico1): {os.path.getsize(path1)/1024:.1f} KB -> Saved to {path1}")
print(f"Miaou Weights (pico2): {os.path.getsize(path2)/1024:.1f} KB -> Saved to {path2}")
print(f"Vocab List: Saved to ./{os.path.join(MODELS_DIR, "vocab.py")}")
print("\nTraining Complete!")