import torch
import torch.nn as nn
from torch.nn import functional as F
import requests
import re
import math
import random
from collections import Counter
import os
from google.colab import files

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training Verified 4L 64-D Engine on: {device}")

# ---------------------------------------------------------
# 1. FETCH FAIRY TALE FOUNDATION
# ---------------------------------------------------------
print("Fetching fairy-tale foundation...")
url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers, stream=True)

with open("stories.txt", "wb") as f:
    for chunk in response.iter_content(chunk_size=512 * 1024):
        f.write(chunk)
        if f.tell() >= 4 * 1024 * 1024:
            break

with open("stories.txt", "r", encoding="utf-8", errors="ignore") as f:
    raw_stories = f.read().lower()

raw_stories = re.sub(r"\b(timmy|tom|jack|sam|billy)\b", "bob", raw_stories)
raw_stories = re.sub(r"\b(lily|anna|sarah|jane|lucy|emma)\b", "alice", raw_stories)

# ---------------------------------------------------------
# 2. COMPILE REASONING TASKS (Symmetric Curriculum & Anti-Bias)
# ---------------------------------------------------------
print("Compiling Strictly Balanced Reasoning Curriculum...")

actors = ["alice", "bob", "meow", "miaou"]
locations = ["garden", "kitchen", "park", "room"]
items = ["milk", "fish", "ball", "box"]
colors = ["red", "blue", "yellow", "green", "pink", "white"]

logic_tasks = []

# --- 1. ACTOR LOCATION ---
# 1-Actor Primers: Cycles every actor through every location equally
for _ in range(25):
    for a in actors:
        for l in locations:
            logic_tasks.append(f"[c] {a} went to the {l} . [q] where is {a} ? [a] {l} . <eos>")
            logic_tasks.append(f"[c] {a} is in the {l} . [q] where is {a} ? [a] {l} . <eos>")

# 2-Actor Standard & Inverted Queries
for _ in range(350):
    a1, a2 = random.sample(actors, 2)
    l1, l2 = random.sample(locations, 2)
    logic_tasks.append(f"[c] {a1} went to the {l1} . {a2} went to the {l2} . [q] where is {a1} ? [a] {l1} . <eos>")
    logic_tasks.append(f"[c] {a1} went to the {l1} . {a2} went to the {l2} . [q] where is {a2} ? [a] {l2} . <eos>")
    logic_tasks.append(f"[c] {a1} is in the {l1} . {a2} is in the {l2} . [q] where is {a1} ? [a] {l1} . <eos>")
    logic_tasks.append(f"[c] {a1} is in the {l1} . {a2} is in the {l2} . [q] where is {a2} ? [a] {l2} . <eos>")

# Explicit Alice vs Bob Location Contrast
for _ in range(300):
    l1, l2 = random.sample(locations, 2)
    logic_tasks.append(f"[c] alice went to the {l1} . bob went to the {l2} . [q] where is alice ? [a] {l1} . <eos>")
    logic_tasks.append(f"[c] alice went to the {l1} . bob went to the {l2} . [q] where is bob ? [a] {l2} . <eos>")
    logic_tasks.append(f"[c] bob went to the {l2} . alice went to the {l1} . [q] where is alice ? [a] {l1} . <eos>")
    logic_tasks.append(f"[c] bob went to the {l2} . alice went to the {l1} . [q] where is bob ? [a] {l2} . <eos>")

# 3-Actor Distractors
for _ in range(250):
    a1, a2, a3 = random.sample(actors, 3)
    l1, l2, l3 = random.sample(locations, 3)
    t_a, t_l = random.choice([(a1, l1), (a2, l2), (a3, l3)])
    logic_tasks.append(f"[c] {a1} went to the {l1} . {a2} went to the {l2} . {a3} went to the {l3} . [q] where is {t_a} ? [a] {t_l} . <eos>")

# Movement Updates
for _ in range(250):
    a = random.choice(actors)
    l1, l2 = random.sample(locations, 2)
    logic_tasks.append(f"[c] {a} went to the {l1} . then {a} went to the {l2} . [q] where is {a} ? [a] {l2} . <eos>")


# --- 2. OBJECT LOCATION (Strict 50/50 Clause Balance) ---
# 1-Object Primers
for _ in range(20):
    for i in items:
        for l in locations:
            logic_tasks.append(f"[c] the {i} is in the {l} . [q] where is the {i} ? [a] {l} . <eos>")

# 2-Object Tasks (Equal Clause 1 and Clause 2 Queries)
for _ in range(350):
    i1, i2 = random.sample(items, 2)
    l1, l2 = random.sample(locations, 2)
    logic_tasks.append(f"[c] the {i1} is in the {l1} . the {i2} is in the {l2} . [q] where is the {i1} ? [a] {l1} . <eos>")
    logic_tasks.append(f"[c] the {i1} is in the {l1} . the {i2} is in the {l2} . [q] where is the {i2} ? [a] {l2} . <eos>")
    logic_tasks.append(f"[c] the {i1} is in the {l1} and the {i2} is in the {l2} . [q] where is the {i1} ? [a] {l1} . <eos>")
    logic_tasks.append(f"[c] the {i1} is in the {l1} and the {i2} is in the {l2} . [q] where is the {i2} ? [a] {l2} . <eos>")

# 3-Object Distractors
for _ in range(200):
    i1, i2, i3 = random.sample(items, 3)
    l1, l2, l3 = random.sample(locations, 3)
    t_i, t_l = random.choice([(i1, l1), (i2, l2), (i3, l3)])
    logic_tasks.append(f"[c] the {i1} is in the {l1} . the {i2} is in the {l2} . the {i3} is in the {l3} . [q] where is the {t_i} ? [a] {t_l} . <eos>")


# --- 3. POSSESSION: HAS & FOUND ---
# 1-Entity Primers
for _ in range(20):
    for a in actors:
        for i in items:
            logic_tasks.append(f"[c] {a} has the {i} . [q] what does {a} have ? [a] {i} . <eos>")
            logic_tasks.append(f"[c] {a} found a {i} . [q] what did {a} find ? [a] {i} . <eos>")

# 2-Entity General Pairs
for _ in range(300):
    a1, a2 = random.sample(actors, 2)
    i1, i2 = random.sample(items, 2)
    logic_tasks.append(f"[c] {a1} has the {i1} . {a2} has the {i2} . [q] what does {a1} have ? [a] {i1} . <eos>")
    logic_tasks.append(f"[c] {a1} has the {i1} . {a2} has the {i2} . [q] what does {a2} have ? [a] {i2} . <eos>")
    logic_tasks.append(f"[c] {a1} found a {i1} . {a2} found a {i2} . [q] what did {a1} find ? [a] {i1} . <eos>")
    logic_tasks.append(f"[c] {a1} found a {i1} . {a2} found a {i2} . [q] what did {a2} find ? [a] {i2} . <eos>")
    logic_tasks.append(f"[c] {a1} has the {i1} . {a2} has the {i2} . [q] who has the {i1} ? [a] {a1} . <eos>")

# Explicit Alice vs Bob Possession Contrast
for _ in range(300):
    i1, i2 = random.sample(items, 2)
    logic_tasks.append(f"[c] alice has the {i1} . bob has the {i2} . [q] what does alice have ? [a] {i1} . <eos>")
    logic_tasks.append(f"[c] alice has the {i1} . bob has the {i2} . [q] what does bob have ? [a] {i2} . <eos>")
    logic_tasks.append(f"[c] bob has the {i2} . alice has the {i1} . [q] what does alice have ? [a] {i1} . <eos>")
    logic_tasks.append(f"[c] bob has the {i2} . alice has the {i1} . [q] what does bob have ? [a] {i2} . <eos>")
    logic_tasks.append(f"[c] alice found a {i1} . bob found a {i2} . [q] what did alice find ? [a] {i1} . <eos>")
    logic_tasks.append(f"[c] alice found a {i1} . bob found a {i2} . [q] what did bob find ? [a] {i2} . <eos>")


# --- 4. TRANSFERS & GIVING / IOI ---
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


# --- 5. COLOR BINDING (Strict 50/50 Balance) ---
# 1-Color Primers (Compact grid: 96 tasks)
for a in actors:
    for c in colors:
        for i in items:
            logic_tasks.append(f"[c] {a} has a {c} {i} . [q] what color is the {i} ? [a] {c} . <eos>")

# 2-Color Tasks: Symmetrically query c1 and c2 equally
for _ in range(450):
    a1, a2 = random.sample(actors, 2)
    c1, c2 = random.sample(colors, 2)
    i1, i2 = random.sample(items, 2)
    logic_tasks.append(f"[c] {a1} has a {c1} {i1} . {a2} has a {c2} {i2} . [q] what color is the {i1} ? [a] {c1} . <eos>")
    logic_tasks.append(f"[c] {a1} has a {c1} {i1} . {a2} has a {c2} {i2} . [q] what color is the {i2} ? [a] {c2} . <eos>")
    logic_tasks.append(f"[c] {a1} has a {c1} {i1} and {a2} has a {c2} {i2} . [q] what color is the {i1} ? [a] {c1} . <eos>")
    logic_tasks.append(f"[c] {a1} has a {c1} {i1} and {a2} has a {c2} {i2} . [q] what color is the {i2} ? [a] {c2} . <eos>")


# --- 6. FELINE PERSONA ---
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
# 4. DATASET TOKENIZATION WITH TARGET MASKING
# ---------------------------------------------------------
print("Tokenizing and constructing masked QA tensors...")

def tokenize_str(s):
    return [WORD_TO_ID.get(w, 0) for w in re.findall(r"\[[cqa]\]|<eos>|<unk>|<pad>|\w+|[.]|[?]|[:]", s.lower())]

qa_tensor_x = []
qa_tensor_y = []

pad_id = WORD_TO_ID["<pad>"]
a_token_id = WORD_TO_ID["[a]"]
ans_token_id = WORD_TO_ID["answer"]

for task in all_qa_tasks:
    toks = tokenize_str(task)
    if len(toks) > BLOCK_SIZE:
        continue

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

qa_x_data = torch.stack(qa_tensor_x)
qa_y_data = torch.stack(qa_tensor_y)
print(f"Constructed {len(qa_x_data)} intact, target-masked QA examples.")

story_tokens = tokenize_str(raw_stories)
story_data = torch.tensor(story_tokens, dtype=torch.long)
print(f"Constructed {len(story_data)} background story tokens.")

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
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

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
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.ln_f(x))
        return logits

model = BabyStories().to(device)

TOTAL_STEPS = 8000
WARMUP = 200
BASE_LR, MIN_LR = 2e-3, 2e-4
HALF_BATCH = 16

optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-2)

def get_lr(step):
    if step < WARMUP:
        return BASE_LR * (step + 1) / WARMUP
    ratio = (step - WARMUP) / (TOTAL_STEPS - WARMUP)
    return MIN_LR + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (BASE_LR - MIN_LR)

def get_balanced_batch():
    qa_idx = torch.randint(len(qa_x_data), (HALF_BATCH,))
    x_qa = qa_x_data[qa_idx].to(device)
    y_qa = qa_y_data[qa_idx].to(device)

    st_idx = torch.randint(len(story_data) - BLOCK_SIZE - 1, (HALF_BATCH,))
    x_st = torch.stack([story_data[i : i + BLOCK_SIZE] for i in st_idx]).to(device)
    y_st = torch.stack([story_data[i + 1 : i + BLOCK_SIZE + 1] for i in st_idx]).to(device)
    return x_qa, y_qa, x_st, y_st

print(f"Training for {TOTAL_STEPS} steps with Rebalanced Gradient Weights...")
model.train()
for step in range(TOTAL_STEPS + 1):
    lr = get_lr(step)
    for p in optimizer.param_groups:
        p["lr"] = lr

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
# 6. EVALUATION SUITE
# ---------------------------------------------------------
EVAL_DATABASE = [
    ("Actor Location", "[c] alice went to the kitchen . bob went to the room . [q] where is alice ? [a]", "kitchen ."),
    ("Actor Location", "[c] alice went to the kitchen . bob went to the room . [q] where is bob ? [a]", "room ."),
    ("Actor Location", "[c] bob went to the room . alice went to the kitchen . [q] where is alice ? [a]", "kitchen ."),
    ("Actor Location", "[c] bob went to the room . alice went to the kitchen . [q] where is bob ? [a]", "room ."),
    ("Actor Location", "[c] meow went to the park . miaou went to the garden . [q] where is meow ? [a]", "park ."),
    ("Actor Location", "[c] meow went to the park . miaou went to the garden . [q] where is miaou ? [a]", "garden ."),
    ("Actor Location", "[c] alice went to the kitchen . bob went to the room . meow went to the park . [q] where is meow ? [a]", "park ."),
    ("Actor Location", "[c] alice went to the kitchen . then alice went to the garden . [q] where is alice ? [a]", "garden ."),
    ("Actor Location", "[c] bob went to the park . then bob went to the room . [q] where is bob ? [a]", "room ."),

    ("Object Location", "[c] the ball is in the garden . the box is in the kitchen . [q] where is the ball ? [a]", "garden ."),
    ("Object Location", "[c] the ball is in the garden . the box is in the kitchen . [q] where is the box ? [a]", "kitchen ."),
    ("Object Location", "[c] the milk is in the room . the fish is in the park . [q] where is the milk ? [a]", "room ."),
    ("Object Location", "[c] the milk is in the room . the fish is in the park . [q] where is the fish ? [a]", "park ."),
    ("Object Location", "[c] the box is in the park . the ball is in the room . [q] where is the box ? [a]", "park ."),
    ("Object Location", "[c] the box is in the park . the ball is in the room . [q] where is the ball ? [a]", "room ."),

    ("Possession (has)", "[c] alice has the ball . bob has the fish . [q] what does alice have ? [a]", "ball ."),
    ("Possession (has)", "[c] alice has the ball . bob has the fish . [q] what does bob have ? [a]", "fish ."),
    ("Possession (has)", "[c] bob has the fish . alice has the ball . [q] what does alice have ? [a]", "ball ."),
    ("Possession (has)", "[c] bob has the fish . alice has the ball . [q] what does bob have ? [a]", "fish ."),
    
    ("Possession (found)", "[c] alice found a ball . bob found a fish . [q] what did alice find ? [a]", "ball ."),
    ("Possession (found)", "[c] alice found a ball . bob found a fish . [q] what did bob find ? [a]", "fish ."),
    ("Possession (found)", "[c] meow found a fish . miaou found a ball . [q] what did meow find ? [a]", "fish ."),
    ("Possession (found)", "[c] meow found a fish . miaou found a ball . [q] what did miaou find ? [a]", "ball ."),

    ("Object Transfer", "[c] bob had the ball . bob gave the ball to alice . [q] who has the ball ? [a]", "alice ."),
    ("Object Transfer", "[c] alice had the fish . alice gave the fish to meow . [q] who has the fish ? [a]", "meow ."),
    ("Object Transfer", "[c] meow had the box . meow gave the box to miaou . [q] who has the box ? [a]", "miaou ."),
    ("Object Transfer", "[c] miaou had the milk . miaou gave the milk to bob . [q] who has the milk ? [a]", "bob ."),

    ("Giving (IOI)", "[c] alice gave the ball to miaou . bob gave the box to meow . [q] who got the ball ? [a]", "miaou ."),
    ("Giving (IOI)", "[c] alice gave the ball to miaou . bob gave the box to meow . [q] who got the box ? [a]", "meow ."),
    ("Giving (IOI)", "[c] bob gave the box to meow . alice gave the ball to miaou . [q] who got the ball ? [a]", "miaou ."),
    ("Giving (IOI)", "[c] bob gave the box to meow . alice gave the ball to miaou . [q] who got the box ? [a]", "meow ."),
    ("Giving (IOI)", "[c] alice gave the milk to meow . bob gave the fish to miaou . [q] who got the milk ? [a]", "meow ."),
    ("Giving (IOI)", "[c] alice gave the milk to meow . bob gave the fish to miaou . [q] who got the fish ? [a]", "miaou ."),

    ("Color Binding", "[c] alice has a red box . bob has a blue ball . [q] what color is the box ? [a]", "red ."),
    ("Color Binding", "[c] alice has a red box . bob has a blue ball . [q] what color is the ball ? [a]", "blue ."),
    ("Color Binding", "[c] meow has a yellow ball . miaou has a pink box . [q] what color is the ball ? [a]", "yellow ."),
    ("Color Binding", "[c] meow has a yellow ball . miaou has a pink box . [q] what color is the box ? [a]", "pink ."),
    ("Color Binding", "[c] bob has a green box . alice has a white ball . [q] what color is the box ? [a]", "green ."),
    ("Color Binding", "[c] bob has a green box . alice has a white ball . [q] what color is the ball ? [a]", "white ."),

    ("Feline Persona", "question : what are your names ? answer :", "we are meow and miaou ."),
    ("Feline Persona", "question : who is meow ? answer :", "meow is the pico wh who talks to you on the web ."),
    ("Feline Persona", "question : who is miaou ? answer :", "miaou is the pico w who calculates the words ."),
    ("Feline Persona", "question : what are you ? answer :", "we are two tiny cat ais living in microcontrollers ."),
    ("Feline Persona", "question : are you real cats ? answer :", "we are digital kittens made of code and wifi ."),
    ("Feline Persona", "question : what do you love ? answer :", "we love milk warm sun naps and catching bugs ."),
    ("Feline Persona", "question : can you purr ? answer :", "purr purr purr yes we are purring right now ."),
    ("Feline Persona", "question : what hardware do you run on ? answer :", "we run on two raspberry pi pico chips over wifi ."),

    ("Storytelling", "one day alice and miaou were in the kitchen .", None),
    ("Storytelling", "once upon a time bob went into the green garden .", None),
    ("Storytelling", "the sun was very warm and the little kitten wanted to", None),
    ("Storytelling", "alice and meow found a little box under the tree .", None),
]

def eval_gpu(prompt_text, max_tokens=30, temp=0.05):
    model.eval()
    tokens = [WORD_TO_ID.get(w, 0) for w in re.findall(r"\[[cqa]\]|<eos>|<unk>|<pad>|\w+|[.]|[?]|[:]", prompt_text.lower())]
    if not tokens: tokens = [WORD_TO_ID["meow"]]
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    out_words = []
    with torch.no_grad():
        for _ in range(max_tokens):
            x_cond = x[:, -BLOCK_SIZE:]
            logits = model(x_cond)
            
            if logits.dim() == 3:
                last_logits = logits[:, -1, :]
            else:
                last_logits = logits[-1:, :]

            last_logits[:, WORD_TO_ID["<unk>"]] = -1e9
            last_logits[:, WORD_TO_ID["<pad>"]] = -1e9

            if temp < 0.1:
                next_tok = torch.argmax(last_logits, dim=-1).item()
            else:
                top_vals, top_idx = torch.topk(last_logits / temp, 5)
                probs = F.softmax(top_vals, dim=-1)
                next_tok = top_idx[0, torch.multinomial(probs, 1)[0, 0]].item()

            w = VOCAB[next_tok]
            if w in ("<eos>", "<pad>"): break
            out_words.append(w)
            x = torch.cat((x, torch.tensor([[next_tok]], device=device)), dim=1)
            if temp < 0.1 and w == ".": break
    return " ".join(out_words)

print("\n" + "=" * 74)
print("     RUNNING FINAL BENCHMARK (4L 64-D EXACT VOCAB)     ")
print("=" * 74)

cat_stats = {}
passed = 0; exact = 0
for idx, (cat, prompt, tgt) in enumerate(EVAL_DATABASE, 1):
    if cat not in cat_stats: cat_stats[cat] = {"passed": 0, "total": 0}
    is_story = tgt is None
    out = eval_gpu(prompt, temp=0.7 if is_story else 0.05).strip().lower()
    if not is_story:
        exact += 1
        cat_stats[cat]["total"] += 1
        is_correct = (out == tgt.strip().lower())
        if is_correct:
            passed += 1
            cat_stats[cat]["passed"] += 1
            print(f"[{idx:02d}] PASS | {cat} | {out}")
        else:
            print(f"[{idx:02d}] FAIL | {cat} | {out} (Expected: {tgt})")
    else:
        has_leak = "[q]" in out or "[a]" in out
        status = "LEAK" if has_leak else "STORY"
        print(f"[{idx:02d}] {status} | {cat} | {out}")

print("\n" + "=" * 74)
print("                          CATEGORY SCORECARD                           ")
print("=" * 74)
for cat, s in cat_stats.items():
    if s["total"] > 0:
        pct = (s["passed"] / s["total"]) * 100
        bar = ("#" * int(pct // 10)).ljust(10, "-")
        print(f"{cat:<28} : {s['passed']:02d}/{s['total']:02d} [{bar}] {pct:5.1f}%")

print("-" * 74)
print(f"TOTAL SCORE: {passed}/{exact} ({(passed/exact)*100:.1f}%)")
print("=" * 74)

# ---------------------------------------------------------
# 7. EXPORT BINARIES
# ---------------------------------------------------------
def dump_t(t, f): f.write(t.detach().cpu().to(torch.float32).numpy().tobytes())

with open("pico1 weights.model", "wb") as f:
    dump_t(model.tok_emb.weight, f)
    dump_t(model.pos_emb.weight, f)
    for i in [0, 1]:
        for p in model.layers[i].parameters(): dump_t(p, f)

with open("pico2 weights.model", "wb") as f:
    for i in [2, 3]:
        for p in model.layers[i].parameters(): dump_t(p, f)
    for p in model.ln_f.parameters(): dump_t(p, f)
    dump_t(model.lm_head.weight, f)
    dump_t(model.lm_head.bias, f)

with open("vocab.py", "w") as f: f.write(f"VOCAB = {repr(VOCAB)}\n")

files.download("pico1 weights.model")
files.download("pico2 weights.model")
files.download("vocab.py")