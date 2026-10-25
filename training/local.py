# =====================================================================
# WET FOOD: BLAZING FAST LOCAL MAC TRAINING (JSON OPTIMIZED)
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
import json

# ---------------------------------------------------------
# 1. LOAD CONFIGURATIONS
# ---------------------------------------------------------
print("Loading configurations...")
with open("config/model.json", "r") as f:
    mod_cfg = json.load(f)

with open("config/logic.json", "r") as f:
    log_cfg = json.load(f)

# Extract Architecture settings
VOCAB_SIZE = mod_cfg["architecture"]["vocab_size"]
BLOCK_SIZE = mod_cfg["architecture"]["block_size"]
EMBED_DIM  = mod_cfg["architecture"]["embed_dim"]
NUM_HEADS  = mod_cfg["architecture"]["num_heads"]
NUM_LAYERS = mod_cfg["architecture"]["num_layers"]

# Extract Paths and Device
MODELS_DIR = mod_cfg["paths"]["models_dir"]
STORIES_FILE = mod_cfg["paths"]["stories_file"]
QUESTIONS_FILE = mod_cfg["paths"]["questions_file"]
DEVICE = mod_cfg["training"]["device"]

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

print("==================================================")
print(f" Training Verified {NUM_LAYERS}L {EMBED_DIM}-D Engine on: {DEVICE.upper()}")
print("==================================================")

# ---------------------------------------------------------
# 2. FETCH FAIRY TALE FOUNDATION
# ---------------------------------------------------------
if os.path.exists(STORIES_FILE):
    print(f"Using cached {STORIES_FILE}...")
    with open(STORIES_FILE, "r", encoding="utf-8", errors="ignore") as f:
        raw_stories = f.read().lower()
else:
    print("Downloading TinyStories...")
    url = mod_cfg["dataset"]["url"]
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, stream=True)
    with open(STORIES_FILE, "wb") as f:
        for chunk in response.iter_content(chunk_size=512 * 1024):
            f.write(chunk)
            if f.tell() >= 6 * 1024 * 1024: break
    with open(STORIES_FILE, "r", encoding="utf-8", errors="ignore") as f:
        raw_stories = f.read().lower()

for target, names in mod_cfg["dataset"]["name_replacements"].items():
    pattern = r"\b(" + "|".join(names) + r")\b"
    raw_stories = re.sub(pattern, target, raw_stories)


# ---------------------------------------------------------
# 3. COMPILE REASONING TASKS & GENERATE EVAL QUESTIONS
# ---------------------------------------------------------
TARGET_LOGIC_SAMPLES = mod_cfg["dataset"].get("logic_samples_target", 25000)
print(f"Compiling strictly balanced logic dataset ({TARGET_LOGIC_SAMPLES} targeted)...")

logic_tasks = []
eval_benchmarks = {} # <--- NOW A DICTIONARY
pools = log_cfg["pools"]
pair_pools = log_cfg.get("pair_pools", {})

def instantiate_template(template, p_pools, pp_pools):
    matches = re.findall(r'\{([a-z]+)(\d+)(?:_([a-z]+))?\}', template)
    replacements = {}
    
    for p_key, p_values in p_pools.items():
        indices = set(m[1] for m in matches if m[0] == p_key)
        if indices:
            sampled = random.sample(p_values, len(indices))
            for idx, val in zip(sorted(list(indices)), sampled):
                replacements[f"{p_key}{idx}"] = val
                
    for p_key, p_values in pp_pools.items():
        indices = set(m[1] for m in matches if m[0] == p_key)
        if indices:
            sampled = random.sample(p_values, len(indices))
            for idx, val_dict in zip(sorted(list(indices)), sampled):
                for sub_key, sub_val in val_dict.items():
                    replacements[f"{p_key}{idx}_{sub_key}"] = sub_val
                    
    return template.format(**replacements)

total_weight = sum(cat.get("weight", 1.0) for cat in log_cfg["categories"])

for cat in log_cfg["categories"]:
    weight = cat.get("weight", 1.0)
    cat_name = cat["name"]
    templates = cat["templates"]
    
    # Initialize the array for this category
    eval_benchmarks[cat_name] = [] 

    cat_total_reps = int((weight / total_weight) * TARGET_LOGIC_SAMPLES)
    reps_per_template = cat_total_reps // len(templates)
    remainder = cat_total_reps % len(templates)
    
    # A. Build the Training Data
    for i, template in enumerate(templates):
        reps = reps_per_template + (1 if i < remainder else 0)
        for _ in range(reps):
            logic_tasks.append(instantiate_template(template, pools, pair_pools))
            
    # B. Generate the Evaluation Benchmarks (5 per category)
    for _ in range(5):
        eval_template = random.choice(templates)
        eval_str = instantiate_template(eval_template, pools, pair_pools)
        if " [a] " in eval_str:
            q_part, a_part = eval_str.split(" [a] ")
            eval_benchmarks[cat_name].append({
                "q": q_part + " [a]",
                "target": a_part.replace("<eos>", "").strip()
            })

# Inject Personas (Training & Eval)
cat_identities = log_cfg["personas"]
all_qa_tasks = logic_tasks + (cat_identities * log_cfg["persona_reps"])

eval_benchmarks["Persona"] = []
for p in cat_identities:
    if " answer : " in p:
        q_part, a_part = p.split(" answer : ")
        eval_benchmarks["Persona"].append({
            "q": q_part + " answer :",
            "target": a_part.replace("<eos>", "").strip()
        })

# Append Open-Ended Storytelling Checks to Eval
eval_benchmarks["Storytelling"] = []
story_prompts = [
    "one day alice and miaou were in the kitchen .",
    "once upon a time bob went into the green garden .",
    "the sun was very warm and the little kitten wanted to",
    "alice and meow found a little box under the tree ."
]
for sp in story_prompts:
    eval_benchmarks["Storytelling"].append({"q": sp, "target": None})

# Dump strictly formatted JSON questions
with open(QUESTIONS_FILE, "w") as f:
    json.dump(eval_benchmarks, f, indent=2)

total_q_count = sum(len(qs) for qs in eval_benchmarks.values())
print(f"Generated {total_q_count} dynamic evaluation questions -> {QUESTIONS_FILE}")

random.seed(42)
random.shuffle(all_qa_tasks)

# ---------------------------------------------------------
# 4. VOCABULARY GENERATION
# ---------------------------------------------------------
CAT_CORE_WORDS = mod_cfg["dataset"]["core_words"]

all_words = re.findall(r"\[[cqa]\]|<eos>|<unk>|<pad>|\w+|[.]|[?]|[:]", raw_stories + " " + " ".join(all_qa_tasks))
counts = Counter(all_words)
for w in CAT_CORE_WORDS:
    counts.pop(w, None)
remaining = VOCAB_SIZE - len(CAT_CORE_WORDS)
top_words = [w for w, _ in counts.most_common(remaining)]
VOCAB = CAT_CORE_WORDS + top_words
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}

# ---------------------------------------------------------
# 5. DATASET TOKENIZATION (STRICT CPU PROCESSING)
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

qa_x_data = torch.stack(qa_tensor_x)
qa_y_data = torch.stack(qa_tensor_y)

story_tokens = tokenize_str(raw_stories)
story_data = torch.tensor(story_tokens, dtype=torch.long)

# ---------------------------------------------------------
# 6. MODEL ARCHITECTURE & BALANCED TRAINING
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
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=DEVICE))
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_f(x))

model = BabyStories().to(DEVICE)

TOTAL_STEPS = mod_cfg["training"]["total_steps"]
WARMUP = mod_cfg["training"]["warmup_steps"]
BASE_LR = mod_cfg["training"]["base_lr"]
MIN_LR = mod_cfg["training"]["min_lr"]
HALF_BATCH = mod_cfg["training"]["half_batch"]
W_DECAY = mod_cfg["training"]["weight_decay"]
G_CLIP = mod_cfg["training"]["grad_clip"]

optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=W_DECAY)

def get_lr(step):
    if step < WARMUP: return BASE_LR * (step + 1) / WARMUP
    ratio = (step - WARMUP) / (TOTAL_STEPS - WARMUP)
    return MIN_LR + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (BASE_LR - MIN_LR)

def get_balanced_batch():
    qa_idx = torch.randint(len(qa_x_data), (HALF_BATCH,))
    x_qa = qa_x_data[qa_idx]
    y_qa = qa_y_data[qa_idx]

    st_idx = torch.randint(len(story_data) - BLOCK_SIZE - 1, (HALF_BATCH,))
    x_st = torch.stack([story_data[i : i + BLOCK_SIZE] for i in st_idx])
    y_st = torch.stack([story_data[i + 1 : i + BLOCK_SIZE + 1] for i in st_idx])
    
    return x_qa.to(DEVICE), y_qa.to(DEVICE), x_st.to(DEVICE), y_st.to(DEVICE)

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

    loss_qa = F.cross_entropy(logits_qa.view(-1, VOCAB_SIZE), y_qa.view(-1), ignore_index=-100)
    loss_st = F.cross_entropy(logits_st.view(-1, VOCAB_SIZE), y_st.view(-1))
    
    w_qa = mod_cfg["training"]["loss_weights"]["qa"]
    w_st = mod_cfg["training"]["loss_weights"]["story"]
    loss = (w_qa * loss_qa) + (w_st * loss_st)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), G_CLIP)
    optimizer.step()

    if step % 500 == 0:
        print(f"Step {step:5d}/{TOTAL_STEPS} | Total: {loss.item():.4f} | QA: {loss_qa.item():.4f} | Story: {loss_st.item():.4f}")

# ---------------------------------------------------------
# 7. EXPORT BINARIES DIRECTLY TO MODELS FOLDER
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

vocab_path = os.path.join(MODELS_DIR, "vocab.py")
with open(vocab_path, "w") as f: 
    f.write(f"VOCAB = {repr(VOCAB)}\n")

print(f"Meow Weights (pico1): {os.path.getsize(path1)/1024:.1f} KB -> Saved to {path1}")
print(f"Miaou Weights (pico2): {os.path.getsize(path2)/1024:.1f} KB -> Saved to {path2}")
print(f"Vocab List: Saved to ./{vocab_path}")
print("\nTraining Complete!")