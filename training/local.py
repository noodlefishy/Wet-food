import torch, struct, requests, re, math, os, json, sys, torch.nn as nn
from torch.nn import functional as F
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from curriculum import generate_atomic_sample, split_qa

with open("config/model.json", "r") as f: mod_cfg = json.load(f)
cfg_path = "config/world.json" if os.path.exists("config/world.json") else "config/logic.json"
with open(cfg_path, "r") as f: log_cfg = json.load(f)
torch.set_num_threads(4)


VOCAB_SIZE = mod_cfg["architecture"]["vocab_size"]
BLOCK_SIZE = mod_cfg["architecture"]["block_size"]
EMBED_DIM  = mod_cfg["architecture"]["embed_dim"]
NUM_HEADS  = mod_cfg["architecture"]["num_heads"]
NUM_LAYERS = mod_cfg["architecture"]["num_layers"]
HEAD_DIM   = EMBED_DIM // NUM_HEADS
DEVICE = mod_cfg["training"]["device"]

MODELS_DIR = mod_cfg["paths"]["models_dir"]
STORIES_FILE = mod_cfg["paths"]["stories_file"]
QUESTIONS_FILE = mod_cfg["paths"]["questions_file"]
os.makedirs(MODELS_DIR, exist_ok=True)

if os.path.exists(STORIES_FILE):
    with open(STORIES_FILE, "r", encoding="utf-8", errors="ignore") as f:
        raw_stories = f.read().lower()
else:
    response = requests.get(mod_cfg["dataset"]["url"], headers={"User-Agent": "Mozilla/5.0"}, stream=True)
    with open(STORIES_FILE, "wb") as f:
        for chunk in response.iter_content(chunk_size=512 * 1024):
            f.write(chunk)
            if f.tell() >= 6 * 1024 * 1024: break
    with open(STORIES_FILE, "r", encoding="utf-8", errors="ignore") as f:
        raw_stories = f.read().lower()

for target, names in mod_cfg["dataset"]["name_replacements"].items():
    raw_stories = re.sub(r"\b(" + "|".join(names) + r")\b", target, raw_stories)

TARGET_LOGIC = mod_cfg["dataset"].get("logic_samples_target", 40000)
all_qa_tasks = [generate_atomic_sample(log_cfg) for _ in range(TARGET_LOGIC)]

eval_benchmarks = {}
categories = [c["name"] for c in log_cfg["categories"]] if "categories" in log_cfg else list(log_cfg["queries"].keys())
for cat in categories:
    eval_benchmarks[cat] = []
    for _ in range(15):
        p, t = split_qa(generate_atomic_sample(log_cfg, target_category=cat))
        eval_benchmarks[cat].append({"q": p, "target": t})

eval_benchmarks["Storytelling"] = [{"q": p, "target": None} for p in [
    "one day alice and felix were in the kitchen .",
    "once upon a time bob went into the green garden .",
    "the sun was very warm and the little kitten wanted to",
    "alice and luna found a little box under the tree ."
]]

with open(QUESTIONS_FILE, "w") as f: json.dump(eval_benchmarks, f, indent=2)

CAT_CORE_WORDS = mod_cfg["dataset"]["core_words"]
all_words = re.findall(r"\[[cqa]\]|<eos>|<unk>|<pad>|\w+|[.]|[?]|[:]", raw_stories + " " + " ".join(all_qa_tasks))
counts = Counter(all_words)
for w in CAT_CORE_WORDS: counts.pop(w, None)
VOCAB = CAT_CORE_WORDS + [w for w, _ in counts.most_common(VOCAB_SIZE - len(CAT_CORE_WORDS))]
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}

def tokenize_str(s):
    return [WORD_TO_ID.get(w, 0) for w in re.findall(r"\[[cqa]\]|<eos>|<unk>|<pad>|\w+|[.]|[?]|[:]", s.lower())]

qa_tensor_x, qa_tensor_y = [], []
pad_id, a_token_id = WORD_TO_ID["<pad>"], WORD_TO_ID["[a]"]

for task in all_qa_tasks:
    toks = tokenize_str(task)
    if len(toks) > BLOCK_SIZE: continue
    x = toks + [pad_id] * (BLOCK_SIZE - len(toks))
    y = [-100] * BLOCK_SIZE
    ans_start = next((idx for idx, t in enumerate(toks) if t == a_token_id), -1)
    if ans_start != -1:
        for idx in range(ans_start, len(toks) - 1): y[idx] = toks[idx + 1]
    qa_tensor_x.append(torch.tensor(x, dtype=torch.long))
    qa_tensor_y.append(torch.tensor(y, dtype=torch.long))

qa_x_data = torch.stack(qa_tensor_x)
qa_y_data = torch.stack(qa_tensor_y)
story_data = torch.tensor(tokenize_str(raw_stories), dtype=torch.long)

class MQA(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(EMBED_DIM, EMBED_DIM + 2 * HEAD_DIM)
        self.out_proj = nn.Linear(EMBED_DIM, EMBED_DIM)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q = qkv[:, :, :C].view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = qkv[:, :, C : C + HEAD_DIM].view(B, T, 1, HEAD_DIM).transpose(1, 2).expand(-1, NUM_HEADS, -1, -1)
        v = qkv[:, :, C + HEAD_DIM :].view(B, T, 1, HEAD_DIM).transpose(1, 2).expand(-1, NUM_HEADS, -1, -1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.sa = MQA()
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.mlp = nn.Sequential(nn.Linear(EMBED_DIM, 4 * EMBED_DIM), nn.ReLU(), nn.Linear(4 * EMBED_DIM, EMBED_DIM))

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.mlp(self.ln2(x))

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
        for layer in self.layers: x = layer(x)
        return self.lm_head(self.ln_f(x))

model = BabyStories().to(DEVICE)
compiled_model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=mod_cfg["training"]["base_lr"], weight_decay=mod_cfg["training"]["weight_decay"])


TOTAL_STEPS = mod_cfg["training"]["total_steps"]
WARMUP = mod_cfg["training"]["warmup_steps"]
BASE_LR, MIN_LR = mod_cfg["training"]["base_lr"], mod_cfg["training"]["min_lr"]
HALF_BATCH = mod_cfg["training"]["half_batch"]


num_chunks = (len(story_data) - 1) // BLOCK_SIZE
story_x_chunks = story_data[:num_chunks * BLOCK_SIZE].view(num_chunks, BLOCK_SIZE)
story_y_chunks = story_data[1 : num_chunks * BLOCK_SIZE + 1].view(num_chunks, BLOCK_SIZE)

def get_balanced_batch():
    qa_idx = torch.randint(len(qa_x_data), (HALF_BATCH,))
    st_idx = torch.randint(num_chunks, (HALF_BATCH,))
    return qa_x_data[qa_idx].to(DEVICE), qa_y_data[qa_idx].to(DEVICE), story_x_chunks[st_idx].to(DEVICE), story_y_chunks[st_idx].to(DEVICE)

def get_lr(step):
    if step < WARMUP: return BASE_LR * (step + 1) / WARMUP
    return MIN_LR + 0.5 * (1.0 + math.cos(math.pi * (step - WARMUP) / (TOTAL_STEPS - WARMUP))) * (BASE_LR - MIN_LR)


model.train()
tStep = None
try:
    for step in range(TOTAL_STEPS + 1):
        tStep = step
        for p in optimizer.param_groups: p["lr"] = get_lr(step)

        x_qa, y_qa, x_st, y_st = get_balanced_batch()
        logits_all = model(torch.cat([x_qa, x_st], dim=0))
        loss_qa = F.cross_entropy(logits_all[:HALF_BATCH].view(-1, VOCAB_SIZE), y_qa.view(-1), ignore_index=-100)
        loss_st = F.cross_entropy(logits_all[HALF_BATCH:].view(-1, VOCAB_SIZE), y_st.view(-1))

        decay_start = int(TOTAL_STEPS * 0.70)
        current_w_st = 0.3 - ((step - decay_start) / (TOTAL_STEPS - decay_start)) * 0.25 if step > decay_start else 0.3
        loss = (2.5 * loss_qa) + (max(current_w_st, 0.05) * loss_st)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), mod_cfg["training"]["grad_clip"])
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step:5d}/{TOTAL_STEPS} | Total: {loss.item():.4f} | QA: {loss_qa.item():.4f} | Story: {loss_st.item():.4f}")
except KeyboardInterrupt:
    print(f"\n[!] Training stopped early at step {tStep}. Saving current weights...")

def dump_t(t, f):
    d = t.detach().cpu()
    if d.ndim == 2:
        for r in d:
            m = r.abs().max().item()
            s = m / 127.0 if m > 0 else 1.0
            f.write(struct.pack('f', s))
            f.write(torch.clamp(torch.round(r / s), -128, 127).to(torch.int8).numpy().tobytes())
    else:
        f.write(d.to(torch.float32).numpy().tobytes())

with open(os.path.join(MODELS_DIR, "pico1 weights.model"), "wb") as f:
    dump_t(model.tok_emb.weight, f); dump_t(model.pos_emb.weight, f)
    for i in [0, 1, 2]:
        for p in model.layers[i].parameters(): dump_t(p, f)

with open(os.path.join(MODELS_DIR, "pico2 weights.model"), "wb") as f:
    for i in [3, 4, 5]:
        for p in model.layers[i].parameters(): dump_t(p, f)
    for p in model.ln_f.parameters(): dump_t(p, f)
    dump_t(model.lm_head.weight, f); dump_t(model.lm_head.bias, f)

with open(os.path.join(MODELS_DIR, "vocab.py"), "w") as f: f.write(f"VOCAB = {repr(VOCAB)}\n")