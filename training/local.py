import torch, struct, requests, re, math, os, json, sys, torch.nn as nn
from torch.nn import functional as F
from collections import Counter

torch.set_num_threads(4)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from curriculum import generate_atomic_sample, split_qa

with open("config/model.json", "r") as f: mod_cfg = json.load(f)
cfg_path = "config/world.json" if os.path.exists("config/world.json") else "config/logic.json"
with open(cfg_path, "r") as f: log_cfg = json.load(f)

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
    with open(STORIES_FILE, "r", encoding="utf-8", errors="ignore") as f: raw_stories = f.read().lower()
else:
    response = requests.get(mod_cfg["dataset"]["url"], headers={"User-Agent": "Mozilla/5.0"}, stream=True)
    with open(STORIES_FILE, "wb") as f:
        for chunk in response.iter_content(chunk_size=512 * 1024):
            f.write(chunk)
            if f.tell() >= 6 * 1024 * 1024: break
    with open(STORIES_FILE, "r", encoding="utf-8", errors="ignore") as f: raw_stories = f.read().lower()

for target, names in mod_cfg["dataset"]["name_replacements"].items():
    raw_stories = re.sub(r"\b(" + "|".join(names) + r")\b", target, raw_stories)

TARGET_LOGIC = 100000 
print(f"Pre-caching {TARGET_LOGIC} logic tasks into RAM...")
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
all_words = re.findall(r"\[[cqa]\]|<eos>|<unk>|<pad>|\w+|[.]|[?]|[:]", raw_stories + " " + " ".join(all_qa_tasks[:5000]))
counts = Counter(all_words)
for w in CAT_CORE_WORDS: counts.pop(w, None)
VOCAB = CAT_CORE_WORDS + [w for w, _ in counts.most_common(VOCAB_SIZE - len(CAT_CORE_WORDS))]
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}
pad_id, a_token_id = WORD_TO_ID["<pad>"], WORD_TO_ID["[a]"]

def tokenize_str(s):
    return [WORD_TO_ID.get(w, 0) for w in re.findall(r"\[[cqa]\]|<eos>|<unk>|<pad>|\w+|[.]|[?]|[:]", s.lower())]

qa_tensor_x, qa_tensor_y = [], []
for task in all_qa_tasks:
    toks = tokenize_str(task)
    if len(toks) > BLOCK_SIZE: toks = toks[:BLOCK_SIZE]
    x = toks + [pad_id] * (BLOCK_SIZE - len(toks))
    y = [-100] * BLOCK_SIZE
    ans_start = next((idx for idx, t in enumerate(toks) if t == a_token_id), -1)
    if ans_start != -1:
        for idx in range(ans_start, len(toks) - 1): y[idx] = toks[idx + 1]
    qa_tensor_x.append(torch.tensor(x, dtype=torch.long))
    qa_tensor_y.append(torch.tensor(y, dtype=torch.long))

qa_x_data = torch.stack(qa_tensor_x).to(DEVICE)
qa_y_data = torch.stack(qa_tensor_y).to(DEVICE)

story_tokens = torch.tensor(tokenize_str(raw_stories), dtype=torch.long)
num_st_chunks = (len(story_tokens) - 1) // BLOCK_SIZE
st_x_data = story_tokens[:num_st_chunks * BLOCK_SIZE].view(num_st_chunks, BLOCK_SIZE).to(DEVICE)
st_y_data = story_tokens[1 : num_st_chunks * BLOCK_SIZE + 1].view(num_st_chunks, BLOCK_SIZE).to(DEVICE)

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

raw_model = BabyStories().to(DEVICE)
optimizer = torch.optim.AdamW(raw_model.parameters(), lr=0.001, weight_decay=mod_cfg["training"]["weight_decay"])

try: compiled_model = torch.compile(raw_model)
except Exception: compiled_model = raw_model

# 3. Supercharged Batch Scaling
TOTAL_STEPS = 6000
WARMUP = 500
BASE_LR, MIN_LR = 0.001, 0.0001
BATCH_QA, BATCH_ST = 32, 4

def get_lr(step):
    if step < WARMUP: return BASE_LR * (step + 1) / WARMUP
    return MIN_LR + 0.5 * (1.0 + math.cos(math.pi * (step - WARMUP) / (TOTAL_STEPS - WARMUP))) * (BASE_LR - MIN_LR)

def get_batch():
    qa_idx = torch.randint(len(qa_x_data), (BATCH_QA,))
    st_idx = torch.randint(num_st_chunks, (BATCH_ST,))
    return qa_x_data[qa_idx], qa_y_data[qa_idx], st_x_data[st_idx], st_y_data[st_idx]

raw_model.train()
try:
    for step in range(TOTAL_STEPS + 1):
        for p in optimizer.param_groups: p["lr"] = get_lr(step)

        x_qa, y_qa, x_st, y_st = get_batch()
        logits_all = compiled_model(torch.cat([x_qa, x_st], dim=0))
        loss_qa = F.cross_entropy(logits_all[:BATCH_QA].view(-1, VOCAB_SIZE), y_qa.view(-1), ignore_index=-100)
        loss_st = F.cross_entropy(logits_all[BATCH_QA:].view(-1, VOCAB_SIZE), y_st.view(-1))

        decay_start = int(TOTAL_STEPS * 0.70)
        current_w_st = 0.3 - ((step - decay_start) / (TOTAL_STEPS - decay_start)) * 0.25 if step > decay_start else 0.3
        loss = (2.5 * loss_qa) + (max(current_w_st, 0.05) * loss_st)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), mod_cfg["training"]["grad_clip"])
        optimizer.step()

        if step % 250 == 0:
            print(f"Step {step:5d}/{TOTAL_STEPS} | Total: {loss.item():.4f} | QA: {loss_qa.item():.4f} | Story: {loss_st.item():.4f}")
except KeyboardInterrupt:
    print(f"\n[!] Training stopped early. Saving current weights...")

def dump_t(t, f, quantize=False):
    d = t.detach().cpu()
    if d.ndim == 2 and quantize:
        for r in d:
            m = r.abs().max().item()
            s = m / 127.0 if m > 0 else 1.0
            f.write(struct.pack('f', s))
            f.write(torch.clamp(torch.round(r / s), -128, 127).to(torch.int8).numpy().tobytes())
    else:
        f.write(d.to(torch.float32).numpy().tobytes())

with open(os.path.join(MODELS_DIR, "pico1 weights.model"), "wb") as f:
    dump_t(raw_model.tok_emb.weight, f, quantize=True)
    dump_t(raw_model.pos_emb.weight, f, quantize=True)
    for i in [0, 1, 2]:
        for p in raw_model.layers[i].parameters(): dump_t(p, f, quantize=True)

with open(os.path.join(MODELS_DIR, "pico2 weights.model"), "wb") as f:
    for i in [3, 4, 5]:
        for p in raw_model.layers[i].parameters(): dump_t(p, f, quantize=True)
    for p in raw_model.ln_f.parameters(): dump_t(p, f, quantize=True)
    dump_t(raw_model.lm_head.weight, f, quantize=True)
    dump_t(raw_model.lm_head.bias, f, quantize=True)

with open(os.path.join(MODELS_DIR, "vocab.py"), "w") as f: f.write(f"VOCAB = {repr(VOCAB)}\n")