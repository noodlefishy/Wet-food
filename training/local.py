import torch, struct, requests, re, math, os, json, sys, time, torch.nn as nn
from torch.nn import functional as F
from collections import Counter

torch.set_num_threads(4)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from curriculum import generate_atomic_sample, split_qa

with open("config/model.json", "r") as f: mod_cfg = json.load(f)
cfg_path = "config/world.json" if os.path.exists("config/world.json") else "config/logic.json"
with open(cfg_path, "r") as f: log_cfg = json.load(f)

VOCAB_SIZE   = mod_cfg["architecture"]["vocab_size"]
BLOCK_SIZE   = mod_cfg["architecture"]["block_size"]
EMBED_DIM    = mod_cfg["architecture"]["embed_dim"]
NUM_HEADS    = mod_cfg["architecture"]["num_heads"]
NUM_LAYERS   = mod_cfg["architecture"]["num_layers"]       # 8 Layers Total (3 Meow, 5 Miaou)
MEOW_LAYERS  = mod_cfg["architecture"]["meow_layers"]      # 3 Layers
MIAOU_LAYERS = mod_cfg["architecture"]["miaou_layers"]     # 5 Layers
HEAD_DIM     = EMBED_DIM // NUM_HEADS
DEVICE       = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else mod_cfg["training"].get("device", "cpu"))

MODELS_DIR = mod_cfg["paths"]["models_dir"]
STORIES_FILE = mod_cfg["paths"]["stories_file"]
QUESTIONS_FILE = mod_cfg["paths"]["questions_file"]
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("firmware", exist_ok=True)

print("=" * 65)
print(f" TRAINING 8-LAYER RoPE ENGINE ({MEOW_LAYERS}L MEOW + {MIAOU_LAYERS}L MIAOU, NO POS EMB)")
print("=" * 65)

if os.path.exists(STORIES_FILE):
    print(f"[*] Reading TinyStories from cached {STORIES_FILE}...")
    with open(STORIES_FILE, "r", encoding="utf-8", errors="ignore") as f: raw_stories = f.read().lower()
else:
    print("[*] Downloading TinyStories dataset...")
    response = requests.get(mod_cfg["dataset"]["url"], headers={"User-Agent": "Mozilla/5.0"}, stream=True)
    with open(STORIES_FILE, "wb") as f:
        for chunk in response.iter_content(chunk_size=512 * 1024):
            f.write(chunk)
            if f.tell() >= 6 * 1024 * 1024: break
    with open(STORIES_FILE, "r", encoding="utf-8", errors="ignore") as f: raw_stories = f.read().lower()

for target, names in mod_cfg["dataset"]["name_replacements"].items():
    raw_stories = re.sub(r"\b(" + "|".join(names) + r")\b", target, raw_stories)

TARGET_LOGIC = mod_cfg["dataset"].get("logic_samples_target", 50000)
print(f"[*] Generating {TARGET_LOGIC} anti-shortcut combinatorial tasks...")
all_qa_tasks = [generate_atomic_sample(log_cfg) for _ in range(TARGET_LOGIC)]

eval_benchmarks = {}
categories = [c["name"] for c in log_cfg["categories"]]
for cat in categories:
    eval_benchmarks[cat] = []
    for _ in range(15):
        p, t = split_qa(generate_atomic_sample(log_cfg, target_category=cat, inject_distractor=False))
        eval_benchmarks[cat].append({"q": p, "target": t})

eval_benchmarks["Storytelling"] = [{"q": p, "target": None} for p in [
    "one day alice and felix were in the kitchen .",
    "once upon a time bob went into the green garden .",
    "the sun was very warm and the little kitten wanted to",
    "alice and luna found a little box under the tree ."
]]
with open(QUESTIONS_FILE, "w") as f: json.dump(eval_benchmarks, f, indent=2)

CAT_CORE_WORDS = mod_cfg["dataset"]["core_words"]
all_words = re.findall(r"\[[cqa]\]|<eos>|<unk>|<pad>|\w+|[.]|[?]|[:]", raw_stories + " " + " ".join(all_qa_tasks[:8000]))
counts = Counter(all_words)
for w in CAT_CORE_WORDS: counts.pop(w, None)
VOCAB = CAT_CORE_WORDS + [w for w, _ in counts.most_common(VOCAB_SIZE - len(CAT_CORE_WORDS))]
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}
pad_id, a_token_id = WORD_TO_ID["<pad>"], WORD_TO_ID["[a]"]

def tokenize_str(s):
    return [WORD_TO_ID.get(w, 0) for w in re.findall(r"\[[cqa]\]|<eos>|<unk>|<pad>|\w+|[.]|[?]|[:]", s.lower())]

print("[*] Tokenizing logic and story streams...")
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

# =====================================================================
# ROTARY POSITION EMBEDDING (RoPE) - ZERO ABSOLUTE POSITION MEMORIZATION
# =====================================================================
def precompute_rope_freqs(head_dim, max_seq_len=BLOCK_SIZE):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()

cos_cached, sin_cached = precompute_rope_freqs(HEAD_DIM, BLOCK_SIZE)

def apply_rope(x, cos, sin):
    # x shape: [B, num_heads, T, HEAD_DIM]
    T = x.shape[2]
    c = cos[:T].unsqueeze(0).unsqueeze(1).to(x.device)
    s = sin[:T].unsqueeze(0).unsqueeze(1).to(x.device)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot1 = x1 * c - x2 * s
    rot2 = x1 * s + x2 * c
    out = torch.empty_like(x)
    out[..., 0::2] = rot1
    out[..., 1::2] = rot2
    return out

class RoPE_MQA(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(EMBED_DIM, EMBED_DIM + 2 * HEAD_DIM)
        self.out_proj = nn.Linear(EMBED_DIM, EMBED_DIM)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q = qkv[:, :, :C].view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = qkv[:, :, C : C + HEAD_DIM].view(B, T, 1, HEAD_DIM).transpose(1, 2)
        v = qkv[:, :, C + HEAD_DIM :].view(B, T, 1, HEAD_DIM).transpose(1, 2)
        
        # Apply RoPE directly to Query and Key representations
        q = apply_rope(q, cos_cached, sin_cached)
        k = apply_rope(k, cos_cached, sin_cached)
        
        # Expand single Key and Value head across Query heads
        k = k.expand(-1, NUM_HEADS, -1, -1)
        v = v.expand(-1, NUM_HEADS, -1, -1)
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.sa = RoPE_MQA()
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.mlp = nn.Sequential(nn.Linear(EMBED_DIM, 4 * EMBED_DIM), nn.ReLU(), nn.Linear(4 * EMBED_DIM, EMBED_DIM))

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class BabyStories(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        # NOTICE: ABSOLUTELY ZERO learned absolute pos_emb!
        self.layers = nn.ModuleList([Block() for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, idx):
        # Pure semantic lookup: Position encoding is strictly handled via relative RoPE rotations in attention!
        x = self.tok_emb(idx)
        for layer in self.layers: x = layer(x)
        return self.lm_head(self.ln_f(x))

raw_model = BabyStories().to(DEVICE)

TOTAL_STEPS = mod_cfg["training"].get("total_steps", 25000)
WARMUP = mod_cfg["training"].get("warmup_steps", 1000)
BASE_LR = mod_cfg["training"].get("base_lr", 0.0015)
MIN_LR = mod_cfg["training"].get("min_lr", 0.0001)
BATCH_QA = mod_cfg["training"].get("batch_qa", 16)
BATCH_ST = mod_cfg["training"].get("batch_st", 10)

optimizer = torch.optim.AdamW(raw_model.parameters(), lr=BASE_LR, weight_decay=mod_cfg["training"]["weight_decay"])

def get_lr(step):
    if step < WARMUP: return BASE_LR * (step + 1) / WARMUP
    return MIN_LR + 0.5 * (1.0 + math.cos(math.pi * (step - WARMUP) / (TOTAL_STEPS - WARMUP))) * (BASE_LR - MIN_LR)

def get_batch():
    qa_idx = torch.randint(len(qa_x_data), (BATCH_QA,))
    st_idx = torch.randint(num_st_chunks, (BATCH_ST,))
    return qa_x_data[qa_idx], qa_y_data[qa_idx], st_x_data[st_idx], st_y_data[st_idx]

print(f"\n[*] Training configuration: {TOTAL_STEPS} steps | Batch: {BATCH_QA} QA + {BATCH_ST} Story")
raw_model.train()
t_start = time.time()

try:
    for step in range(TOTAL_STEPS + 1):
        for p in optimizer.param_groups: p["lr"] = get_lr(step)

        x_qa, y_qa, x_st, y_st = get_batch()
        logits_all = raw_model(torch.cat([x_qa, x_st], dim=0))
        loss_qa = F.cross_entropy(logits_all[:BATCH_QA].view(-1, VOCAB_SIZE), y_qa.view(-1), ignore_index=-100)
        loss_st = F.cross_entropy(logits_all[BATCH_QA:].view(-1, VOCAB_SIZE), y_st.view(-1))

        decay_start = int(TOTAL_STEPS * 0.70)
        current_w_st = 0.35 - ((step - decay_start) / (TOTAL_STEPS - decay_start)) * 0.20 if step > decay_start else 0.35
        loss = (2.0 * loss_qa) + (max(current_w_st, 0.15) * loss_st)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), mod_cfg["training"]["grad_clip"])
        optimizer.step()

        if step % 500 == 0:
            elapsed = time.time() - t_start
            s_per_step = elapsed / max(step, 1)
            eta_min = (TOTAL_STEPS - step) * s_per_step / 60
            print(f"Step {step:5d}/{TOTAL_STEPS} | Total: {loss.item():.4f} | QA: {loss_qa.item():.4f} | Story: {loss_st.item():.4f} | ETA: {eta_min:.1f}m", flush=True)
except KeyboardInterrupt:
    print(f"\n[!] Training paused. Saving current checkpoint...")

print("\n[*] Quantizing weights (INT8 W8A32) and exporting for Meow (3L) & Miaou (5L)...")

def dump_t(t, f, quantize=True):
    d = t.detach().cpu()
    if d.ndim == 2 and quantize:
        for r in d:
            m = r.abs().max().item()
            s = m / 127.0 if m > 0 else 1.0
            f.write(struct.pack('f', s))
            f.write(torch.clamp(torch.round(r / s), -128, 127).to(torch.int8).numpy().tobytes())
    else:
        f.write(d.to(torch.float32).numpy().tobytes())

p1_path = os.path.join(MODELS_DIR, "pico1 weights.model")
p2_path = os.path.join(MODELS_DIR, "pico2 weights.model")

# Meow gets: tok_emb + Layers 0, 1, 2
with open(p1_path, "wb") as f:
    dump_t(raw_model.tok_emb.weight, f, quantize=True)
    for i in range(MEOW_LAYERS):
        for p in raw_model.layers[i].parameters(): dump_t(p, f, quantize=True)

# Miaou gets: Layers 3, 4, 5, 6, 7 + ln_f + lm_head
with open(p2_path, "wb") as f:
    for i in range(MEOW_LAYERS, NUM_LAYERS):
        for p in raw_model.layers[i].parameters(): dump_t(p, f, quantize=True)
    for p in raw_model.ln_f.parameters(): dump_t(p, f, quantize=True)
    dump_t(raw_model.lm_head.weight, f, quantize=True)
    dump_t(raw_model.lm_head.bias, f, quantize=True)

vocab_code = f"VOCAB = {repr(VOCAB)}\n"
with open(os.path.join(MODELS_DIR, "vocab.py"), "w") as f: f.write(vocab_code)
with open("firmware/vocab.py", "w") as f: f.write(vocab_code)
if os.path.exists("models/vocab.py"):
    with open("models/vocab.py", "w") as f: f.write(vocab_code)

print(f"[✓] Meow Weights (pico1, {MEOW_LAYERS} Layers): {os.path.getsize(p1_path)/1024:.1f} KB -> Saved to {p1_path}")
print(f"[✓] Miaou Weights (pico2, {MIAOU_LAYERS} Layers): {os.path.getsize(p2_path)/1024:.1f} KB -> Saved to {p2_path}")
print(f"[✓] Vocab: Saved to models/local/vocab.py & firmware/vocab.py")
print("Training script ready! Run whenever you want to train the model.")
