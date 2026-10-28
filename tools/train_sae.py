import os, sys, math, json, struct, time, torch
import torch.nn as nn
from torch.nn import functional as F

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

try:
    from firmware.vocab import VOCAB
except ImportError:
    from models.local.vocab import VOCAB

WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}
EMBED_DIM = 80
BLOCK_SIZE = 64
HEAD_DIM = EMBED_DIM // 4

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Initializing SAE training pipeline on device: {device.upper()}")

# =====================================================================
# 1. LOAD LAYERS 0-2 FROM PICO 1 WEIGHTS
# =====================================================================
class MQA(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(EMBED_DIM, EMBED_DIM + 2 * HEAD_DIM)
        self.out_proj = nn.Linear(EMBED_DIM, EMBED_DIM)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q = qkv[:, :, :C].view(B, T, 4, HEAD_DIM).transpose(1, 2)
        k = qkv[:, :, C : C + HEAD_DIM].view(B, T, 1, HEAD_DIM).transpose(1, 2).expand(-1, 4, -1, -1)
        v = qkv[:, :, C + HEAD_DIM :].view(B, T, 1, HEAD_DIM).transpose(1, 2).expand(-1, 4, -1, -1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.sa, self.ln2 = nn.LayerNorm(EMBED_DIM), MQA(), nn.LayerNorm(EMBED_DIM)
        self.mlp = nn.Sequential(nn.Linear(EMBED_DIM, 4 * EMBED_DIM), nn.ReLU(), nn.Linear(4 * EMBED_DIM, EMBED_DIM))

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class MeowLayers012(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(1024, EMBED_DIM)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.layers = nn.ModuleList([Block() for _ in range(3)])

    def forward(self, idx):
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(idx.size(1), device=idx.device))
        for layer in self.layers:
            x = layer(x)
        return x

p1_path = os.path.join(project_root, "models", "local", "pico1 weights.model")
if not os.path.exists(p1_path):
    print(f"[!] Cannot find {p1_path}")
    sys.exit(1)

model_l02 = MeowLayers012().to(device)

def load_param(f, tensor, quantized=True):
    import numpy as np
    if tensor.ndim == 2 and quantized:
        rows, cols = tensor.shape
        data = np.empty((rows, cols), dtype=np.float32)
        for r in range(rows):
            s = struct.unpack('f', f.read(4))[0]
            data[r] = np.fromfile(f, dtype=np.int8, count=cols).astype(np.float32) * s
        tensor.data.copy_(torch.from_numpy(data))
    else:
        tensor.data.copy_(torch.from_numpy(np.fromfile(f, dtype=np.float32, count=tensor.numel())).view_as(tensor))

with open(p1_path, "rb") as f1:
    load_param(f1, model_l02.tok_emb.weight, quantized=True)
    load_param(f1, model_l02.pos_emb.weight, quantized=True)
    for i in range(3):
        for p in model_l02.layers[i].parameters():
            load_param(f1, p, quantized=True)

model_l02.eval()
print("[✓] Loaded Meow Layers 0–2 successfully.")

# =====================================================================
# 2. COLLECT LAYER 2 ACTIVATION DATASET
# =====================================================================
stories_file = os.path.join(project_root, "stories.txt")
tokens_pool = []

if os.path.exists(stories_file):
    print(f"[*] Reading training stories from {stories_file}...")
    with open(stories_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().lower()
    words = re_words = [WORD_TO_ID.get(w, 0) for w in text.split()]
    tokens_pool.extend(words)
else:
    print("[*] Generating synthetic story samples for profiling...")
    import random
    names = ["bob", "alice", "felix", "luna"]
    places = ["garden", "kitchen", "park", "forest", "room", "castle"]
    items = ["ball", "box", "key", "dragon", "cake", "cup"]
    verbs = ["went", "ran", "jumped", "found", "saw", "climbed"]
    for _ in range(5000):
        s = f"{random.choice(names)} {random.choice(verbs)} to the {random.choice(places)} and saw a {random.choice(items)} ."
        tokens_pool.extend([WORD_TO_ID.get(w, 0) for w in s.split()])

print(f"[*] Preparing {min(len(tokens_pool), 150000)} tokens for activation harvesting...")
chunk_len = 64
num_chunks = min(len(tokens_pool) // chunk_len, 2000)
input_ids = torch.tensor(tokens_pool[: num_chunks * chunk_len], dtype=torch.long).view(num_chunks, chunk_len)

activations = []
batch_size = 64
with torch.no_grad():
    for b in range(0, num_chunks, batch_size):
        batch = input_ids[b : b + batch_size].to(device)
        # Pass through Layers 0, 1, 2 to get thought vectors at the Wi-Fi bridge
        out_layer2 = model_l02(batch)
        activations.append(out_layer2.view(-1, EMBED_DIM).cpu())
        print(f"\r  -> Harvesting thought vectors: {min(b + batch_size, num_chunks)}/{num_chunks} chunks", end="")

activations = torch.cat(activations, dim=0).to(device)
print(f"\n[✓] Collected {activations.size(0):,} Layer 2 vectors ({activations.size(0) * EMBED_DIM * 4 / 1024 / 1024:.1f} MB in VRAM).")

# =====================================================================
# 3. TOP-K SPARSE AUTOENCODER ARCHITECTURE
# =====================================================================
class TopKSAE(nn.Module):
    def __init__(self, d_in=80, d_hidden=320, k=12):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.k = k

        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.w_enc = nn.Linear(d_in, d_hidden)
        self.w_dec = nn.Linear(d_hidden, d_in, bias=False)

        # Initialize decoder columns to unit vectors
        nn.init.orthogonal_(self.w_dec.weight)
        self.w_enc.weight.data = self.w_dec.weight.data.t().clone()
        self.w_enc.bias.data.zero_()

    @torch.no_grad()
    def normalize_decoder(self):
        # Enforce unit norm on decoder columns (w_dec.weight has shape [d_in, d_hidden])
        norms = torch.norm(self.w_dec.weight.data, p=2, dim=0, keepdim=True)
        self.w_dec.weight.data.div_(norms.clamp(min=1e-8))

    def encode(self, x):
        x_cent = x - self.b_dec
        pre_act = self.w_enc(x_cent)
        # Strict Top-K sparsity selection
        topk_vals, topk_indices = torch.topk(pre_act, self.k, dim=-1)
        topk_vals = F.relu(topk_vals)
        f = torch.zeros_like(pre_act).scatter_(-1, topk_indices, topk_vals)
        return f

    def decode(self, f):
        return self.w_dec(f) + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f

# 80-D -> 320-D (4x expansion) with K=12 active features (96.25% sparsity!)
D_HIDDEN = 320
K_SPARSE = 12
sae = TopKSAE(d_in=EMBED_DIM, d_hidden=D_HIDDEN, k=K_SPARSE).to(device)
optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3, betas=(0.9, 0.999))

# =====================================================================
# 4. TRAIN THE SPARSE AUTOENCODER
# =====================================================================
print("\n" + "=" * 65)
print(f" TRAINING TOP-{K_SPARSE} SPARSE AUTOENCODER (80-D -> {D_HIDDEN}-D) ")
print("=" * 65)

sae_batch_size = 512
num_samples = activations.size(0)
epochs = 8
t_start = time.time()

for epoch in range(epochs):
    perm = torch.randperm(num_samples)
    total_loss = 0.0
    steps = 0

    for i in range(0, num_samples, sae_batch_size):
        idx = perm[i : i + sae_batch_size]
        x_batch = activations[idx]

        x_hat, f_acts = sae(x_batch)
        loss = F.mse_loss(x_hat, x_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sae.normalize_decoder()

        total_loss += loss.item()
        steps += 1

    # Measure Explained Variance: 1 - Var(Residual) / Var(Data)
    with torch.no_grad():
        x_sample = activations[:2048]
        x_sample_hat, _ = sae(x_sample)
        total_var = torch.var(x_sample, dim=0).sum()
        residual_var = torch.var(x_sample - x_sample_hat, dim=0).sum()
        exp_var = (1.0 - residual_var / total_var).item() * 100

    print(f"Epoch {epoch+1:02d}/{epochs:02d} | MSE: {total_loss/steps:.4f} | Explained Variance: {exp_var:5.1f}% | Active K: {K_SPARSE}/{D_HIDDEN}")

print(f"[✓] SAE training complete in {time.time() - t_start:.1f}s.")

# =====================================================================
# 5. AUTOMATED FEATURE INTERPRETATION (THE MONOSEMANTIC DICTIONARY)
# =====================================================================
print("\n[*] Profiling all 320 features across the vocabulary...")
vocab_tensor = torch.arange(1024, device=device).unsqueeze(1)
with torch.no_grad():
    vocab_thought_vectors = model_l02(vocab_tensor).squeeze(1) # [1024, 80]
    sae_vocab_acts = sae.encode(vocab_thought_vectors).cpu()  # [1024, 320]

feature_dict = {}
decoder_matrix = sae.w_dec.weight.data.cpu() # [80, 320]

for feat_idx in range(D_HIDDEN):
    # Find words with highest activation on this feature
    col = sae_vocab_acts[:, feat_idx]
    top_indices = torch.topk(col, 4).indices.tolist()
    top_words = [VOCAB[w] for w in top_indices if col[w] > 0.0]
    
    if not top_words:
        top_words = ["<residual>"]
    
    label = f"Feature #{feat_idx}: [{', '.join(top_words)}]"
    feature_dict[feat_idx] = {
        "words": top_words,
        "max_activation": float(col.max().item()),
        "vector": [round(v, 4) for v in decoder_matrix[:, feat_idx].tolist()]
    }

# Display a sample of discovered pure features
print("\n" + "=" * 65)
print(" SAMPLE OF MONOSEMANTIC FEATURES UNCOVERED BY SAE ")
print("=" * 65)
discovered = [f for f in feature_dict.values() if len(f["words"]) >= 2 and f["words"][0] != "<residual>"]
for f in discovered[:12]:
    print(f"  • {f['words'][0].upper():<12} -> Fires on: {', '.join(f['words'])}")

# =====================================================================
# 6. EXPORT LEAN 8-BIT QUANTIZED STEERING VECTORS (DROP PRECISION)
# =====================================================================
out_path = os.path.join(project_root, "firmware", "cortex.json")
print(f"\n[*] Updating {out_path} with compact 8-bit quantized SAE vectors...")

try:
    with open(out_path, "r") as f:
        cortex_payload = json.load(f)
except Exception:
    cortex_payload = {}

# Curated sparks for the UI: 6 words per category = ~36 words total
COMPACT_CATALYSTS = {
    "Mythical & Royalty": ["dragon", "castle", "magic", "monster", "giant", "king"],
    "Creatures & Wildlife": ["wolf", "fox", "bear", "lion", "rabbit", "bird"],
    "Sensory & Comfort": ["honey", "cookies", "cake", "soft", "fluffy", "warm"],
    "Atmosphere & Settings": ["garden", "kitchen", "park", "forest", "woods", "river"],
    "Mischief & Action": ["mud", "messy", "silly", "broke", "bite", "trouble"],
    "Action Verbs": ["jump", "climb", "flew", "fly", "run", "chased"]
}

cortex_payload["catalysts"] = COMPACT_CATALYSTS
cortex_payload["sae_vectors"] = {}
cortex_payload["sae_features"] = {}

# Flatten curated spark words
curated_words = set(w for words in COMPACT_CATALYSTS.values() for w in words)

for word in curated_words:
    if word in WORD_TO_ID:
        tid = WORD_TO_ID[word]
        acts = sae_vocab_acts[tid]
        best_feat = int(acts.argmax().item())
        raw_vec = feature_dict[best_feat]["vector"]
        
        # INT8 Fixed-point quantization:
        # Normalize to unit length and round to 2 decimal places to minimize JSON characters
        norm = math.sqrt(sum(v * v for v in raw_vec)) or 1.0
        compact_vec = [round(v / norm, 2) for v in raw_vec]
        
        cortex_payload["sae_vectors"][word] = compact_vec
        cortex_payload["sae_features"][word] = f"F#{best_feat} ({feature_dict[best_feat]['words'][0]})"

cortex_payload["vectors"] = cortex_payload["sae_vectors"]

# Ensure theme radar is also compact (2 decimal places)
if "radar" in cortex_payload:
    for theme, vec in cortex_payload["radar"].items():
        cortex_payload["radar"][theme] = [round(v, 2) for v in vec]

with open(out_path, "w") as f:
    json.dump(cortex_payload, f, separators=(',', ':'))  # Minified JSON without whitespace

size_kb = os.path.getsize(out_path) / 1024
print(f"[✓] Successfully compressed cortex.json to {size_kb:.1f} KB (fits easily on LittleFS)!")