import json, os, struct
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

HEADER_MAGIC = b"PICO"
HEADER_FMT = "<4s7I"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 32 bytes

def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def precompute_rope(head_dim, max_len):
    inv = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(max_len).float(), inv)
    return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):
    T = x.shape[2]
    c = cos[:T].unsqueeze(0).unsqueeze(1).to(x.device)
    s = sin[:T].unsqueeze(0).unsqueeze(1).to(x.device)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out

class RoPE_MHA(nn.Module):
    def __init__(self, embed_dim, num_heads, cos_buf, sin_buf):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkvProj = nn.Linear(embed_dim, 3 * embed_dim)
        self.outProj = nn.Linear(embed_dim, embed_dim)
        self.register_buffer("cos_cached", cos_buf, persistent=False)
        self.register_buffer("sin_cached", sin_buf, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkvProj(x).chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, self.cos_cached, self.sin_cached)
        k = apply_rope(k, self.cos_cached, self.sin_cached)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.outProj(out.transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, cos_buf, sin_buf):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.sa = RoPE_MHA(embed_dim, num_heads, cos_buf, sin_buf)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        return x + self.mlp(self.ln2(x + self.sa(self.ln1(x))))

class BabyStories(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, block_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_layers = num_layers
        self.block_size = block_size

        cos_buf, sin_buf = precompute_rope(self.head_dim, block_size)
        self.register_buffer("cos_cached", cos_buf, persistent=False)
        self.register_buffer("sin_cached", sin_buf, persistent=False)

        self.tokEmb = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            Block(embed_dim, num_heads, cos_buf, sin_buf) for _ in range(num_layers)
        ])
        self.lnF = nn.LayerNorm(embed_dim)
        self.lmHead = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        x = self.tokEmb(idx)
        for layer in self.layers:
            x = layer(x)
        return self.lmHead(self.lnF(x))

def write_model_header(f, vocab_size, embed_dim, num_heads, total_layers, node_layers, block_size):
    f.write(struct.pack(
        HEADER_FMT,
        HEADER_MAGIC, 1,
        vocab_size, embed_dim, num_heads, total_layers, node_layers, block_size
    ))

def read_model_header(f, fallback_cfg=None, default_node_layers=3):
    raw = f.read(HEADER_SIZE)
    if len(raw) == HEADER_SIZE and raw[:4] == HEADER_MAGIC:
        _, version, v_size, e_dim, n_heads, tot_l, node_l, b_size = struct.unpack(HEADER_FMT, raw)
        return {
            "version": version,
            "vocab_size": v_size,
            "embed_dim": e_dim,
            "num_heads": n_heads,
            "num_layers": tot_l,
            "node_layers": node_l,
            "block_size": b_size,
        }
    f.seek(0)
    arch = (fallback_cfg or {}).get("architecture", {})
    return {
        "version": 0,
        "vocab_size": arch.get("vocab_size", 1024),
        "embed_dim": arch.get("embed_dim", 96),
        "num_heads": arch.get("num_heads", 4),
        "num_layers": arch.get("num_layers", 6),
        "node_layers": default_node_layers,
        "block_size": arch.get("block_size", 64),
    }

def dump_tensor(t, f, quantize=True):
    d = t.detach().cpu()
    if d.ndim == 2 and quantize:
        for r in d:
            m = r.abs().max().item()
            s = m / 127.0 if m > 0 else 1.0
            f.write(struct.pack("f", s))
            f.write(torch.clamp(torch.round(r / s), -128, 127).to(torch.int8).numpy().tobytes())
    else:
        f.write(d.to(torch.float32).numpy().tobytes())

def load_param(f, tensor, quantized=True):
    if tensor.ndim == 2 and quantized:
        rows, cols = tensor.shape
        data = np.empty((rows, cols), dtype=np.float32)
        for r in range(rows):
            s = struct.unpack("f", f.read(4))[0]
            data[r] = np.frombuffer(f.read(cols), dtype=np.int8).astype(np.float32) * s
        tensor.data.copy_(torch.from_numpy(data))
    else:
        raw_buf = np.frombuffer(f.read(tensor.numel() * 4), dtype=np.float32).copy()
        tensor.data.copy_(torch.from_numpy(raw_buf).view_as(tensor))

def load_distributed_model(project_root, device=None):
    if device is None:
        device = get_best_device()

    cfg_path = os.path.join(project_root, "config", "model.json")
    with open(cfg_path, "r") as f:
        mod_cfg = json.load(f)

    import importlib
    from models.local import vocab as vocab_mod
    importlib.reload(vocab_mod)
    vocab = vocab_mod.VOCAB
    word_to_id = {w: i for i, w in enumerate(vocab)}

    p1 = os.path.join(project_root, "models", "local", "pico1 weights.model")
    p2 = os.path.join(project_root, "models", "local", "pico2 weights.model")

    with open(p1, "rb") as f1, open(p2, "rb") as f2:
        h1 = read_model_header(f1, mod_cfg, mod_cfg["architecture"].get("meow_layers", 3))
        h2 = read_model_header(f2, mod_cfg, mod_cfg["architecture"].get("miaou_layers", 3))

        meow_layers = h1["node_layers"]
        num_layers = h1["num_layers"]
        vocab_size = h1["vocab_size"]
        embed_dim = h1["embed_dim"]
        num_heads = h1["num_heads"]
        block_size = h1["block_size"]

        model = BabyStories(vocab_size, embed_dim, num_heads, num_layers, block_size).to(device)
        load_param(f1, model.tokEmb.weight)
        for i in range(meow_layers):
            for p in model.layers[i].parameters():
                load_param(f1, p)

        for i in range(meow_layers, num_layers):
            for p in model.layers[i].parameters():
                load_param(f2, p)
        for p in model.lnF.parameters():
            load_param(f2, p)
        load_param(f2, model.lmHead.weight)
        load_param(f2, model.lmHead.bias)

    model.eval()
    return {
        "model": model,
        "cfg": mod_cfg,
        "vocab": vocab,
        "word_to_id": word_to_id,
        "device": device,
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "head_dim": embed_dim // num_heads,
        "num_layers": num_layers,
        "meow_layers": meow_layers,
        "miaou_layers": h2["node_layers"],
        "block_size": block_size,
    }