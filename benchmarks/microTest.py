import math, os, sys
import torch
import torch.nn.functional as F
try:
    import readline  # noqa: F401
except ImportError:
    pass

projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, projectRoot)

from pico.model import apply_rope, load_distributed_model
from pico.tokenizer import canonicalize_to_qca, tokenize

ctx = load_distributed_model(projectRoot)
model = ctx["model"]
VOCAB = ctx["vocab"]
WORD_TO_ID = ctx["word_to_id"]
BLOCK_SIZE = ctx["block_size"]
NUM_HEADS = ctx["num_heads"]
HEAD_DIM = ctx["head_dim"]
NUM_LAYERS = ctx["num_layers"]
MEOW_LAYERS = ctx["meow_layers"]
device = ctx["device"]
padId = WORD_TO_ID.get("<pad>", 1)

activation_cache = {i: None for i in range(NUM_LAYERS)}

def get_attention_hook(layer_idx):
    def hook(module, inp, out):
        x = inp[0]
        B, T, _ = x.shape
        q, k, _ = module.qkvProj(x).chunk(3, dim=-1)
        q = apply_rope(q.view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2), module.cos_cached, module.sin_cached)
        k = apply_rope(k.view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2), module.cos_cached, module.sin_cached)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        attn_matrix = F.softmax(scores, dim=-1)
        activation_cache[layer_idx] = attn_matrix[0].mean(dim=0).detach().cpu()
    return hook

for i in range(NUM_LAYERS):
    model.layers[i].sa.register_forward_hook(get_attention_hook(i))

def make_bar(pct, width=12):
    filled = int(width * (pct / 100))
    return "█" * filled + "░" * (width - filled)

meow_idx = MEOW_LAYERS - 1
miaou_idx = NUM_LAYERS - 1

print("=" * 75)
print(f"     picoAI MICRO-TESTER (MEOW L{meow_idx} vs MIAOU L{miaou_idx} COMPARISON)")
print("=" * 75)
print("Type a QCA prompt. Type 'exit' to quit.\n")

while True:
    try:
        raw = input("Prompt > ")
        if raw.lower().strip() in ("exit", "quit"):
            break
        if not raw.strip():
            continue

        qca, _ = canonicalize_to_qca(raw)
        toks = tokenize(qca, WORD_TO_ID)
        if not toks:
            continue
        x = torch.tensor([toks + [padId] * (BLOCK_SIZE - len(toks))], dtype=torch.long, device=device)

        print("\n" + "=" * 75)
        curr_len = len(toks)

        with torch.no_grad():
            for step in range(3):
                logits = model(x)
                pos = curr_len + step - 1
                if pos >= BLOCK_SIZE - 1:
                    break

                probs = F.softmax(logits[0, pos], dim=-1)
                top_probs, top_indices = torch.topk(probs, 3)
                predId = top_indices[0].item()
                predWord = VOCAB[predId]

                print(f"STEP {step+1:02d} | PREDICTION: {predWord.upper()}")
                for i in range(3):
                    w = VOCAB[top_indices[i].item()]
                    p = top_probs[i].item() * 100
                    print(f"  {w:>10} | {make_bar(p, 10)} {p:5.1f}%")

                meow_attn = activation_cache[meow_idx]
                miaou_attn = activation_cache[miaou_idx]
                meow_step = meow_attn[pos, : pos + 1] if meow_attn is not None and pos < meow_attn.shape[0] else None
                miaou_step = miaou_attn[pos, : pos + 1] if miaou_attn is not None and pos < miaou_attn.shape[0] else None

                print("\n  [SIDE-BY-SIDE ATTENTION COMPARISON]")
                print(f"  {'TOKEN':<12} | {f'MEOW (L{meow_idx} - Context)':<25} | {f'MIAOU (L{miaou_idx} - Logic Gate)':<25}")
                print("  " + "-" * 71)

                for i in range(pos + 1):
                    word = VOCAB[x[0, i].item()]
                    m_val = (meow_step[i].item() * 100) if meow_step is not None else 0.0
                    mi_val = (miaou_step[i].item() * 100) if miaou_step is not None else 0.0
                    if m_val >= 0.5 or mi_val >= 0.5:
                        m_str = f"{make_bar(m_val, 8)} {m_val:4.1f}%"
                        mi_str = f"{make_bar(mi_val, 8)} {mi_val:4.1f}%"
                        print(f"  {word[:12]:<12} | {m_str:<25} | {mi_str:<25}")

                print("=" * 75 + "\n")

                if predWord in ("<eos>", "<pad>"):
                    break
                x[0, pos + 1] = predId

    except (KeyboardInterrupt, EOFError):
        break