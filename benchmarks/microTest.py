import os, sys, re, math, torch
import torch.nn.functional as F
try: import readline
except ImportError: pass

projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, projectRoot)

from benchmarks.fastEvaluator import (
    model, VOCAB, WORD_TO_ID, BLOCK_SIZE, padId, device,
    NUM_HEADS, HEAD_DIM, cosC, sinC, applyRope, NUM_LAYERS
)

model.eval()
TOKEN_REGEX = r"\[[cqa]\]|<eos>|<unk>|<pad>|'s|\w+|[.,?!:]"

def tokenize(s):
    for c in [".", "?", "!", ",", ":"]: s = s.replace(c, f" {c} ")
    s = s.replace("'s", " 's ")
    return [WORD_TO_ID.get(w, 0) for w in re.findall(TOKEN_REGEX, s.lower())]

activation_cache = {i: None for i in range(NUM_LAYERS)}

def get_attention_hook(layer_idx):
    def hook(module, inp, out):
        x = inp[0]
        B, T, C = x.shape
        qkv = module.qkvProj(x)
        q, k, _ = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        
        q = applyRope(q, cosC, sinC)
        k = applyRope(k, cosC, sinC)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        
        attn_matrix = F.softmax(scores, dim=-1)
        avg_head_attn = attn_matrix[0].mean(dim=0)
        activation_cache[layer_idx] = avg_head_attn.detach().cpu()
    return hook

for i in range(NUM_LAYERS):
    model.layers[i].sa.register_forward_hook(get_attention_hook(i))

def make_bar(pct, width=12):
    filled = int(width * (pct / 100))
    return "█" * filled + "░" * (width - filled)

print("=" * 75)
print("     picoAI MICRO-TESTER (MEOW L2 vs MIAOU L5 COMPARISON)")
print("=" * 75)
print("Type a QCA prompt. Type 'exit' to quit.\n")

while True:
    try:
        raw = input("Prompt > ")
        if raw.lower().strip() in ("exit", "quit"): break
        if not raw.strip(): continue
        
        if "[q]" not in raw: raw = f"[q] {raw}"
        if "[c]" not in raw: raw = f"{raw} [c]"
        if not raw.endswith("[a]"): raw = raw.rstrip("?.: ") + " ? [a]"
        
        toks = tokenize(raw)
        x = torch.tensor([toks + [padId] * (BLOCK_SIZE - len(toks))], dtype=torch.long, device=device)
        
        print("\n" + "=" * 75)
        curr_len = len(toks)
        
        with torch.no_grad():
            for step in range(3): 
                logits = model(x)
                pos = curr_len + step - 1
                if pos >= BLOCK_SIZE - 1: break
                
                probs = F.softmax(logits[0, pos], dim=-1)
                top_probs, top_indices = torch.topk(probs, 3)
                predId = top_indices[0].item()
                predWord = VOCAB[predId]
                
                print(f"STEP {step+1:02d} | PREDICTION: {predWord.upper()}")
                for i in range(3):
                    w = VOCAB[top_indices[i].item()]
                    p = top_probs[i].item() * 100
                    print(f"  {w:>10} | {make_bar(p, 10)} {p:5.1f}%")
                
                # Grab Attention for MEOW (Layer 2) and MIAOU (Layer 5)
                meow_attn = activation_cache[2]
                miaou_attn = activation_cache[5]
                
                meow_step = meow_attn[pos, :pos+1] if meow_attn is not None and pos < meow_attn.shape[0] else None
                miaou_step = miaou_attn[pos, :pos+1] if miaou_attn is not None and pos < miaou_attn.shape[0] else None
                
                print(f"\n  [SIDE-BY-SIDE ATTENTION COMPARISON]")
                print(f"  {'TOKEN':<12} | {'MEOW (L2 - Context)':<25} | {'MIAOU (L5 - Logic Gate)':<25}")
                print("  " + "-" * 71)
                
                for i in range(pos + 1):
                    tok_id = x[0, i].item()
                    word = VOCAB[tok_id]
                    
                    m_val = (meow_step[i].item() * 100) if meow_step is not None else 0.0
                    mi_val = (miaou_step[i].item() * 100) if miaou_step is not None else 0.0
                    
                    # Only show tokens that have at least 0.5% attention in either board to save space
                    if m_val >= 0.5 or mi_val >= 0.5:
                        m_str = f"{make_bar(m_val, 8)} {m_val:4.1f}%"
                        mi_str = f"{make_bar(mi_val, 8)} {mi_val:4.1f}%"
                        print(f"  {word[:12]:<12} | {m_str:<25} | {mi_str:<25}")
                
                print("=" * 75 + "\n")
                
                if predWord in ("<eos>", "<pad>"): break
                x[0, pos + 1] = predId
                
    except (KeyboardInterrupt, EOFError):
        break