import os, sys, json, re, torch, torch.nn.functional as F
projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, projectRoot)
from benchmarks.fastEvaluator import model, VOCAB, WORD_TO_ID, BLOCK_SIZE, padId, device

model.eval()
TOKEN_REGEX = r"\[[cqa]\]|<eos>|<unk>|<pad>|'s|\w+|[.,?!:]"

def tokenize(s):
    for c in [".", "?", "!", ",", ":"]: s = s.replace(c, f" {c} ")
    s = s.replace("'s", " 's ")
    return [WORD_TO_ID.get(w, 0) for w in re.findall(TOKEN_REGEX, s.lower())]

print("=" * 65)
print("     picoAI MICRO-TESTER (PROBABILITY INSPECTOR)")
print("=" * 65)
print("Type a premise and question. Type 'exit' to quit.\n")

while True:
    try:
        raw = input("Prompt > ")
        if raw.lower().strip() in ("exit", "quit"): break
        if not raw.strip(): continue
        
        if "[c]" not in raw: raw = f"[c] {raw}"
        if "[q]" in raw and not raw.endswith("[a]"): raw = raw.rstrip("?.: ") + " ? [a]"
        
        toks = tokenize(raw)
        x = torch.tensor([toks + [padId] * (BLOCK_SIZE - len(toks))], dtype=torch.long, device=device)
        
        print("\n--- GENERATION TRACE ---")
        curr_len = len(toks)
        
        with torch.no_grad():
            for step in range(20):
                logits = model(x)
                pos = curr_len + step - 1
                if pos >= BLOCK_SIZE - 1: break
                
                probs = F.softmax(logits[0, pos], dim=-1)
                top_probs, top_indices = torch.topk(probs, 3)
                
                predId = top_indices[0].item()
                predWord = VOCAB[predId]
                
                trace = []
                for i in range(3):
                    w = VOCAB[top_indices[i].item()]
                    p = top_probs[i].item() * 100
                    trace.append(f"{w} ({p:.1f}%)")
                
                print(f"Step {step+1:02d} | {' | '.join(trace)}")
                
                if predWord in ("<eos>", "<pad>"): break
                x[0, pos + 1] = predId
                
        print("-" * 24 + "\n")
    except (KeyboardInterrupt, EOFError):
        break