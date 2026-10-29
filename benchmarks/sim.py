import os, sys, re, torch, random
try: import readline
except ImportError: pass

projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, projectRoot)

from benchmarks.fastEvaluator import model, VOCAB, WORD_TO_ID, BLOCK_SIZE, padId, dotId, eosId, device

TOKEN_REGEX = r"\[[cqa]\]|<eos>|<unk>|<pad>|'s|\w+|[.,?!:]"

def tokenize(s):
    for c in [".", "?", "!", ",", ":"]: s = s.replace(c, f" {c} ")
    s = s.replace("'s", " 's ")
    return [WORD_TO_ID.get(w, 0) for w in re.findall(TOKEN_REGEX, s.lower())]

model.eval()

def generate(promptTxt, maxTokens=20):
    isLog = any(t in promptTxt for t in ("[c]", "[q]", "[a]")) or any(promptTxt.lower().strip().startswith(w) for w in ("who", "what", "where", "is", "does", "why", "how", "question"))
    p = promptTxt.strip()
    if isLog:
        if "[c]" not in p and not p.startswith("question"): p = f"[c] {p}"
        if "[q]" in p and not p.endswith("[a]"): p = p.rstrip("?.: ") + " ? [a]"

    toks = tokenize(p)
    if not toks: return ""

    promptLen = len(toks)
    if promptLen >= BLOCK_SIZE:
        toks = toks[:BLOCK_SIZE - 1]
        promptLen = len(toks)

    padded = toks + [padId] * (BLOCK_SIZE - promptLen)
    x = torch.tensor([padded], dtype=torch.long, device=device)
    
    out_words = []
    with torch.no_grad():
        for step in range(maxTokens):
            logits = model(x)
            pos = promptLen + step - 1
            if pos >= BLOCK_SIZE - 1: break
            
            lg = logits[0, pos].clone()
            lg[WORD_TO_ID["<unk>"]] = lg[WORD_TO_ID["<pad>"]] = -1e9
            
            if isLog:
                predId = lg.argmax().item()
            else:
                # Safe repetition penalty + temperature for open stories
                for pastId in set(toks + [WORD_TO_ID.get(w, 0) for w in out_words[-8:]]):
                    lg[pastId] -= 1.8
                probs = torch.softmax(lg / 0.7, dim=-1)
                predId = torch.multinomial(probs, 1).item()

            predWord = VOCAB[predId] # type: ignore
            if predWord in ("<eos>", "<pad>"): break
            out_words.append(predWord)
            if isLog and predWord == ".": break
            
            # Feed prediction back into sequence
            x[0, pos + 1] = predId

    return " ".join(out_words)

if __name__ == "__main__":
    print("=" * 65)
    print("      WET FOOD LOCAL SIMULATOR (32-BIT HARDWARE VERIFIED)      ")
    print("=" * 65)
    print("Type a prompt or 'exit' to quit.\n")
    while True:
        try:
            line = input("> ")
            if not line.strip(): continue
            if line.strip().lower() in ["exit", "quit"]: break
            print(f"  {generate(line)}\n")
        except (KeyboardInterrupt, EOFError): break