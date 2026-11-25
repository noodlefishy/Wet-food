import os, sys
import torch
import readline 


projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, projectRoot)

from pico.model import load_distributed_model
from pico.tokenizer import canonicalize_to_qca, tokenize

_ctx = load_distributed_model(projectRoot)
model = _ctx["model"]
VOCAB = _ctx["vocab"]
WORD_TO_ID = _ctx["word_to_id"]
BLOCK_SIZE = _ctx["block_size"]
device = _ctx["device"]
padId = WORD_TO_ID.get("<pad>", 1)

def generate(promptTxt, maxTokens=25):
    qca_prompt, isLog = canonicalize_to_qca(promptTxt)
    toks = tokenize(qca_prompt, WORD_TO_ID)
    if not toks:
        return ""

    if len(toks) >= BLOCK_SIZE:
        toks = toks[: BLOCK_SIZE - 1]
    promptLen = len(toks)

    padded = toks + [padId] * (BLOCK_SIZE - promptLen)
    x = torch.tensor([padded], dtype=torch.long, device=device)

    out_words = []
    with torch.no_grad():
        for step in range(maxTokens):
            logits = model(x)
            pos = promptLen + step - 1
            if pos >= BLOCK_SIZE - 1:
                break

            lg = logits[0, pos].clone()
            lg[WORD_TO_ID["<unk>"]] = lg[WORD_TO_ID["<pad>"]] = -1e9

            if isLog:
                predId = lg.argmax().item()
            else:
                for pastId in set(toks + [WORD_TO_ID.get(w, 0) for w in out_words[-8:]]):
                    lg[pastId] -= 1.8
                probs = torch.softmax(lg / 0.7, dim=-1)
                predId = torch.multinomial(probs, 1).item()

            predWord = VOCAB[predId]
            if predWord in ("<eos>", "<pad>"):
                break
            out_words.append(predWord)
            x[0, pos + 1] = predId

    return " ".join(out_words)

if __name__ == "__main__":
    print("=" * 65)
    print("      WET FOOD LOCAL SIMULATOR (QUESTION-FIRST ENABLED)      ")
    print("=" * 65)
    print("Type a prompt or 'exit' to quit.\n")
    while True:
        try:
            line = input("> ")
            if not line.strip():
                continue
            if line.strip().lower() in ["exit", "quit"]:
                break
            print(f"  {generate(line)}\n")
        except (KeyboardInterrupt, EOFError):
            break