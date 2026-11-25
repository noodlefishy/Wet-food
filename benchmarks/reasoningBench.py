import os, re, sys, torch

projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, projectRoot)

from benchmarks.fastEvaluator import model, VOCAB, WORD_TO_ID, BLOCK_SIZE, padId, eosId, device

model.eval()
TOKEN_REGEX = r"\[[cqa]\]|<eos>|<unk>|<pad>|'s|\w+|[.,?!:]"

def tokenize(s):
    s = s.lower()
    for c in (".", "?", "!", ",", ":"):
        s = s.replace(c, f" {c} ")
    s = s.replace("'s", " 's ")
    return [WORD_TO_ID.get(w, 0) for w in re.findall(TOKEN_REGEX, s)]

def run_query(qca_prompt):
    toks = tokenize(qca_prompt)
    padded = toks + [padId] * (BLOCK_SIZE - len(toks))
    x = torch.tensor([padded[:BLOCK_SIZE]], dtype=torch.long, device=device)
    out = []
    with torch.no_grad():
        for step in range(20):
            pos = len(toks) + step - 1
            if pos >= BLOCK_SIZE - 1:
                break
            pred = model(x)[0, pos].argmax().item()
            if pred in (eosId, padId):
                break
            out.append(VOCAB[pred])
            x[0, pos + 1] = pred
    text = " ".join(out).strip().rstrip(". ")
    return text.split(":")[-1].strip().rstrip(". ") if ":" in text else text, " ".join(out)

TESTS = {
    "1. Distractor Resistance": [
        ("[q] where is alice ? [c] the cup is red . alice went to the kitchen . bob went to the park . [a]", "kitchen"),
        ("[q] who has the ball ? [c] luna went to the garden . bob found the ball in the room . the hat is blue . [a]", "bob"),
        ("[q] is felix in the yard ? [c] max went to the attic . felix is in the yard . the coin is yellow . [a]", "yes"),
    ],
    "2. Composed 2-Skill Hops": [
        ("[q] where is the ball now ? [c] alice gave the ball to bob . bob went to the kitchen . luna went to the park . [a]", "kitchen"),
        ("[q] where is the cup now ? [c] bob is in the garden or the room . bob is not in the garden . bob has the cup . [a]", "room"),
        ("[q] what is south of the garden ? [c] alice went to the garden . the garden is north of the porch . the porch is north of the park . [a]", "porch"),
    ],
    "3. Arithmetic & Counting": [
        ("[q] what is five plus three ? [c] five plus three . [a]", "eight"),
        ("[q] how many items does alice have ? [c] alice has one cup . bob gave alice two cups . [a]", "three"),
        ("[q] how many items does bob have ? [c] bob has the ball , the cup , and the book . [a]", "three"),
    ],
    "4. RAG Rule Deduction": [
        ("[q] is the cup hot ? [c] red things are hot . the cup is red . [a]", "yes"),
        ("[q] is the cup hot ? [c] red things are hot . the cup is blue . [a]", "no"),
        ("[q] is the ball wet ? [c] if the ball is in the garden then it is wet . alice put the ball in the garden . [a]", "yes"),
    ],
}

print("=" * 68)
print("          NEW REASONING & RAG-READINESS BENCHMARK")
print("=" * 68)

grand_pass, grand_tot = 0, 0
for group, cases in TESTS.items():
    passed = 0
    print(f"\n[{group}]")
    for prompt, target in cases:
        ans, raw = run_query(prompt)
        ok = ans == target
        passed += int(ok)
        mark = "✓" if ok else "✗"
        print(f"  [{mark}] Expected: {target:<8} | Got: {ans:<10} | Raw: {raw}")
    grand_pass += passed
    grand_tot += len(cases)
    print(f"  -> Subtotal: {passed}/{len(cases)}")

print("\n" + "=" * 68)
print(f" TOTAL: {grand_pass}/{grand_tot} ({grand_pass/grand_tot*100:.1f}%)")
print("=" * 68)