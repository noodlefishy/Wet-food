import os, sys, time
import torch

projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, projectRoot)

from pico.model import load_distributed_model
from pico.tokenizer import tokenize

ctx = load_distributed_model(projectRoot)
model = ctx["model"]
VOCAB = ctx["vocab"]
WORD_TO_ID = ctx["word_to_id"]
BLOCK_SIZE = ctx["block_size"]
device = ctx["device"]

aId = WORD_TO_ID.get("[a]", 9)
padId = WORD_TO_ID.get("<pad>", 1)

def evalSuite(tests, desc):
    total = len(tests)
    passed = 0
    t0 = time.time()
    for _, q, target in tests:
        toks = tokenize(q, WORD_TO_ID)
        try:
            aPos = next(j for j in range(len(toks) - 1, -1, -1) if toks[j] == aId)
        except StopIteration:
            toks.append(aId)
            aPos = len(toks) - 1

        padded = toks[: aPos + 1] + [padId] * (BLOCK_SIZE - len(toks[: aPos + 1]))
        x = torch.tensor([padded[:BLOCK_SIZE]], dtype=torch.long, device=device)

        gen = []
        with torch.no_grad():
            curr_pos = aPos
            for _ in range(6):
                logits = model(x)
                predId = logits[0, min(curr_pos, BLOCK_SIZE - 1)].argmax().item()
                w = VOCAB[predId] if predId < len(VOCAB) else "<unk>"
                if w in ("<eos>", "<pad>"):
                    break
                gen.append(w)
                curr_pos += 1
                if curr_pos < BLOCK_SIZE:
                    x[0, curr_pos] = predId

        outStr = " ".join(gen)
        predClean = outStr.split(":")[-1].strip().rstrip(". ") if ":" in outStr else outStr.strip().rstrip(". ")
        tgtClean = target.split(":")[-1].strip().rstrip(". ") if ":" in target else target.strip().rstrip(". ")

        if predClean.split() and tgtClean.split() and predClean.split()[0] == tgtClean.split()[0]:
            passed += 1

    pct = (passed / total * 100) if total else 0
    print(f"  {desc:<45}: {passed:>4}/{total:<4} ({pct:5.1f}%) in {time.time()-t0:.2f}s")
    return passed, total

print("=" * 65)
print("     OUT-OF-DISTRIBUTION (OOD) GENERALIZATION BENCHMARK")
print("=" * 65)

PROBE_ENTITIES = [
    ("Unseen Entity (Milo/Barn)", "[c] milo went to the barn . [q] where is milo ? [a]", "barn"),
    ("Unseen Entity (Cleo/Attic)", "[c] cleo went to the attic . [q] where is cleo ? [a]", "attic"),
    ("Unseen Entity (Jasper/Cellar)", "[c] jasper went to the cellar . [q] where is jasper ? [a]", "cellar"),
    ("Unseen Entity (Ziggy/Roof)", "[c] ziggy went to the roof . [q] where is ziggy ? [a]", "roof"),
]
evalSuite(PROBE_ENTITIES, "1. Novel Entity Grounding")

PROBE_SYNTAX = [
    ("Passive Voice", "[c] the red ball was taken by alice . [q] who has the red ball ? [a]", "alice"),
    ("Inverted Clause", "[c] to the kitchen went bob . [q] where is bob ? [a]", "kitchen"),
    ("Compound Subject", "[c] felix and luna were in the forest . [q] where is felix ? [a]", "forest"),
]
evalSuite(PROBE_SYNTAX, "2. Novel Syntax Generalization")

PROBE_MATH = [
    ("3-Item Addition", "[c] alice has one cup . bob gave alice two cups . [q] how many items does alice have ? [a]", "three"),
    ("Zero Remainder", "[c] bob has two balls . bob gave two balls to felix . [q] how many items does bob have ? [a]", "none"),
    ("Single Transfer", "[c] felix has three books . felix gave one book to luna . [q] how many items does felix have ? [a]", "two"),
]
evalSuite(PROBE_MATH, "3. Multi-hop Arithmetic Probe")
print("=" * 65)