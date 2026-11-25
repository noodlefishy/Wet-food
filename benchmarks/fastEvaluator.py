import json, os, sys, time
import torch

projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, projectRoot)

from pico.model import load_distributed_model
from pico.tokenizer import tokenize

def extract_answer(text):
    if not text or not text.strip():
        return ""
    clean = text.replace("<eos>", "").replace("<pad>", "").strip().rstrip(". ")
    if ":" in clean:
        val = clean.split(":")[-1].strip()
        return " ".join(val.rstrip(". ").split())
    return " ".join(clean.split()).rstrip(". ")

def main():
    ctx = load_distributed_model(projectRoot)
    model = ctx["model"]
    VOCAB = ctx["vocab"]
    WORD_TO_ID = ctx["word_to_id"]
    BLOCK_SIZE = ctx["block_size"]
    NUM_LAYERS = ctx["num_layers"]
    device = ctx["device"]

    padId = WORD_TO_ID.get("<pad>", 1)
    eosId = WORD_TO_ID.get("<eos>", 2)

    print(f"[✓] Model loaded ({NUM_LAYERS}L RoPE) on {device.upper()}\n")

    suitePath = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else os.path.join(projectRoot, "config", "stressTest.json")
    if not os.path.exists(suitePath):
        suitePath = os.path.join(projectRoot, "config", "questions.json")

    print(f"[*] Loading evaluation suite: {suitePath}")
    with open(suitePath) as f:
        rawData = json.load(f)

    allTests = []
    for cat, items in rawData.items():
        for item in items:
            if item.get("target") is not None:
                allTests.append((cat, item["q"], str(item["target"])))

    total = len(allTests)
    print(f"[*] Evaluating {total:,} QA tests across {len(rawData)} categories...\n")

    catStats, totalPassed, failures = {}, 0, []
    t0 = time.time()
    BATCH = 512

    with torch.no_grad():
        for i in range(0, total, BATCH):
            batch = allTests[i : i + BATCH]
            prompts = [tokenize(q, WORD_TO_ID) for _, q, _ in batch]
            lens = [len(p) for p in prompts]

            padded = [p + [padId] * (BLOCK_SIZE - len(p)) for p in prompts]
            x = torch.tensor(padded, dtype=torch.long, device=device)

            genTokens = [[] for _ in batch]
            active = [True] * len(batch)

            for step in range(20):
                logits = model(x)
                for j in range(len(batch)):
                    if not active[j]:
                        continue
                    pos = lens[j] + step - 1
                    if pos >= BLOCK_SIZE - 1:
                        active[j] = False
                        continue
                    predId = logits[j, pos].argmax().item()
                    genTokens[j].append(VOCAB[predId])
                    if predId in (eosId, padId):
                        active[j] = False
                    if pos + 1 < BLOCK_SIZE:
                        x[j, pos + 1] = predId
                if not any(active):
                    break

            for j, (cat, q, target) in enumerate(batch):
                genStr = " ".join(genTokens[j])
                predAns = extract_answer(genStr)
                expectedAns = extract_answer(target)

                if " and " in expectedAns and " and " in predAns:
                    if set(predAns.split(" and ")) == set(expectedAns.split(" and ")):
                        predAns = expectedAns

                ok = (predAns == expectedAns and predAns != "")
                if cat not in catStats:
                    catStats[cat] = [0, 0]
                catStats[cat][1] += 1
                if ok:
                    catStats[cat][0] += 1
                    totalPassed += 1
                elif len(failures) < 50:
                    failures.append((cat, q, target, genStr))

            done = min(i + BATCH, total)
            pct = (totalPassed / done * 100) if done else 0
            elapsed = time.time() - t0
            eta = (total - done) * (elapsed / done) if done else 0
            sys.stdout.write(
                f"\r  [Evaluator] Progress: {done:>7,}/{total:,} [{done/total*100:5.1f}%] | Acc: {pct:5.1f}% | Elapsed: {elapsed:4.1f}s | ETA: {eta:4.1f}s"
            )
            sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"\n\n{'='*65}")
    print(f" FINAL REASONING ACCURACY: {totalPassed:,}/{total:,} = {totalPassed/total*100:.1f}%  ({elapsed:.2f}s)")
    print(f"{'='*65}")

    for cat in sorted(catStats.keys()):
        passed, n = catStats[cat]
        pct = (passed / n * 100) if n else 0
        mark = "✓" if pct >= 90 else ("~" if pct >= 70 else "✗")
        print(f"  [{mark}] {cat:<38} {passed:>6}/{n:<6} {pct:5.1f}%")

    ff = [{"category": cat, "question": q, "target": tgt, "got": pred} for cat, q, tgt, pred in failures]
    for out_path in ["failures.json", "/content/failures.json"]:
        try:
            with open(out_path, "w") as fOut:
                json.dump(ff, fOut, indent=2)
            print(f"[✓] Saved {len(ff)} failures to {out_path}")
        except Exception:
            pass

    if failures:
        print(f"\n{'='*65}")
        print(" SAMPLE MISPREDICTIONS (First 15)")
        print(f"{'='*65}")
        for cat, q, tgt, pred in failures[:15]:
            print(f"  [{cat}]\n    Q: {q}\n    Expected: {tgt!r}\n    Generated: {pred!r}\n")

if __name__ == "__main__":
    main()