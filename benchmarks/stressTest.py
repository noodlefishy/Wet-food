import os, sys, json, itertools, random, re

projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, projectRoot)

from chunks import preprocess_chunks
from curriculum import splitQa, formatQaSequence, pluralize

cfgPath = os.path.join(projectRoot, "config", "world.json")
with open(cfgPath, "r") as f: cfg = json.load(f)

pools     = cfg["pools"]
actors    = pools["actor"]
locations = pools["location"]
items     = pools["item"]
colours   = pools.get("colour", ["red", "blue", "yellow", "green"])
NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
FEMALE_ACTORS = {"alice", "luna", "sophie", "chloe", "daisy", "bella", "lucy", "lily", "mia", "ruby", "zoe"}

stressSuite = {}
totalCount = 0

print("=" * 65)
print(" GENERATING EXHAUSTIVE CHUNKED MICRO-COT STRESS TEST SUITE")
print("=" * 65)

try:
    from curriculum import generatePropositionalSample, generateComparativeSample, generateConversationSample
    hasGen = True
except ImportError:
    hasGen = False

for cat in cfg["categories"]:
    catName = cat["name"]
    ctype = cat.get("type", "")
    stressSuite[catName] = []

    if ctype == "direct_arithmetic":
        for n1 in range(10):
            for n2 in range(10):
                for op, isSub in [("plus", False), ("minus", True)]:
                    if isSub and n2 > n1: continue
                    ans = (n1 - n2) if isSub else (n1 + n2)
                    if ans > 10: continue
                    for tmpl in cat["premise"]:
                        if isSub != ("minus" in tmpl): continue
                        for qObj in cat["queries"]:
                            b = {"n1_word": NUM_WORDS[n1], "n2_word": NUM_WORDS[n2], "ans_word": NUM_WORDS[ans] if ans > 0 else "zero"}
                            p, t = splitQa(formatQaSequence(tmpl.format(**b), qObj["q"].format(**b), qObj["a"].format(**b)))
                            stressSuite[catName].append({"q": p, "target": t})

    elif ctype == "arithmetic" or "Arithmetic" in catName:
        for a1, a2 in itertools.permutations(actors[:8], 2):
            for it in items[:6]:
                for isSub in (False, True):
                    pairs = [(n1, n2) for n1 in range(1, 6) for n2 in range(1, n1 + 1)] if isSub else [(n1, n2) for n1 in range(1, 5) for n2 in range(1, 5)]
                    for n1, n2 in pairs:
                        ans = (n1 - n2) if isSub else (n1 + n2)
                        b = {
                            "actor1": a1, "actor2": a2, "item1": it,
                            "item1_n1": pluralize(it, n1), "item1_n2": pluralize(it, n2),
                            "item1_ans": pluralize(it, ans), "item1_plural": pluralize(it, 2),
                            "n1_word": NUM_WORDS[n1], "n2_word": NUM_WORDS[n2],
                            "ans_word": NUM_WORDS[ans] if ans > 0 else "none"
                        }
                        for tmpl in cat["premise"]:
                            if isSub != ("gave" in tmpl and "to {actor2}" in tmpl): continue
                            for qObj in cat["queries"]:
                                p, t = splitQa(formatQaSequence(tmpl.format(**b), qObj["q"].format(**b), qObj["a"].format(**b)))
                                stressSuite[catName].append({"q": p, "target": t})

    elif ctype in ("propositional_logic", "comparative_chains", "conversation") and hasGen:
        genFunc = {
            "propositional_logic": generatePropositionalSample,
            "comparative_chains": generateComparativeSample,
            "conversation": generateConversationSample
        }[ctype]
        seen = set()
        for _ in range(1000):
            sample = genFunc()
            if sample not in seen:
                seen.add(sample)
                p, t = splitQa(sample)
                stressSuite[catName].append({"q": p, "target": t})

    else:
        premises = cat.get("premise", cat.get("premises", [""]))
        queries  = cat["queries"]
        maxCombos = 250
        combos = []
        seen = set()
        for _ in range(maxCombos * 4):
            a_t = tuple(random.sample(actors, 3))
            l_t = tuple(random.sample(locations, 3))
            i_t = tuple(random.sample(items, 3))
            key = (a_t, l_t, i_t)
            if key not in seen:
                seen.add(key)
                combos.append((a_t, l_t, i_t))
            if len(combos) >= maxCombos: break

        for (a1, a2, a3), (l1, l2, l3), (i1, i2, i3) in combos:
            b = {
                "actor1": a1, "actor2": a2, "actor3": a3,
                "location1": l1, "location2": l2, "location3": l3,
                "item1": i1, "item2": i2, "item3": i3,
                "item1_plural": pluralize(i1, 2), "item2_plural": pluralize(i2, 2),
                "agent1_name": a1, "agent1_pronoun": "she" if a1 in FEMALE_ACTORS else "he",
                "colour1": colours[0], "colour2": colours[1]
            }
            for tmpl in premises:
                for qObj in queries:
                    try:
                        p, t = splitQa(formatQaSequence(tmpl.format(**b), qObj["q"].format(**b), qObj["a"].format(**b)))
                        stressSuite[catName].append({"q": p, "target": t})
                    except KeyError: continue

    count = len(stressSuite[catName])
    totalCount += count
    print(f"  {catName:<40}: {count:>6,} test cases")

outPath = os.path.join(projectRoot, "config", "stressTest.json")
with open(outPath, "w") as f: json.dump(stressSuite, f)
print("=" * 65)
print(f"[✓] TOTAL TESTS GENERATED: {totalCount:,} -> Saved to {outPath}")
print("=" * 65)