import itertools, json, os, random, sys

projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, projectRoot)

from curriculum import (
    formatQaSequence,
    generateComposedCoTSample,
    generateDisjunctionSample,
    generateMultiCountSample,
    generateRagRuleSample,
    pluralize,
    splitQa,
)

random.seed(42)

cfgPath = os.path.join(projectRoot, "config", "world.json")
with open(cfgPath, "r") as f:
    cfg = json.load(f)

pools = cfg["pools"]
actors = pools["actor"]
locations = pools["location"]
items = pools["item"]
colours = pools.get("colour", ["red", "blue", "yellow", "green"])
NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
FEMALE_ACTORS = {"alice", "luna", "sophie", "chloe", "daisy", "bella", "lucy", "lily", "mia", "ruby", "zoe"}

stressSuite = {}
totalCount = 0

print("=" * 65)
print(" GENERATING EXHAUSTIVE CHUNKED MICRO-COT STRESS TEST SUITE")
print("=" * 65)

for cat in cfg["categories"]:
    catName = cat["name"]
    ctype = cat.get("type", "")
    stressSuite[catName] = []

    if ctype == "direct_arithmetic":
        for n1 in range(11):
            for n2 in range(11):
                for isSub in (False, True):
                    if isSub and n2 > n1:
                        continue
                    ans = (n1 - n2) if isSub else (n1 + n2)
                    if ans > 10:
                        continue
                    for tmpl in cat["premise"]:
                        if isSub != ("minus" in tmpl):
                            continue
                        for qObj in cat["queries"]:
                            if isSub != ("minus" in qObj["q"]):
                                continue
                            b = {
                                "n1_word": NUM_WORDS[n1],
                                "n2_word": NUM_WORDS[n2],
                                "ans_word": NUM_WORDS[ans],
                            }
                            p, t = splitQa(formatQaSequence(tmpl.format(**b), qObj["q"].format(**b), qObj["a"].format(**b)))
                            stressSuite[catName].append({"q": p, "target": t})

    elif ctype == "arithmetic" or "Arithmetic" in catName:
        cases = []
        for a1, a2 in itertools.permutations(actors[:8], 2):
            for it in items[:6]:
                for isSub in (False, True):
                    pairs = (
                        [(n1, n2) for n1 in range(1, 7) for n2 in range(1, n1 + 1)]
                        if isSub
                        else [(n1, n2) for n1 in range(1, 6) for n2 in range(1, 6)]
                    )
                    for n1, n2 in pairs:
                        ans = (n1 - n2) if isSub else (n1 + n2)
                        op_word = "minus" if isSub else "plus"
                        b = {
                            "actor1": a1,
                            "actor2": a2,
                            "item1": it,
                            "item1_n1": pluralize(it, n1),
                            "item1_n2": pluralize(it, n2),
                            "item1_ans": pluralize(it, ans),
                            "item1_plural": pluralize(it, 2),
                            "n1_word": NUM_WORDS[n1],
                            "n2_word": NUM_WORDS[n2],
                            "expr": f"{NUM_WORDS[n1]} {op_word} {NUM_WORDS[n2]}",
                            "ans_word": NUM_WORDS[ans] if ans > 0 else "none",
                        }
                        for tmpl in cat["premise"]:
                            if isSub != ("gave" in tmpl and "to {actor2}" in tmpl):
                                continue
                            for qObj in cat["queries"]:
                                p, t = splitQa(formatQaSequence(tmpl.format(**b), qObj["q"].format(**b), qObj["a"].format(**b)))
                                cases.append({"q": p, "target": t})
        if len(cases) > 1500:
            cases = random.sample(cases, 1500)
        stressSuite[catName].extend(cases)

    elif ctype == "disjunction":
        seen = set()
        for _ in range(1500):
            sample = generateDisjunctionSample(cfg)
            if sample not in seen:
                seen.add(sample)
                p, t = splitQa(sample)
                stressSuite[catName].append({"q": p, "target": t})
            if len(stressSuite[catName]) >= 1000:
                break

    elif ctype in ("multi_counting", "composed_cot", "rag_rules"):
        gen_fn = {
            "multi_counting": lambda: generateMultiCountSample(cfg),
            "composed_cot": lambda: generateComposedCoTSample(cfg, allow_distractor=True),
            "rag_rules": lambda: generateRagRuleSample(cfg, allow_distractor=True),
        }[ctype]
        target_n = 500 if ctype == "multi_counting" else 1000
        seen = set()
        for _ in range(target_n * 3):
            sample = gen_fn()
            if sample not in seen:
                seen.add(sample)
                p, t = splitQa(sample)
                stressSuite[catName].append({"q": p, "target": t})
            if len(stressSuite[catName]) >= target_n:
                break

    else:
        premises = cat.get("premise", cat.get("premises", [""]))
        queries = cat["queries"]
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
                combos.append(key)
            if len(combos) >= maxCombos:
                break

        for (a1, a2, a3), (l1, l2, l3), (i1, i2, i3) in combos:
            b = {
                "actor1": a1, "actor2": a2, "actor3": a3,
                "location1": l1, "location2": l2, "location3": l3,
                "item1": i1, "item2": i2, "item3": i3,
                "item1_plural": pluralize(i1, 2),
                "item2_plural": pluralize(i2, 2),
                "agent1_name": a1,
                "agent1_pronoun": "she" if a1 in FEMALE_ACTORS else "he",
                "colour1": colours[0],
                "colour2": colours[1],
            }
            for tmpl in premises:
                for qObj in queries:
                    try:
                        p, t = splitQa(formatQaSequence(tmpl.format(**b), qObj["q"].format(**b), qObj["a"].format(**b)))
                        stressSuite[catName].append({"q": p, "target": t})
                    except KeyError:
                        continue

    count = len(stressSuite[catName])
    totalCount += count
    print(f"  {catName:<40}: {count:>6,} test cases")

outPath = os.path.join(projectRoot, "config", "stressTest.json")
with open(outPath, "w") as f:
    json.dump(stressSuite, f)
print("=" * 65)
print(f"[✓] TOTAL TESTS GENERATED: {totalCount:,} -> Saved to {outPath}")
print("=" * 65)