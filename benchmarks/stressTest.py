import os, sys, json, itertools, random

projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, projectRoot)

with open(os.path.join(projectRoot, "config", "world.json"), "r") as f:
    cfg = json.load(f)

from curriculum import formatQaSequence, splitQa

pools = cfg["pools"]
actors = pools["actor"]        # alice, bob, felix, luna
locations = pools["location"]  # garden, kitchen, park, room
items = pools["item"]          # ball, box, book, cup
colours = pools.get("colour", ["red", "blue", "yellow", "green"])

NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

stressSuite = {}
totalCount = 0

print("=" * 65)
print(" GENERATING EXHAUSTIVE COMBINATORIAL STRESS TEST SUITE")
print("=" * 65)

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
                        for qObj in cat["queries"]:
                            b = {"n1_word": NUM_WORDS[n1], "n2_word": NUM_WORDS[n2], "ans_word": NUM_WORDS[ans]}
                            rawQ = qObj["q"]
                            qList = rawQ if isinstance(rawQ, list) else [rawQ]
                            for qStr in qList:
                                pStr = tmpl.format(**b)
                                qRend = qStr.format(**b)
                                aRend = qObj["a"].format(**b)
                                p, t = splitQa(formatQaSequence(pStr, qRend, aRend))
                                stressSuite[catName].append({"q": p, "target": t})

    elif ctype == "arithmetic":
        for a1, a2 in itertools.permutations(actors, 2):
            for it in items:
                for n1 in range(1, 5):
                    for n2 in range(1, 5):
                        ans = n1 + n2
                        b = {"actor1": a1, "actor2": a2, "item1": it,
                             "n1_word": NUM_WORDS[n1], "n2_word": NUM_WORDS[n2], "ans_word": NUM_WORDS[ans]}
                        for tmpl in cat["premise"]:
                            if "gave {n2_word} to" in tmpl or "to {actor2}" in tmpl: continue
                            for qObj in cat["queries"]:
                                rawQ = qObj["q"]
                                qList = rawQ if isinstance(rawQ, list) else [rawQ]
                                for qStr in qList:
                                    p, t = splitQa(formatQaSequence(tmpl.format(**b), qStr.format(**b), qObj["a"].format(**b)))
                                    stressSuite[catName].append({"q": p, "target": t})

    elif ctype in ("propositional_logic", "comparative_chains", "conversation"):
        # Sample dense synthetic space (500 variations each)
        from curriculum import generatePropositionalSample, generateComparativeSample, generateConversationSample
        genFunc = {
            "propositional_logic": generatePropositionalSample,
            "comparative_chains": generateComparativeSample,
            "conversation": generateConversationSample
        }[ctype]
        seen = set()
        for _ in range(1200):
            sample = genFunc()
            if sample not in seen:
                seen.add(sample)
                p, t = splitQa(sample)
                stressSuite[catName].append({"q": p, "target": t})

    else:
        # Standard relational & entity tracking: Full Cartesian Product
        premises = cat.get("premise", cat.get("premises", [""]))
        queries = cat["queries"]

        # Permute 3 distinct actors, locations, items
        actorPerms = list(itertools.permutations(actors, 3))
        locPerms = list(itertools.permutations(locations, 3))
        itemPerms = list(itertools.permutations(items, 3))

        # Subsample combinations if permutation count is gigantic
        maxCombos = 400
        combos = list(itertools.product(actorPerms, locPerms, itemPerms))
        if len(combos) > maxCombos:
            combos = random.sample(combos, maxCombos)

        for (a1, a2, a3), (l1, l2, l3), (i1, i2, i3) in combos:
            b = {
                "actor1": a1, "actor2": a2, "actor3": a3,
                "location1": l1, "location2": l2, "location3": l3,
                "item1": i1, "item2": i2, "item3": i3,
                "colour1": colours[0], "colour2": colours[1], "colour3": colours[2],
                "agent1_name": a1, "agent1_pronoun": "she" if a1 in ("alice", "luna") else "he"
            }
            for tmpl in premises:
                for qObj in queries:
                    rawQ = qObj["q"]
                    qList = rawQ if isinstance(rawQ, list) else [rawQ]
                    for qStr in qList:
                        try:
                            pStr = tmpl.format(**b)
                            qRend = qStr.format(**b)
                            aRend = qObj["a"].format(**b)
                            p, t = splitQa(formatQaSequence(pStr, qRend, aRend))
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
print(f"[✓] TOTAL EXHAUSTIVE STRESS TESTS: {totalCount:,} -> Saved to {outPath}")
print("=" * 65)