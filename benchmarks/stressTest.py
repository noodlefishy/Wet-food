import os, sys, json, itertools, random, re

projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, projectRoot)

cfgPath = os.path.join(projectRoot, "config", "world.json")
if not os.path.exists(cfgPath):
    cfgPath = os.path.join(projectRoot, "config", "logic.json")
with open(cfgPath, "r") as f:
    cfg = json.load(f)

def normalizeText(text):
    t = re.sub(r'\s*([?.!:,])\s*', r' \1 ', text.strip())
    return re.sub(r'\s+', ' ', t).strip()

def formatQaSequence(premise, question, answer):
    p = normalizeText(premise.strip().rstrip(". "))
    q = normalizeText(question.strip().rstrip("?.: "))
    a = normalizeText(answer.strip().rstrip(". "))
    return f"[c] {p} . [q] {q} ? [a] {a} . <eos>"

def splitQa(sequenceStr):
    parts = sequenceStr.split(" [a] ")
    return parts[0].strip() + " [a]", parts[1].replace("<eos>", "").strip()

def pluralize(word, count=2):
    if count == 1: return word
    if word.endswith(("s", "x", "z", "ch", "sh")): return word + "es"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou": return word[:-1] + "ies"
    return word + "s"

pools     = cfg["pools"]
actors    = pools["actor"]
locations = pools["location"]
items     = pools["item"]
colours   = pools.get("colour", ["red", "blue", "yellow", "green", "white", "black"])

NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

stressSuite = {}
totalCount = 0

print("=" * 65)
print(" GENERATING EXHAUSTIVE COMBINATORIAL STRESS TEST SUITE")
print("=" * 65)

# Try importing generative logic samples from curriculum if present
try:
    from curriculum import generatePropositionalSample, generateComparativeSample, generateConversationSample
    hasGen = True
except ImportError:
    hasGen = False

for cat in cfg["categories"]:
    catName = cat["name"]
    ctype = cat.get("type", "")
    stressSuite[catName] = []

    # 1. Direct Arithmetic (e.g. "three plus four")
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
                            b = {
                                "n1_word": NUM_WORDS[n1], "n2_word": NUM_WORDS[n2],
                                "ans_word": NUM_WORDS[ans] if ans > 0 else "zero"
                            }
                            rawQ = qObj["q"]
                            qList = rawQ if isinstance(rawQ, list) else [rawQ]
                            for qStr in qList:
                                pStr = tmpl.format(**b)
                                qRend = qStr.format(**b)
                                aRend = qObj["a"].format(**b)
                                p, t = splitQa(formatQaSequence(pStr, qRend, aRend))
                                stressSuite[catName].append({"q": p, "target": t})

    # 2. In-Context Story Arithmetic (e.g. "alice had two balls...")
    elif ctype == "arithmetic" or "Arithmetic" in catName:
        # Sample combinations of actor pairs and items
        for a1, a2 in itertools.permutations(actors[:8], 2):
            for it in items[:6]:
                for isSub in (False, True):
                    if isSub:
                        pairs = [(n1, n2) for n1 in range(1, 6) for n2 in range(1, n1 + 1)]
                    else:
                        pairs = [(n1, n2) for n1 in range(1, 5) for n2 in range(1, 5)]

                    for n1, n2 in pairs:
                        ans = (n1 - n2) if isSub else (n1 + n2)
                        ans_word = NUM_WORDS[ans] if ans > 0 else ("none" if isSub else "zero")
                        
                        # Full dictionary with all legacy and plural keys
                        b = {
                            "actor1": a1, "actor2": a2,
                            "item1": it,
                            "item1_n1": pluralize(it, n1),
                            "item1_n2": pluralize(it, n2),
                            "item1_ans": pluralize(it, ans),
                            "item1_plural": pluralize(it, 2),
                            "n1_word": NUM_WORDS[n1], "n2_word": NUM_WORDS[n2],
                            "ans_word": ans_word
                        }
                        for tmpl in cat["premise"]:
                            is_sub_tmpl = "gave {n2_word}" in tmpl or "to {actor2}" in tmpl
                            if isSub != is_sub_tmpl: continue
                            for qObj in cat["queries"]:
                                rawQ = qObj["q"]
                                qList = rawQ if isinstance(rawQ, list) else [rawQ]
                                for qStr in qList:
                                    p, t = splitQa(formatQaSequence(tmpl.format(**b), qStr.format(**b), qObj["a"].format(**b)))
                                    stressSuite[catName].append({"q": p, "target": t})

    # 3. Dynamic Generated Logic Categories
    elif ctype in ("propositional_logic", "comparative_chains", "conversation") and hasGen:
        genFunc = {
            "propositional_logic": generatePropositionalSample, # pyright: ignore[reportPossiblyUnboundVariable]
            "comparative_chains": generateComparativeSample, # pyright: ignore[reportPossiblyUnboundVariable]
            "conversation": generateConversationSample # pyright: ignore[reportPossiblyUnboundVariable]
        }[ctype]
        seen = set()
        for _ in range(1000):
            sample = genFunc()
            if sample not in seen:
                seen.add(sample)
                p, t = splitQa(sample)
                stressSuite[catName].append({"q": p, "target": t})

    # 4. Standard Relational & Entity Tracking Tasks
    else:
        premises = cat.get("premise", cat.get("premises", [""]))
        queries  = cat["queries"]

        # Bound actor, location, and item sampling to prevent memory spikes
        maxCombos = 250
        seenCombos = set()
        combos = []
        for _ in range(maxCombos * 4):
            a_tuple = tuple(random.sample(actors, 3))
            l_tuple = tuple(random.sample(locations, 3))
            i_tuple = tuple(random.sample(items, 3))
            key = (a_tuple, l_tuple, i_tuple)
            if key not in seenCombos:
                seenCombos.add(key)
                combos.append((a_tuple, l_tuple, i_tuple))
            if len(combos) >= maxCombos:
                break

        for (a1, a2, a3), (l1, l2, l3), (i1, i2, i3) in combos:
            b = {
                "actor1": a1, "actor2": a2, "actor3": a3,
                "location1": l1, "location2": l2, "location3": l3,
                "item1": i1, "item2": i2, "item3": i3,
                "item1_plural": pluralize(i1, 2), "item2_plural": pluralize(i2, 2), "item3_plural": pluralize(i3, 2),
                "item1_n1": pluralize(i1, 1), "item1_n2": pluralize(i1, 2), "item1_ans": pluralize(i1, 2),
                "colour1": colours[0], "colour2": colours[1], "colour3": colours[2],
                "agent1_name": a1, "agent1_pronoun": "she" if a1 in ("alice", "luna", "sophie", "chloe", "daisy", "bella", "lucy", "lily", "mia", "ruby", "zoe") else "he"
            }
            for tmpl in premises:
                for qObj in queries:
                    rawQ = qObj["q"]
                    qList = rawQ if isinstance(rawQ, list) else [rawQ]
                    for qStr in qList:
                        try:
                            pStr  = tmpl.format(**b)
                            qRend = qStr.format(**b)
                            aRend = qObj["a"].format(**b)
                            p, t  = splitQa(formatQaSequence(pStr, qRend, aRend))
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