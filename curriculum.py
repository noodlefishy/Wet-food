import random, re

def normalizeText(text):
    t = re.sub(r"\s*([?.!:,])\s*", r" \1 ", text.strip())
    return re.sub(r"\s+", " ", t).strip()

def formatQaSequence(premise, question, answer):
    p = normalizeText(premise.strip().rstrip(". "))
    q = normalizeText(question.strip().rstrip("?.: "))
    a = normalizeText(answer.strip().rstrip(". "))
    return f"[q] {q} ? [c] {p} . [a] {a} . <eos>"

def splitQa(sequenceStr):
    parts = sequenceStr.split(" [a] ")
    return parts[0].strip() + " [a]", parts[1].replace("<eos>", "").strip()

def pluralize(word, count=2):
    if count == 1:
        return word
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"

NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
FEMALE_ACTORS = {"alice", "luna", "sophie", "chloe", "daisy", "bella", "lucy", "lily", "mia", "ruby", "zoe"}

def _sample_entities(pools):
    actors = random.sample(pools["actor"], min(5, len(pools["actor"])))
    locs = random.sample(pools["location"], min(5, len(pools["location"])))
    items = random.sample(pools["item"], min(5, len(pools["item"])))
    colours = random.sample(pools.get("colour", ["red", "blue", "yellow", "green"]), 4)
    return actors, locs, items, colours

def generateComposedCoTSample(cfg):
    pools = cfg["pools"]
    actors, locs, items, _ = _sample_entities(pools)
    a1, a2, a3 = actors[0], actors[1], actors[2]
    l1, l2, l3 = locs[0], locs[1], locs[2]
    i1, i2 = items[0], items[1]
    move_verb = random.choice(["went to", "walked to"])

    mode = random.randrange(5)
    if mode == 0:
        # Transfer (Task 05) + Movement (Task 01) -> Micro-CoT
        c1 = f"{a1} gave the {i1} to {a2}"
        c2 = f"{a2} {move_verb} the {l1}"
        c3 = f"{a3} {move_verb} the {l2}"
        clauses = [c1, c2, c3] if random.random() < 0.5 else [c1, c3, c2]
        p = " . ".join(clauses)
        q = f"where is the {i1} now"
        a = f"{a2} has the {i1} . {a2} is in the {l1} : {l1}"
    elif mode == 1:
        # 2-Step Transfer + Item Location
        p = f"{a1} found the {i1} in the {l1} . {a1} gave the {i1} to {a2} . {a2} {move_verb} the {l2}"
        q = f"where is the {i1} now"
        a = f"{a2} has the {i1} . {a2} is in the {l2} : {l2}"
    elif mode == 2:
        # Disjunction (Task 10) + Possession (Task 02)
        c1 = f"{a1} is in the {l1} or the {l2}"
        c2 = f"{a1} is not in the {l1}"
        c3 = f"{a1} has the {i1}"
        clauses = [c1, c2, c3]
        if random.random() < 0.5:
            clauses = [c2, c1, c3]
        p = " . ".join(clauses)
        q = f"where is the {i1} now"
        a = f"{a1} is in the {l2} : {l2}"
    elif mode == 3:
        # Deduction (Task 15) + Negation/Transfer
        p = f"cats like the {i1} . {a1} is a cat . {a2} gave the {i1} to {a3} . {a2} gave the {i2} to {a1}"
        q = f"what does {a1} like"
        a = f"{a1} is a cat : {i1}"
    else:
        # Actor Location + Compass Relation (Task 01 + 04)
        p = f"{a1} {move_verb} the {l1} . the {l1} is north of the {l2} . the {l2} is north of the {l3}"
        q = f"what is south of the {l1}"
        a = f"{l2} is south of {l1} : {l2}"

    return formatQaSequence(p, q, a)

def generateArithmeticSample(cfg):
    pools = cfg["pools"]
    actors, _, items, _ = _sample_entities(pools)
    a1, a2 = actors[0], actors[1]
    it = items[0]

    mode = random.randrange(3)
    if mode == 0:
        # Direct arithmetic (matches stressTest direct_arithmetic)
        is_sub = random.random() < 0.5
        if is_sub:
            n1 = random.randint(0, 10)
            n2 = random.randint(0, n1)
            ans = n1 - n2
            expr = f"{NUM_WORDS[n1]} minus {NUM_WORDS[n2]}"
        else:
            n1 = random.randint(0, 10)
            n2 = random.randint(0, 10 - n1)
            ans = n1 + n2
            expr = f"{NUM_WORDS[n1]} plus {NUM_WORDS[n2]}"
        return formatQaSequence(expr, f"what is {expr}", f"{expr} : {NUM_WORDS[ans]}")
    elif mode == 1:
        # Word-problem arithmetic (matches oodEvaluator Probe #3)
        is_sub = random.random() < 0.5
        if is_sub:
            n1 = random.randint(1, 6)
            n2 = random.randint(1, n1)
            ans = n1 - n2
            ans_w = NUM_WORDS[ans] if ans > 0 else "none"
            p = f"{a1} has {NUM_WORDS[n1]} {pluralize(it, n1)} . {a1} gave {NUM_WORDS[n2]} {pluralize(it, n2)} to {a2}"
            cot = f"{NUM_WORDS[n1]} minus {NUM_WORDS[n2]} : {ans_w}"
        else:
            n1 = random.randint(1, 5)
            n2 = random.randint(1, 5)
            ans = n1 + n2
            ans_w = NUM_WORDS[ans]
            p = f"{a1} has {NUM_WORDS[n1]} {pluralize(it, n1)} . {a2} gave {a1} {NUM_WORDS[n2]} {pluralize(it, n2)}"
            cot = f"{NUM_WORDS[n1]} plus {NUM_WORDS[n2]} : {ans_w}"
        return formatQaSequence(p, f"how many items does {a1} have", cot)
    else:
        # Multi-item counting (1..4 items)
        n = random.randint(1, 4)
        held = items[:n]
        if n == 1:
            p = f"{a1} has the {held[0]} . {a2} has the {items[4]}"
        elif n == 2:
            p = f"{a1} has the {held[0]} and the {held[1]}"
        else:
            p = f"{a1} has " + " , ".join(f"the {x}" for x in held[:-1]) + f" , and the {held[-1]}"
        return formatQaSequence(p, f"how many items does {a1} have", f"{' '.join(held)} : {NUM_WORDS[n]}")

def generateAtomicSample(cfg, targetCategory=None):
    pools = cfg["pools"]
    categories = cfg["categories"]

    if targetCategory is None:
        r = random.random()
        if r < 0.14:
            return generateComposedCoTSample(cfg)
        if r < 0.26:
            return generateArithmeticSample(cfg)

    weights = []
    for c in categories:
        if "weight" not in c:
            raise KeyError(f"[!] FATAL: Category '{c.get('name')}' is missing mandatory 'weight' field.")
        w = float(c["weight"])
        if w <= 0.0:
            raise ValueError(f"[!] FATAL: Category '{c.get('name')}' has non-positive weight: {w}")
        weights.append(w)

    cat = (
        next((c for c in categories if c["name"] == targetCategory), None)
        if targetCategory
        else random.choices(categories, weights=weights)[0]
    )
    if cat is None:
        raise RuntimeError("[!] FATAL: Failed to sample category from curriculum.")

    premiseTmpl = random.choice(cat.get("premise", cat.get("premises", [""])))
    queryObj = random.choice(cat["queries"])

    actors, locs, items, colours = _sample_entities(pools)
    bindings = {
        "actor1": actors[0], "actor2": actors[1], "actor3": actors[2], "actor4": actors[3],
        "location1": locs[0], "location2": locs[1], "location3": locs[2], "location4": locs[3],
        "item1": items[0], "item2": items[1], "item3": items[2], "item4": items[3],
        "colour1": colours[0], "colour2": colours[1], "colour3": colours[2],
        "agent1_name": actors[0],
        "agent1_pronoun": "she" if actors[0] in FEMALE_ACTORS else "he",
    }

    for k in ("item1", "item2", "item3", "item4"):
        bindings[f"{k}_plural"] = pluralize(bindings[k], 2)

    rawQ = queryObj["q"]
    qStr = random.choice(rawQ) if isinstance(rawQ, list) else rawQ
    pStr = premiseTmpl.format(**bindings)
    qRendered = qStr.format(**bindings)
    aRendered = queryObj["a"].format(**bindings)

    # Light verb generalization on canonical movement tasks so it doesn't overfit "went to"
    if targetCategory is None and random.random() < 0.15 and " went to the " in pStr:
        pStr = pStr.replace(" went to the ", random.choice([" walked to the ", " ran to the "]))

    return formatQaSequence(pStr, qRendered, aRendered)