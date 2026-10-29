import json, random, re

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

VERB_JITTER = {
    "went to the": ["went to the", "walked to the", "ran to the", "arrived at the"],
    "is in the": ["is in the", "was in the", "went to the"],
    "found the": ["found the", "got the", "picked up the"],
    "gave the": ["gave the", "handed the"],
    "has the": ["has the", "had the", "holds the"]
}

DISTRACTOR_TEMPLATES = [
    "{actor} saw a little bird",
    "{actor} sat in the sun",
    "{actor} wanted to play",
    "{actor} was very happy",
    "{actor} looked around",
    "{actor} found a little flower"
]

def applyVerbJitter(text):
    for base, variants in VERB_JITTER.items():
        if base in text and random.random() < 0.4:
            text = text.replace(base, random.choice(variants), 1)
    return text

def generateAtomicSample(cfg, targetCategory=None, injectDistractor=True):
    pools = cfg["pools"]
    pairPools = cfg.get("pair_pools", {})
    categories = cfg["categories"]
    weights = [c.get("weight", 2.0) for c in categories]
    cat = next((c for c in categories if c["name"] == targetCategory), None) if targetCategory else random.choices(categories, weights=weights)[0]
    if cat is None: cat = random.choice(categories)
    premiseTmpl = random.choice(cat["premise"] if "premise" in cat else cat["premises"])

    queries = cat["queries"]
    hasYes, hasNo = any(q["a"] == "yes" for q in queries), any(q["a"] == "no" for q in queries)
    if hasYes and hasNo:
        wantYes = random.random() < 0.5
        filtered = [q for q in queries if (q["a"] == "yes") == wantYes]
        queryObj = random.choice(filtered) if filtered else random.choice(queries)
    else:
        queryObj = random.choice(queries)

    actors = random.sample(pools["actor"], len(pools["actor"]))
    locs = random.sample(pools["location"], len(pools["location"]))
    items = random.sample(pools["item"], len(pools["item"]))
    colours = random.sample(pools.get("colour", ["red", "blue", "yellow", "green"]), 4)
    agentPool = pairPools.get("agent", [])
    agents = random.sample(agentPool, min(2, len(agentPool))) if agentPool else []

    numWords = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    bindings = {
        "actor1": actors[0], "actor2": actors[1], "actor3": actors[2],
        "location1": locs[0], "location2": locs[1], "location3": locs[2],
        "item1": items[0], "item2": items[1], "item3": items[2],
        "colour1": colours[0], "colour2": colours[1], "colour3": colours[2]
    }
    if agents:
        bindings["agent1_name"] = agents[0]["name"]
        bindings["agent1_pronoun"] = agents[0]["pronoun"]

    if cat.get("type") == "arithmetic" or "Arithmetic" in cat.get("name", ""):
        isSub = "to {actor2}" in premiseTmpl or "gave {n2_word} to" in premiseTmpl
        if isSub:
            n1 = random.randint(1, 5); n2 = random.randint(1, n1); ans = n1 - n2
        else:
            n1 = random.randint(1, 4); n2 = random.randint(1, 4); ans = n1 + n2
        bindings["n1_word"] = numWords[n1]
        bindings["n2_word"] = numWords[n2]
        bindings["ans_word"] = numWords[ans] if ans > 0 else "none"

    rawQ = queryObj["q"]
    qStr = random.choice(rawQ) if isinstance(rawQ, list) else rawQ
    pStr = applyVerbJitter(premiseTmpl.format(**bindings))
    qRendered = qStr.format(**bindings)
    aRendered = queryObj["a"].format(**bindings)

    if injectDistractor and random.random() < 0.35:
        unusedActors = [a for a in pools["actor"] if a not in qRendered and a not in aRendered]
        if unusedActors:
            distActor = random.choice(unusedActors)
            distPhrase = random.choice(DISTRACTOR_TEMPLATES).format(actor=distActor)
            pos = random.choice(["prefix", "suffix", "middle"])
            if pos == "prefix":
                pStr = f"{distPhrase} . {pStr}"
            elif pos == "suffix":
                pStr = f"{pStr} . {distPhrase}"
            else:
                parts = pStr.split(" . ")
                if len(parts) > 1:
                    parts.insert(random.randint(1, len(parts) - 1), distPhrase)
                    pStr = " . ".join(parts)
                else:
                    pStr = f"{pStr} . {distPhrase}"

    return formatQaSequence(pStr, qRendered, aRendered)

# Backwards compatibility aliases
normalize_text = normalizeText
format_qa_sequence = formatQaSequence
split_qa = splitQa
generate_atomic_sample = generateAtomicSample
