import json, random, re

def normalizeText(text):
    t = re.sub(r'\s*([?.!:,])\s*', r' \1 ', text.strip())
    return re.sub(r'\s+', ' ', t).strip()

def formatQaSequence(premise, question, answer):
    p = normalizeText(premise.strip().rstrip(". "))
    q = normalizeText(question.strip().rstrip("?.: "))
    a = normalizeText(answer.strip().rstrip(". "))
    # QUESTION-FIRST FORMAT: The mathematically correct way for causal attention
    return f"[q] {q} ? [c] {p} . [a] {a} . <eos>"

def splitQa(sequenceStr):
    parts = sequenceStr.split(" [a] ")
    return parts[0].strip() + " [a]", parts[1].replace("<eos>", "").strip()

def pluralize(word, count=2):
    if count == 1: return word
    if word.endswith(("s", "x", "z", "ch", "sh")): return word + "es"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou": return word[:-1] + "ies"
    return word + "s"

NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
FEMALE_ACTORS = {"alice", "luna", "sophie", "chloe", "daisy", "bella", "lucy", "lily", "mia", "ruby", "zoe"}

def generateAtomicSample(cfg, targetCategory=None):
    pools      = cfg["pools"]
    categories = cfg["categories"]

    weights = []
    for c in categories:
        if "weight" not in c:
            raise KeyError(f"[!] FATAL: Category '{c.get('name')}' is missing mandatory 'weight' field.")
        w = float(c["weight"])
        if w <= 0.0:
            raise ValueError(f"[!] FATAL: Category '{c.get('name')}' has non-positive weight: {w}")
        weights.append(w)

    cat = (next((c for c in categories if c["name"] == targetCategory), None)
           if targetCategory else random.choices(categories, weights=weights)[0])
    if cat is None:
        raise RuntimeError("[!] FATAL: Failed to sample category from curriculum.")

    ctype = cat.get("type", "")
    premiseTmpl = random.choice(cat.get("premise", cat.get("premises", [""])))
    queries     = cat["queries"]
    
    queryObj = random.choice(queries)

    actors  = random.sample(pools["actor"], min(5, len(pools["actor"])))
    locs    = random.sample(pools["location"], min(5, len(pools["location"])))
    items   = random.sample(pools["item"], min(5, len(pools["item"])))
    colours = random.sample(pools.get("colour", ["red", "blue", "yellow", "green"]), 4)

    bindings = {
        "actor1": actors[0], "actor2": actors[1], "actor3": actors[2], "actor4": actors[3],
        "location1": locs[0], "location2": locs[1], "location3": locs[2], "location4": locs[3],
        "item1": items[0],   "item2": items[1],   "item3": items[2],   "item4": items[3],
        "colour1": colours[0], "colour2": colours[1], "colour3": colours[2],
        "agent1_name": actors[0],
        "agent1_pronoun": "she" if actors[0] in FEMALE_ACTORS else "he"
    }

    for k in ("item1", "item2", "item3", "item4"):
        bindings[f"{k}_plural"] = pluralize(bindings[k], 2)

    rawQ      = queryObj["q"]
    qStr      = random.choice(rawQ) if isinstance(rawQ, list) else rawQ
    pStr      = premiseTmpl.format(**bindings)
    qRendered = qStr.format(**bindings)
    aRendered = queryObj["a"].format(**bindings)

    return formatQaSequence(pStr, qRendered, aRendered)