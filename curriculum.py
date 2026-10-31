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

    # 1. Strict 50/50 balance for boolean / polar queries
    hasYes = any("yes" in str(q.get("a", "")) for q in queries)
    hasNo  = any("no" in str(q.get("a", "")) for q in queries)
    if hasYes and hasNo:
        wantYes = random.random() < 0.50
        filtered = [q for q in queries if ("yes" in str(q.get("a", ""))) == wantYes]
        queryObj = random.choice(filtered) if filtered else random.choice(queries)

    # 2. Adversarial Anti-Recency Engine: Penalizes lazy token-distance shortcuts
    elif "{actor1}" in premiseTmpl and "{actor2}" in premiseTmpl:
        dist_actor = "actor1" if premiseTmpl.find("{actor1}") < premiseTmpl.find("{actor2}") else "actor2"
        rec_actor = "actor2" if dist_actor == "actor1" else "actor1"
        
        dist_queries = [q for q in queries if f"{{{dist_actor}}}" in str(q)]
        rec_queries  = [q for q in queries if f"{{{rec_actor}}}" in str(q)]
        other_queries = [q for q in queries if q not in dist_queries and q not in rec_queries]
        
        r = random.random()
        if r < 0.60 and dist_queries:      # 60% Distant Penalty forces true pointer routing
            queryObj = random.choice(dist_queries)
        elif r < 0.85 and rec_queries:     # 25% Recent
            queryObj = random.choice(rec_queries)
        elif other_queries:                # 15% Other
            queryObj = random.choice(other_queries)
        else:
            queryObj = random.choice(queries)
    else:
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