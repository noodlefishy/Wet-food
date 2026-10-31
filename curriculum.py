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

# ── Propositional Logic ────────────────────────────────────────────────────────
PROP_RULES = [
    {"cond": "the door is open",   "concl": "the dog is outside",   "target": "is the dog outside"},
    {"cond": "it is raining",      "concl": "the ground is wet",     "target": "is the ground wet"},
    {"cond": "the sun is warm",    "concl": "the park is crowded",   "target": "is the park crowded"},
    {"cond": "the box is heavy",   "concl": "it is hard to carry",   "target": "is it hard to carry"},
    {"cond": "alice is hungry",    "concl": "alice wants to eat",    "target": "does alice want to eat"},
    {"cond": "the window is open", "concl": "the room is cold",      "target": "is the room cold"},
    {"cond": "bob is tired",       "concl": "bob wants to sleep",    "target": "does bob want to sleep"},
    {"cond": "the cat is inside",  "concl": "the cat is warm",       "target": "is the cat warm"},
    {"cond": "luna is happy",      "concl": "luna is smiling",       "target": "is luna smiling"},
    {"cond": "the jar is full",    "concl": "the jar is heavy",      "target": "is the jar heavy"}
]

def generatePropositionalSample():
    rule = random.choice(PROP_RULES)
    modePonens = random.random() < 0.5
    if modePonens:
        premise = f"if {rule['cond']} then {rule['concl']} . {rule['cond']}"
        answer  = "yes"
    else:
        premise = f"if {rule['cond']} then {rule['concl']} . {rule['concl']} is not true"
        answer  = "no"
    return formatQaSequence(premise, rule["target"], answer)

# ── Comparative Chains ─────────────────────────────────────────────────────────
COMP_PAIRS    = [("bigger", "biggest"), ("smaller", "smallest"),
                 ("faster", "fastest"), ("slower",  "slowest")]
COMP_ENTITIES = ["the elephant", "the lion", "the dog", "the rabbit", "the turtle",
                 "the cat", "the bird", "alice", "bob", "felix", "luna"]

def generateComparativeSample():
    comp, sup = random.choice(COMP_PAIRS)
    e1, e2, e3 = random.sample(COMP_ENTITIES, 3)
    opp_sup = {"biggest": "smallest", "smallest": "biggest",
               "fastest": "slowest",  "slowest":  "fastest"}[sup]
    askTop = random.random() < 0.5
    premise = f"{e1} is {comp} than {e2} . {e2} is {comp} than {e3}"
    if askTop:
        question = f"who is the {sup}"
        answer   = e1
    else:
        question = f"who is the {opp_sup}"
        answer   = e3
    return formatQaSequence(premise, question, answer)

# ── General Conversation ───────────────────────────────────────────────────────
_CONV = {
    "greet": {
        "u": ["hello", "hi", "hey", "good morning", "good day"],
        "r": ["hello friend"]
    },
    "howru": {
        "u": ["how are you", "how are you doing", "how is your day", "are you well"],
        "r": ["i am doing great"]
    },
    "identity": {
        "u": ["who are you", "what are you", "are you an ai", "what can you do"],
        "r": ["i am an ai"]
    },
    "thanks": {
        "u": ["thank you", "thanks", "thank you very much"],
        "r": ["you are welcome"]
    },
    "bye": {
        "u": ["goodbye", "bye", "see you later", "have a good day"],
        "r": ["goodbye friend"]
    }
}

def generateConversationSample():
    intent = random.choice(list(_CONV.keys()))
    utterance = random.choice(_CONV[intent]["u"])
    reply     = random.choice(_CONV[intent]["r"])
    return formatQaSequence(f"user : {utterance}", "how do you respond", f"reply : {reply}")

NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
FEMALE_ACTORS = {"alice", "luna", "sophie", "chloe", "daisy", "bella", "lucy", "lily", "mia", "ruby", "zoe"}

def generateAtomicSample(cfg, targetCategory=None):
    pools      = cfg["pools"]
    categories = cfg["categories"]
    weights    = [c.get("weight", 2.0) for c in categories]

    cat = (next((c for c in categories if c["name"] == targetCategory), None)
           if targetCategory else random.choices(categories, weights=weights)[0])
    if cat is None: cat = random.choice(categories)

    ctype = cat.get("type", "")
    if ctype == "propositional_logic": return generatePropositionalSample()
    if ctype == "comparative_chains":  return generateComparativeSample()
    if ctype == "conversation":        return generateConversationSample()

    premiseTmpl = random.choice(cat.get("premise", cat.get("premises", [""])))
    queries     = cat["queries"]

    hasYes = any("yes" in str(q.get("a", "")) for q in queries)
    hasNo  = any("no" in str(q.get("a", "")) for q in queries)
    if hasYes and hasNo:
        wantYes = random.random() < 0.50
        filtered = [q for q in queries if ("yes" in str(q.get("a", ""))) == wantYes]
        queryObj = random.choice(filtered) if filtered else random.choice(queries)
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

    if ctype == "direct_arithmetic":
        isSub = "minus" in premiseTmpl
        if isSub:
            n1 = random.randint(1, 9); n2 = random.randint(0, n1); ans = n1 - n2
        else:
            n1 = random.randint(0, 9); n2 = random.randint(0, 9 - n1); ans = n1 + n2
        bindings["n1_word"]  = NUM_WORDS[n1]
        bindings["n2_word"]  = NUM_WORDS[n2]
        bindings["ans_word"] = NUM_WORDS[ans] if ans > 0 else "zero"

    elif ctype == "arithmetic" or "Arithmetic" in cat.get("name", ""):
        isSub = "gave" in premiseTmpl and "to {actor2}" in premiseTmpl
        if isSub:
            n1 = random.randint(1, 5); n2 = random.randint(1, n1); ans = n1 - n2
        else:
            n1 = random.randint(1, 4); n2 = random.randint(1, 4); ans = n1 + n2
            
        bindings["n1_word"]   = NUM_WORDS[n1]
        bindings["n2_word"]   = NUM_WORDS[n2]
        bindings["ans_word"]  = NUM_WORDS[ans] if ans > 0 else "none"
        bindings["item1_n1"]  = pluralize(bindings["item1"], n1)
        bindings["item1_n2"]  = pluralize(bindings["item1"], n2)
        bindings["item1_ans"] = pluralize(bindings["item1"], ans)

    rawQ      = queryObj["q"]
    qStr      = random.choice(rawQ) if isinstance(rawQ, list) else rawQ
    pStr      = premiseTmpl.format(**bindings)
    qRendered = qStr.format(**bindings)
    aRendered = queryObj["a"].format(**bindings)

    return formatQaSequence(pStr, qRendered, aRendered)