import json, random, re
from chunks import preprocess_chunks

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

VERB_JITTER = {
    "went to the": ["went to the", "walked to the", "arrived at the"],
    "is in the":   ["is in the", "was in the"],
    "found the":   ["found the", "got the", "picked up the"],
    "gave the":    ["gave the", "handed the"],
    "has the":     ["has the", "had the"]
}

DISTRACTOR_TEMPLATES = [
    "{actor} saw a little bird",
    "{actor} sat in the sun",
    "{actor} wanted to play",
    "{actor} was very happy",
    "{actor} looked around"
]

def applyVerbJitter(text):
    for base, variants in VERB_JITTER.items():
        if base in text and random.random() < 0.40:
            text = text.replace(base, random.choice(variants), 1)
    return text

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
        answer  = "rule holds : yes"
    else:
        premise = f"if {rule['cond']} then {rule['concl']} . {rule['concl']} is not true"
        answer  = "not true : no"
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
        answer   = f"{sup} : {e1}"
    else:
        question = f"who is the {opp_sup}"
        answer   = f"{opp_sup} : {e3}"
    return formatQaSequence(premise, question, answer)

# ── General Conversation ───────────────────────────────────────────────────────
_CONV = {
    "greet": {
        "u": ["hello", "hi", "hey", "good morning", "good day"],
        "r": ["hello friend", "hi there", "hello", "hey there", "good to see you"]
    },
    "howru": {
        "u": ["how are you", "how are you doing", "how is your day", "are you well"],
        "r": ["i am doing great", "i am doing well thanks", "all good here", "i am fine thank you"]
    },
    "identity": {
        "u": ["who are you", "what are you", "are you an ai", "what can you do"],
        "r": [
            "i am a tiny ai on two pico chips",
            "i am an ai that can reason and tell stories",
            "i live in two microcontroller chips"
        ]
    },
    "thanks": {
        "u": ["thank you", "thanks", "thank you very much"],
        "r": ["you are welcome", "happy to help", "no problem", "glad to help"]
    },
    "bye": {
        "u": ["goodbye", "bye", "see you later", "have a good day"],
        "r": ["goodbye friend", "see you later", "take care", "bye"]
    }
}

def generateConversationSample():
    intent = random.choice(list(_CONV.keys()))
    utterance = random.choice(_CONV[intent]["u"])
    reply     = random.choice(_CONV[intent]["r"])
    return formatQaSequence(f"user : {utterance}", "how do you respond", f"reply : {reply}")

NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
FEMALE_ACTORS = {"alice", "luna", "sophie", "chloe", "daisy", "bella", "lucy", "lily", "mia", "ruby", "zoe"}

def generateAtomicSample(cfg, targetCategory=None, injectDistractor=True):
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

    # 1. Hard-negative balance for polar queries
    hasYes = any(": yes" in q.get("a", "") for q in queries)
    hasNo  = any(": no" in q.get("a", "")  for q in queries)
    if hasYes and hasNo:
        wantYes = random.random() < 0.45
        filtered = [q for q in queries if (": yes" in q.get("a", "")) == wantYes]
        queryObj = random.choice(filtered) if filtered else random.choice(queries)
    # 2. Anti-recency sampling: force query to target distant entity 65% of the time
    elif "{actor1}" in premiseTmpl and "{actor2}" in premiseTmpl:
        dist_actor = "actor1" if premiseTmpl.find("{actor1}") < premiseTmpl.find("{actor2}") else "actor2"
        rec_actor = "actor2" if dist_actor == "actor1" else "actor1"
        
        if random.random() < 0.50:
            d = [q for q in queries if f"{{{dist_actor}}}" in str(q)]
            queryObj = random.choice(d) if d else random.choice(queries)
        else:
            r = [q for q in queries if f"{{{rec_actor}}}" in str(q)]
            queryObj = random.choice(r) if r else random.choice(queries)
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
    pStr      = applyVerbJitter(premiseTmpl.format(**bindings))
    qRendered = qStr.format(**bindings)
    aRendered = queryObj["a"].format(**bindings)

    if injectDistractor and random.random() < 0.35:
        unused = [a for a in actors if a not in qRendered and a not in aRendered]
        if unused:
            distPhrase = random.choice(DISTRACTOR_TEMPLATES).format(actor=random.choice(unused))
            # Shuffle distractor to beginning, middle, or end to prevent "ignore last sentence" heuristics
            parts = [s.strip() for s in pStr.split(" . ") if s.strip()]
            parts.insert(random.randint(0, len(parts)), distPhrase)
            pStr = " . ".join(parts)

    return formatQaSequence(pStr, qRendered, aRendered)