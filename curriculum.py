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

VERB_JITTER = {
    "went to the": ["went to the", "walked to the", "ran to the", "arrived at the"],
    "is in the":   ["is in the", "was in the", "went to the"],
    "found the":   ["found the", "got the", "picked up the"],
    "gave the":    ["gave the", "handed the"],
    "has the":     ["has the", "had the", "holds the"]
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
    {"cond": "the jar is full",    "concl": "the jar is heavy",      "target": "is the jar heavy"},
]

def generatePropositionalSample():
    rule = random.choice(PROP_RULES)
    modePonens = random.random() < 0.5   # True → premise holds → answer yes
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
    # Chain: e1 > e2 > e3, so e1 is superlative, e3 is opposite superlative
    opp_sup = {"biggest": "smallest", "smallest": "biggest",
               "fastest": "slowest",  "slowest":  "fastest"}[sup]
    askTop = random.random() < 0.7
    if askTop:
        premise = f"{e1} is {comp} than {e2} . {e2} is {comp} than {e3}"
        question = f"who is the {sup}"
        answer   = e1
    else:
        premise  = f"{e1} is {comp} than {e2} . {e2} is {comp} than {e3}"
        question = f"who is the {opp_sup}"
        answer   = e3
    return formatQaSequence(premise, question, answer)

# ── General Conversation ───────────────────────────────────────────────────────
_CONV = {
    "greet": {
        "u": ["hello", "hi", "hey", "good morning", "good day", "greetings", "hello there"],
        "r": ["hello friend", "hi there", "greetings", "hello", "hey there", "good to see you"]
    },
    "howru": {
        "u": ["how are you", "how are you doing", "how is your day", "are you well", "how do you feel"],
        "r": ["i am doing great", "i am doing well thanks", "all good here", "i am fine thank you", "doing well"]
    },
    "identity": {
        "u": ["who are you", "what are you", "are you an ai", "what can you do", "tell me about yourself"],
        "r": [
            "i am a tiny ai running on two microcontrollers",
            "i am an ai that can reason and tell stories",
            "i am a small language model built on two pico chips",
            "i live in a microcontroller and i can reason and chat",
        ]
    },
    "thanks": {
        "u": ["thank you", "thanks", "thank you very much", "thanks a lot"],
        "r": ["you are welcome", "happy to help", "no problem", "glad to help", "anytime"]
    },
    "bye": {
        "u": ["goodbye", "bye", "see you later", "have a good day", "farewell"],
        "r": ["goodbye friend", "see you later", "have a good day", "take care", "bye"]
    },
    "fun": {
        "u": ["tell me something interesting", "say something", "what do you think", "do you like stories"],
        "r": [
            "i find it interesting that small ai can reason",
            "i like the idea of thinking in tiny spaces",
            "it is nice to talk with you",
            "i think curiosity is a great thing",
            "stories are a wonderful way to learn",
        ]
    },
}

def generateConversationSample():
    intent = random.choice(list(_CONV.keys()))
    utterance = random.choice(_CONV[intent]["u"])
    reply     = random.choice(_CONV[intent]["r"])
    return formatQaSequence(f"user : {utterance}", "how do you respond", reply)

# ── Main sample generator ──────────────────────────────────────────────────────
NUM_WORDS = ["zero", "one", "two", "three", "four", "five",
             "six", "seven", "eight", "nine", "ten"]

def generateAtomicSample(cfg, targetCategory=None, injectDistractor=True):
    pools     = cfg["pools"]
    pairPools = cfg.get("pair_pools", {})
    categories = cfg["categories"]
    weights    = [c.get("weight", 2.0) for c in categories]

    cat = (next((c for c in categories if c["name"] == targetCategory), None)
           if targetCategory else random.choices(categories, weights=weights)[0])
    if cat is None:
        cat = random.choice(categories)

    # Delegate fully-generated types immediately
    ctype = cat.get("type", "")
    if ctype == "propositional_logic":  return generatePropositionalSample()
    if ctype == "comparative_chains":   return generateComparativeSample()
    if ctype == "conversation":         return generateConversationSample()

    premiseTmpl = random.choice(cat.get("premise", cat.get("premises", [""])))

    queries  = cat["queries"]
    hasYes   = any(q["a"] == "yes" for q in queries)
    hasNo    = any(q["a"] == "no"  for q in queries)
    if hasYes and hasNo:
        wantYes   = random.random() < 0.5
        filtered  = [q for q in queries if (q["a"] == "yes") == wantYes]
        queryObj  = random.choice(filtered) if filtered else random.choice(queries)
    else:
        queryObj = random.choice(queries)

    actors  = random.sample(pools["actor"],    len(pools["actor"]))
    locs    = random.sample(pools["location"], len(pools["location"]))
    items   = random.sample(pools["item"],     len(pools["item"]))
    colours = random.sample(pools.get("colour", ["red", "blue", "yellow", "green"]), 4)
    agentPool = pairPools.get("agent", [])
    agents    = random.sample(agentPool, min(2, len(agentPool))) if agentPool else []

    bindings = {
        "actor1": actors[0], "actor2": actors[1], "actor3": actors[2],
        "location1": locs[0], "location2": locs[1], "location3": locs[2],
        "item1": items[0],   "item2": items[1],   "item3": items[2],
        "colour1": colours[0], "colour2": colours[1], "colour3": colours[2]
    }
    if agents:
        bindings["agent1_name"]    = agents[0]["name"]
        bindings["agent1_pronoun"] = agents[0]["pronoun"]

    if ctype == "direct_arithmetic":
        isSub = "minus" in premiseTmpl
        if isSub:
            n1 = random.randint(1, 9); n2 = random.randint(0, n1); ans = n1 - n2
        else:
            n1 = random.randint(0, 9); n2 = random.randint(0, 9 - n1); ans = n1 + n2
        bindings["n1_word"]  = NUM_WORDS[n1]
        bindings["n2_word"]  = NUM_WORDS[n2]
        bindings["ans_word"] = NUM_WORDS[ans] if ans > 0 else "zero"

    elif ctype == "arithmetic" or (ctype == "" and "Arithmetic" in cat.get("name", "")):
        isSub = "to {actor2}" in premiseTmpl or "gave {n2_word} to" in premiseTmpl
        if isSub:
            n1 = random.randint(1, 5); n2 = random.randint(1, n1); ans = n1 - n2
        else:
            n1 = random.randint(1, 4); n2 = random.randint(1, 4); ans = n1 + n2
        
        bindings["n1_word"]  = NUM_WORDS[n1]
        bindings["n2_word"]  = NUM_WORDS[n2]
        bindings["ans_word"] = NUM_WORDS[ans] if ans > 0 else "none"
        
        baseItem = bindings["item1"]
        bindings["item1_n1"] = pluralize(baseItem, n1)
        bindings["item1_n2"] = pluralize(baseItem, n2)
        bindings["item1_ans"] = pluralize(baseItem, ans)

    rawQ     = queryObj["q"]
    qStr     = random.choice(rawQ) if isinstance(rawQ, list) else rawQ
    pStr     = applyVerbJitter(premiseTmpl.format(**bindings))
    qRendered = qStr.format(**bindings)
    aRendered = queryObj["a"].format(**bindings)

    if injectDistractor and random.random() < 0.35:
        unusedActors = [a for a in pools["actor"] if a not in qRendered and a not in aRendered]
        if unusedActors:
            distActor  = random.choice(unusedActors)
            distPhrase = random.choice(DISTRACTOR_TEMPLATES).format(actor=distActor)
            pos        = random.choice(["prefix", "suffix", "middle"])
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
normalize_text      = normalizeText
format_qa_sequence  = formatQaSequence
split_qa            = splitQa
generate_atomic_sample = generateAtomicSample
