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

FEMALE_ACTORS = {
    "alice", "luna", "sophie", "chloe", "daisy", "bella", "lucy", "lily",
    "mia", "ruby", "zoe", "claire", "anna", "emma", "sally", "molly", "amy", "mary"
}

# 1. Strictly Proper Names (Natural English, zero articles, verified in 2048 vocab.py)
CLEAN_OPEN_ACTORS = [
    "harry", "claire", "henry", "george", "peter", "mary",
    "emma", "anna", "tom", "sally", "billy", "sammy", "molly", "amy"
]

# 2. True Geographic Locations & Rooms (Natural for "went to" and "north of")
CLEAN_OPEN_LOCATIONS = [
    "farm", "island", "office", "bathroom", "mountain",
    "river", "station", "beach", "castle", "shop", "school", "tower", "closet"
]

# 3. Strictly Countable Items (Both singular AND plural verified in 2048 vocab.py)
CLEAN_COUNTABLE_ITEMS = [
    "ball", "box", "book", "cup", "key", "bag", "toy", "stick",
    "blanket", "apple", "hat", "bottle", "bowl", "plate", "pencil",
    "coin", "flower", "rock", "cookie", "egg"
]

# 4. Mutually Exclusive Adjective Poles (Ensures negative answers are 100% logically valid)
ADJ_PAIRS = [
    ("hot", "cold"),
    ("clean", "dirty"),
    ("open", "closed"),
    ("heavy", "light"),
    ("red", "blue"),
    ("green", "yellow")
]

STATES = ["hungry", "tired", "thirsty"]
MOVE_VERBS = ["went to", "walked to", "ran to", "arrived at"]

def _sample_entities(pools, force_open=False):
    """
    Samples entities. 40% of the time, mixes in open-domain TinyStories nouns
    to break SFT entity-subspace confinement and force pure syntactic role binding.
    """
    use_open = force_open or (random.random() < 0.40)
    actor_pool = (pools["actor"] + CLEAN_OPEN_ACTORS) if use_open else pools["actor"]
    loc_pool = (pools["location"] + CLEAN_OPEN_LOCATIONS) if use_open else pools["location"]
    item_pool = CLEAN_COUNTABLE_ITEMS if use_open else pools["item"]
    col_pool = pools.get("colour", ["red", "blue", "yellow", "green"])

    actors = random.sample(actor_pool, min(5, len(actor_pool)))
    locs = random.sample(loc_pool, min(5, len(loc_pool)))
    items = random.sample(item_pool, min(5, len(item_pool)))
    colours = random.sample(col_pool, min(4, len(col_pool)))
    return actors, locs, items, colours

def _pick_unused(pool, used_text, exclude=()):
    used = set(used_text.lower().split()) | set(exclude)
    candidates = [w for w in pool if w.lower() not in used]
    return random.choice(candidates) if candidates else None

def _inject_distractor(premise_str, question_str, pools):
    if len(premise_str.split()) > 28:
        return premise_str

    combined = premise_str + " " + question_str
    u_actor = _pick_unused(pools["actor"] + CLEAN_OPEN_ACTORS, combined)
    u_loc = _pick_unused(pools["location"] + CLEAN_OPEN_LOCATIONS, combined, exclude=("kitchen", "room"))
    u_item = _pick_unused(CLEAN_COUNTABLE_ITEMS, combined)
    u_col = _pick_unused(pools.get("colour", ["red", "blue", "yellow", "green"]), combined)

    if random.random() < 0.5 and u_actor and u_loc:
        distractor = f"{u_actor} {random.choice(MOVE_VERBS)} the {u_loc}"
    elif u_item and u_col:
        distractor = f"the {u_item} is {u_col}"
    else:
        return premise_str

    clauses = [c.strip() for c in premise_str.split(".") if c.strip()]
    clauses.insert(random.randint(0, len(clauses)), distractor)
    return " . ".join(clauses)

# -------------------------------------------------------------------------
# Specialized Generators (Categories 21 - 25 & Anti-Shortcut Handlers)
# -------------------------------------------------------------------------

def generateDirectArithmeticSample():
    """Category 21: Direct Symbolic Arithmetic."""
    if random.random() < 0.5:
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

def generateWordArithmeticSample(cfg):
    """Category 22: Word Problem Arithmetic with Strictly Countable Items."""
    pools = cfg["pools"]
    actors, _, items, _ = _sample_entities(pools)
    a1, a2 = actors[0], actors[1]
    it = items[0]

    if random.random() < 0.5:
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

def generateMultiCountSample(cfg):
    """Category 23: Counting 1 to 4 Diverse Objects."""
    pools = cfg["pools"]
    actors, _, items, _ = _sample_entities(pools)
    a1, a2 = actors[0], actors[1]
    n = random.randint(1, 4)
    held = items[:n]
    if n == 1:
        p = f"{a1} has the {held[0]} . {a2} has the {items[4]}"
    elif n == 2:
        p = f"{a1} has the {held[0]} and the {held[1]}"
    else:
        p = f"{a1} has " + " , ".join(f"the {x}" for x in held[:-1]) + f" , and the {held[-1]}"
    return formatQaSequence(p, f"how many items does {a1} have", f"{' '.join(held)} : {NUM_WORDS[n]}")

def generateDisjunctionSample(cfg):
    """
    Eliminates the 'second-slot' position shortcut in Disjunctive Syllogism (Task 10).
    50% of the time negates loc1 (ans=loc2), 50% negates loc2 (ans=loc1).
    Randomizes clause arrival order.
    """
    pools = cfg["pools"]
    actors, locs, _, _ = _sample_entities(pools)
    a1 = actors[0]
    l1, l2 = locs[0], locs[1]

    negate_second = random.random() < 0.5
    negated_loc = l2 if negate_second else l1
    correct_loc = l1 if negate_second else l2

    c_disj = f"{a1} is in the {l1} or the {l2}"
    c_neg = f"{a1} is not in the {negated_loc}"

    clauses = [c_disj, c_neg] if random.random() < 0.5 else [c_neg, c_disj]
    p = " . ".join(clauses)
    q = f"where is {a1}"
    a = f"{c_neg} : {correct_loc}"
    return formatQaSequence(p, q, a)

def generateSingleTransferSample(cfg):
    """Eliminates the grammatical subject bias in single object transfers."""
    pools = cfg["pools"]
    actors, _, items, _ = _sample_entities(pools)
    a1, a2 = actors[0], actors[1]
    it = items[0]

    p = f"{a1} gave the {it} to {a2}"
    mode = random.randrange(3)
    if mode == 0:
        q = f"who has the {it}"
        a = f"{a2} ."
    elif mode == 1:
        q = f"who gave the {it} to {a2}"
        a = f"{a1} ."
    else:
        q = f"who got the {it} from {a1}"
        a = f"{a2} ."
    return formatQaSequence(p, q, a)

def generateComposedCoTSample(cfg, allow_distractor=True):
    """Category 24: Multi-hop reasoning across composed skills with Micro-CoT scratchpad."""
    pools = cfg["pools"]
    actors, locs, items, _ = _sample_entities(pools)
    a1, a2, a3 = actors[0], actors[1], actors[2]
    l1, l2, l3 = locs[0], locs[1], locs[2]
    i1, i2 = items[0], items[1]
    v1 = random.choice(MOVE_VERBS)
    v2 = random.choice(MOVE_VERBS)

    mode = random.randrange(6)
    if mode == 0:
        # Transfer + Movement
        c1 = f"{a1} gave the {i1} to {a2}"
        c2 = f"{a2} {v1} the {l1}"
        c3 = f"{a3} {v2} the {l2}"
        clauses = [c1, c2, c3] if random.random() < 0.5 else [c1, c3, c2]
        p = " . ".join(clauses)
        q = f"where is the {i1} now"
        a = f"{a2} has the {i1} . {a2} is in the {l1} : {l1}"
    elif mode == 1:
        # Drop / Found + Transfer + Move
        p = f"{a1} found the {i1} in the {l1} . {a1} gave the {i1} to {a2} . {a2} {v1} the {l2}"
        q = f"where is the {i1} now"
        a = f"{a2} has the {i1} . {a2} is in the {l2} : {l2}"
    elif mode == 2:
        # Disjunction + Possession (bidirectional negation)
        negate_second = random.random() < 0.5
        neg_loc = l2 if negate_second else l1
        ans_loc = l1 if negate_second else l2
        c1 = f"{a1} is in the {l1} or the {l2}"
        c2 = f"{a1} is not in the {neg_loc}"
        c3 = f"{a1} has the {i1}"
        clauses = [c1, c2, c3]
        random.shuffle(clauses)
        p = " . ".join(clauses)
        q = f"where is the {i1} now"
        a = f"{a1} is in the {ans_loc} : {ans_loc}"
    elif mode == 3:
        # Deduction + Transfer
        p = f"cats like the {i1} . {a1} is a cat . {a2} gave the {i1} to {a3} . {a2} gave the {i2} to {a1}"
        q = f"what does {a1} like"
        a = f"{a1} is a cat : {i1}"
    elif mode == 4:
        # Location + Compass Transitivity
        p = f"{a1} {v1} the {l1} . the {l1} is north of the {l2} . the {l2} is north of the {l3}"
        q = f"what is south of the {l1}"
        a = f"{l2} is south of the {l1} : {l2}"
    else:
        # Single Transfer + Item Status
        p = f"{a1} gave the {i1} to {a2} . the {i1} is dirty"
        q = f"who has the {i1}"
        a = f"{a2} ."

    if allow_distractor and random.random() < 0.35:
        p = _inject_distractor(p, q, pools)

    return formatQaSequence(p, q, a)

def generateRagRuleSample(cfg, allow_distractor=True):
    """
    Category 25: RAG Rules & Grounded Abstention.
    Uses mutually exclusive adjective poles to eliminate the contradictory logic bug.
    """
    pools = cfg["pools"]
    actors, locs, items, colours = _sample_entities(pools)
    a1, a2, a3 = actors[0], actors[1], actors[2]
    l1, l2 = locs[0], locs[1]
    i1, i2 = items[0], items[1]
    c1 = colours[0]

    # Sample a clean, mutually exclusive property pair
    pair = random.choice(ADJ_PAIRS)
    adj_pos, adj_neg = (pair[0], pair[1]) if random.random() < 0.5 else (pair[1], pair[0])
    rule_adj = random.choice(["safe", "dangerous", "good", "bad"])

    r = random.random()
    if r < 0.22:
        # Mode 0: Rule-First Property Inheritance (Rule-first CoT forces premise comparison)
        is_pos = random.random() < 0.5
        rule = f"{adj_pos} things are {rule_adj}"
        fact = f"the {i1} is {adj_pos if is_pos else adj_neg}"
        clauses = [rule, fact] if random.random() < 0.5 else [fact, rule]
        p = " . ".join(clauses)
        q = f"is the {i1} {rule_adj}"
        a = f"{rule} . {fact} : {'yes' if is_pos else 'no'}"
    elif r < 0.44:
        # Mode 1: Prepositional Location Rule ("things in the {loc} are {adj}")
        is_pos = random.random() < 0.5
        rule = f"things in the {l1} are {adj_pos}"
        actual_loc = l1 if is_pos else l2
        fact = f"the {i1} is in the {actual_loc}" if random.random() < 0.5 else f"{a1} put the {i1} in the {actual_loc}"
        clauses = [rule, fact] if random.random() < 0.5 else [fact, rule]
        p = " . ".join(clauses)
        q = f"is the {i1} {adj_pos}"
        a = f"{rule} . {fact} : {'yes' if is_pos else 'no'}"
    elif r < 0.74:
        # Mode 2: Wh- rules (where / what colour / who)
        wh_mode = random.randrange(4)
        if wh_mode == 0:
            state = random.choice(STATES)
            rule = f"if {a1} is {state} then {a1} is in the {l1}"
            fact = f"{a1} is {state}"
            clauses = [rule, fact] if random.random() < 0.5 else [fact, rule]
            p = " . ".join(clauses)
            q = f"where is {a1}"
            a = f"{a1} is {state} : {l1}"
        elif wh_mode == 1:
            rule = f"{adj_pos} things are in the {l1}"
            fact = f"the {i1} is {adj_pos}"
            clauses = [rule, fact] if random.random() < 0.5 else [fact, rule]
            p = " . ".join(clauses)
            q = f"where is the {i1}"
            a = f"the {i1} is {adj_pos} : {l1}"
        elif wh_mode == 2:
            rule = f"if the {i1} is in the {l1} then it is {c1}"
            fact = f"the {i1} is in the {l1}" if random.random() < 0.5 else f"{a1} put the {i1} in the {l1}"
            clauses = [rule, fact] if random.random() < 0.5 else [fact, rule]
            p = " . ".join(clauses)
            q = f"what colour is the {i1}"
            a = f"the {i1} is in the {l1} : {c1}"
        else:
            rule = f"cats like the {i1}"
            fact = f"{a1} is a cat"
            clauses = [rule, fact] if random.random() < 0.5 else [fact, rule]
            p = " . ".join(clauses)
            q = f"who likes the {i1}"
            a = f"{a1} is a cat : {a1}"
    else:
        # Mode 3: Grounded Abstention (Entity is genuinely absent from premise)
        abs_mode = random.randrange(4)
        v1 = random.choice(MOVE_VERBS)
        v2 = random.choice(MOVE_VERBS)
        if abs_mode == 0:
            p = f"{a2} {v1} the {l1} . {a3} {v2} the {l2}"
            q = f"where is {a1}"
            a = "nowhere"
        elif abs_mode == 1:
            p = f"{a1} {v1} the {l1} . the {i2} is in the {l2}"
            q = f"where is the {i1}"
            a = "nowhere"
        elif abs_mode == 2:
            p = f"{a1} {v1} the {l1} . {a2} {v2} the {l2}"
            q = f"who has the {i1}"
            a = "nobody"
        else:
            p = f"{a2} has the {i1} . {a3} {v1} the {l1}"
            q = f"what does {a1} have"
            a = "nothing"
        return formatQaSequence(p, q, a)

    if allow_distractor and random.random() < 0.4:
        p = _inject_distractor(p, q, pools)

    return formatQaSequence(p, q, a)

# -------------------------------------------------------------------------
# Master SFT Dispatcher
# -------------------------------------------------------------------------

def generateAtomicSample(cfg, targetCategory=None):
    pools = cfg["pools"]
    categories = cfg["categories"]

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

    ctype = cat.get("type", "")
    if ctype == "direct_arithmetic":
        return generateDirectArithmeticSample()
    if ctype == "arithmetic":
        return generateWordArithmeticSample(cfg)
    if ctype == "multi_counting":
        return generateMultiCountSample(cfg)
    if ctype == "disjunction":
        return generateDisjunctionSample(cfg)
    if ctype == "composed_cot":
        return generateComposedCoTSample(cfg, allow_distractor=(targetCategory is None))
    if ctype == "rag_rules":
        return generateRagRuleSample(cfg, allow_distractor=(targetCategory is None))

    # Single transfer edge-case injection into Task 05 (35% of the time)
    if "05:" in cat["name"] and targetCategory is None and random.random() < 0.35:
        return generateSingleTransferSample(cfg)

    # Indefinite Knowledge shortcut breaker (Task 10)
    if "10:" in cat["name"] and targetCategory is None:
        return generateDisjunctionSample(cfg)

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

    if targetCategory is None:
        if random.random() < 0.20 and " went to the " in pStr:
            pStr = pStr.replace(" went to the ", f" {random.choice(MOVE_VERBS)} the ")
        if random.random() < 0.25:
            pStr = _inject_distractor(pStr, qRendered, pools)

    return formatQaSequence(pStr, qRendered, aRendered)