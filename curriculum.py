import json, random, re

def normalize_text(text):
    t = text.strip()
    t = re.sub(r'\s*([?.!:,])\s*', r' \1 ', t)
    return re.sub(r'\s+', ' ', t).strip()

def format_qa_sequence(premise, question, answer):
    p = normalize_text(premise.strip().rstrip(". "))
    q = normalize_text(question.strip().rstrip("?.: "))
    a = normalize_text(answer.strip().rstrip(". "))
    return f"[c] {p} . [q] {q} ? [a] {a} . <eos>"

def split_qa(sequence_str):
    parts = sequence_str.split(" [a] ")
    prompt = parts[0].strip() + " [a]"
    target = parts[1].replace("<eos>", "").strip()
    return prompt, target

# Lexical jitter replacements for action verbs (all words exist in core vocab)
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

def apply_verb_jitter(text):
    for base, variants in VERB_JITTER.items():
        if base in text and random.random() < 0.4:
            text = text.replace(base, random.choice(variants), 1)
    return text

def generate_atomic_sample(cfg, target_category=None, inject_distractor=True):
    pools = cfg["pools"]
    pair_pools = cfg.get("pair_pools", {})
    categories = cfg["categories"]
    
    weights = [c.get("weight", 2.0) for c in categories]
    cat = next((c for c in categories if c["name"] == target_category), None) if target_category else random.choices(categories, weights=weights)[0]
    if cat is None: cat = random.choice(categories)
    
    premise_tmpl = random.choice(cat["premise"] if "premise" in cat else cat["premises"])
    
    # 50/50 Polarity Balancer for polar (Yes/No) questions
    queries = cat["queries"]
    has_yes = any(q["a"] == "yes" for q in queries)
    has_no = any(q["a"] == "no" for q in queries)
    
    if has_yes and has_no:
        want_yes = (random.random() < 0.5)
        filtered = [q for q in queries if (q["a"] == "yes") == want_yes]
        query_obj = random.choice(filtered) if filtered else random.choice(queries)
    else:
        query_obj = random.choice(queries)
        
    actors = random.sample(pools["actor"], len(pools["actor"]))
    locs = random.sample(pools["location"], len(pools["location"]))
    items = random.sample(pools["item"], len(pools["item"]))
    colours = random.sample(pools.get("colour", ["red", "blue", "yellow", "green"]), 4)
    agent_pool = pair_pools.get("agent", [])
    agents = random.sample(agent_pool, min(2, len(agent_pool))) if agent_pool else []
    
    NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    
    bindings = {
        "actor1": actors[0], "actor2": actors[1], "actor3": actors[2],
        "location1": locs[0], "location2": locs[1], "location3": locs[2],
        "item1": items[0], "item2": items[1], "item3": items[2],
        "colour1": colours[0], "colour2": colours[1], "colour3": colours[2]
    }
    if agents:
        bindings["agent1_name"] = agents[0]["name"]
        bindings["agent1_pronoun"] = agents[0]["pronoun"]

    # Special handling for arithmetic category
    if cat.get("type") == "arithmetic" or "Arithmetic" in cat.get("name", ""):
        # Determine if premise is addition or subtraction
        is_subtraction = "to {actor2}" in premise_tmpl or "gave {n2_word} to" in premise_tmpl
        if is_subtraction:
            n1 = random.randint(1, 5)
            n2 = random.randint(1, n1)
            ans = n1 - n2
        else:
            n1 = random.randint(1, 4)
            n2 = random.randint(1, 4)
            ans = n1 + n2
        bindings["n1_word"] = NUM_WORDS[n1]
        bindings["n2_word"] = NUM_WORDS[n2]
        bindings["ans_word"] = NUM_WORDS[ans] if ans > 0 else "none"
        
    # Unpack queries if query string is a list of synonyms
    raw_q = query_obj["q"]
    q_str = random.choice(raw_q) if isinstance(raw_q, list) else raw_q
    
    p_str = premise_tmpl.format(**bindings)
    q_rendered = q_str.format(**bindings)
    a_rendered = query_obj["a"].format(**bindings)
    
    # Apply lexical phrase jitter
    p_str = apply_verb_jitter(p_str)
    
    # Anti-Shortcut: Random Distractor Injection (35% probability during training)
    if inject_distractor and random.random() < 0.35:
        # Pick actor not queried to avoid logic contradictions
        unused_actors = [a for a in pools["actor"] if a not in q_rendered and a not in a_rendered]
        if unused_actors:
            dist_actor = random.choice(unused_actors)
            dist_phrase = random.choice(DISTRACTOR_TEMPLATES).format(actor=dist_actor)
            pos = random.choice(["prefix", "suffix", "middle"])
            if pos == "prefix":
                p_str = f"{dist_phrase} . {p_str}"
            elif pos == "suffix":
                p_str = f"{p_str} . {dist_phrase}"
            else:
                parts = p_str.split(" . ")
                if len(parts) > 1:
                    idx = random.randint(1, len(parts) - 1)
                    parts.insert(idx, dist_phrase)
                    p_str = " . ".join(parts)
                else:
                    p_str = f"{p_str} . {dist_phrase}"
                    
    return format_qa_sequence(p_str, q_rendered, a_rendered)

def generate_benchmark_suite(cfg, samples_per_category=400):
    dataset = {}
    for cat in cfg["categories"]:
        cat_name = cat["name"]
        dataset[cat_name] = []
        seen = set()
        attempts = 0
        while len(dataset[cat_name]) < samples_per_category and attempts < samples_per_category * 20:
            attempts += 1
            # Benchmarks do not inject random distractors to stay deterministic
            seq = generate_atomic_sample(cfg, target_category=cat_name, inject_distractor=False)
            p, t = split_qa(seq)
            if p not in seen:
                seen.add(p)
                dataset[cat_name].append({"q": p, "target": t})
    return dataset
