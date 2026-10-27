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

def generate_atomic_sample(cfg, target_category=None):
    pools = cfg["pools"]
    pair_pools = cfg.get("pair_pools", {})
    categories = cfg["categories"]
    
    weights = [c.get("weight", 2.0) for c in categories]
    cat = next((c for c in categories if c["name"] == target_category), None) if target_category else random.choices(categories, weights=weights)[0]
    if cat is None: cat = random.choice(categories)
    
    premise_tmpl = random.choice(cat["premises"])
    query_obj = random.choice(cat["queries"])
    
    actors = random.sample(pools["actor"], len(pools["actor"]))
    locs = random.sample(pools["location"], len(pools["location"]))
    items = random.sample(pools["item"], len(pools["item"]))
    agent_pool = pair_pools.get("agent", [])
    agents = random.sample(agent_pool, min(2, len(agent_pool))) if agent_pool else []
    
    bindings = {
        "actor1": actors[0], "actor2": actors[1], "actor3": actors[2],
        "location1": locs[0], "location2": locs[1], "location3": locs[2],
        "item1": items[0], "item2": items[1], "item3": items[2],
    }
    if agents:
        bindings["agent1_name"] = agents[0]["name"]
        bindings["agent1_pronoun"] = agents[0]["pronoun"]
        
    p_str = premise_tmpl.format(**bindings)
    q_str = query_obj["q"].format(**bindings)
    a_str = query_obj["a"].format(**bindings)
    
    return format_qa_sequence(p_str, q_str, a_str)

def generate_benchmark_suite(cfg, samples_per_category=400):
    dataset = {}
    for cat in cfg["categories"]:
        cat_name = cat["name"]
        dataset[cat_name] = []
        seen = set()
        attempts = 0
        while len(dataset[cat_name]) < samples_per_category and attempts < samples_per_category * 15:
            attempts += 1
            seq = generate_atomic_sample(cfg, target_category=cat_name)
            p, t = split_qa(seq)
            if p not in seen:
                seen.add(p)
                dataset[cat_name].append({"q": p, "target": t})
    return dataset