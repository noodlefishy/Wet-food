import json
import random
import re

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

def format_item_list(items, article="the"):
    if not items: return ""
    if len(items) == 1: return f"{article} {items[0]}"
    if len(items) == 2: return f"{article} {items[0]} and {article} {items[1]}"
    return ", ".join([f"{article} {i}" for i in items[:-1]]) + f" and {article} {items[-1]}"

def generate_atomic_sample(log_cfg, target_category=None):
    pools = log_cfg["pools"]
    rules = log_cfg.get("world_rules", {})
    preds = log_cfg["predicates"]
    queries = log_cfg["queries"]
    needs = log_cfg.get("states", {}).get("needs", [])

    num_actors = random.choice(rules.get("actors_per_story", [1, 2]))
    active_actors = random.sample(pools["actor"], num_actors)
    avail_items = list(pools["item"])
    random.shuffle(avail_items)

    world = {}
    story_clauses = []

    for actor in active_actors:
        world[actor] = {"location": None, "state": None, "items": []}
        actor_sentences = []

        if needs and random.random() < rules.get("state_probability", 0.5):
            need = random.choice(needs)
            world[actor]["state"] = need["state"]
            world[actor]["location"] = need["target"]
            actor_sentences.append(random.choice(preds["state"]).format(actor=actor, state=need["state"]))
            actor_sentences.append(random.choice(preds["motion"]).format(actor=actor, location=need["target"]))
        else:
            loc = random.choice(pools["location"])
            world[actor]["location"] = loc
            actor_sentences.append(random.choice(preds["motion"]).format(actor=actor, location=loc))

        k_items = random.choice(rules.get("items_per_actor", [0, 1, 2]))
        assigned = [avail_items.pop() for _ in range(min(k_items, len(avail_items)))]
        world[actor]["items"] = assigned

        if assigned:
            item_str = format_item_list(assigned, article=random.choice(["a", "the"]))
            actor_sentences.append(random.choice(preds["possession"]).format(actor=actor, items=item_str))

        if len(actor_sentences) > 1 and world[actor]["state"]:
            story_clauses.append(" . then ".join(actor_sentences[:2]) + (" . " + actor_sentences[2] if len(actor_sentences) > 2 else ""))
        else:
            story_clauses.append(" . ".join(actor_sentences))

    premise = " . ".join(story_clauses)
    if target_category is None:
        target_category = random.choice(list(queries.keys()))
        
    q_tmpl = random.choice(queries[target_category])
    t_actor = random.choice(active_actors)
    absent_actors = [a for a in pools["actor"] if a not in active_actors]
    unmentioned = random.choice(absent_actors) if absent_actors else "nobody"

    q_str, a_str = "", ""

    if "location" in target_category or "Single" in target_category or "Frame" in target_category:
        if "who" in q_tmpl:
            q_str = q_tmpl.format(location=world[t_actor]["location"])
            a_str = t_actor
        else:
            q_str = q_tmpl.format(actor=t_actor)
            a_str = world[t_actor]["location"]

    elif "Lists" in target_category:
        if random.random() < 0.2 and unmentioned != "nobody":
            q_str = q_tmpl.format(actor=unmentioned)
            a_str = "nothing"
        else:
            q_str = q_tmpl.format(actor=t_actor)
            items = world[t_actor]["items"]
            a_str = "nothing" if not items else format_item_list(items, article="")

    elif "Counting" in target_category:
        if random.random() < 0.2 and unmentioned != "nobody":
            q_str = q_tmpl.format(actor=unmentioned)
            a_str = "none"
        else:
            q_str = q_tmpl.format(actor=t_actor)
            count_map = {0: "none", 1: "one", 2: "two"}
            a_str = count_map[len(world[t_actor]["items"])]

    elif "Truth" in target_category:
        loc = world[t_actor]["location"]
        if random.random() < 0.5:
            q_str = q_tmpl.format(actor=t_actor, location=loc)
            a_str = "yes"
        else:
            wrong_loc = random.choice([l for l in pools["location"] if l != loc])
            q_str = q_tmpl.format(actor=t_actor, location=wrong_loc)
            a_str = "no"

    elif "Polar Possession" in target_category:
        held = world[t_actor]["items"]
        if held and random.random() < 0.5:
            q_str = q_tmpl.format(actor=t_actor, item=random.choice(held))
            a_str = "yes"
        else:
            not_held = [i for i in pools["item"] if i not in held]
            q_str = q_tmpl.format(actor=t_actor, item=random.choice(not_held))
            a_str = "no"

    elif "Motivations" in target_category:
        st = world[t_actor]["state"]
        if st and "why" in q_tmpl:
            q_str = q_tmpl.format(actor=t_actor, location=world[t_actor]["location"])
            a_str = st
        else:
            active_with_state = [a for a in active_actors if world[a]["state"]]
            if active_with_state:
                ch = random.choice(active_with_state)
                q_str = q_tmpl.format(state=world[ch]["state"], actor=ch, location="")
                a_str = ch
            else:
                q_str = q_tmpl.format(state=random.choice(needs)["state"], actor="", location="")
                a_str = "nobody"

    return format_qa_sequence(premise, q_str, a_str)

def generate_benchmark_suite(log_cfg, samples_per_category=400):
    dataset = {}
    for cat in log_cfg["queries"].keys():
        dataset[cat] = []
        seen = set()
        attempts = 0
        while len(dataset[cat]) < samples_per_category and attempts < samples_per_category * 15:
            attempts += 1
            seq = generate_atomic_sample(log_cfg, target_category=cat)
            p, t = split_qa(seq)
            if p not in seen:
                seen.add(p)
                dataset[cat].append({"q": p, "target": t})
    return dataset