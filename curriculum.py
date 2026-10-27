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
    if not items: return "nothing"
    prefix = f"{article} " if article else ""
    if len(items) == 1: return f"{prefix}{items[0]}"
    if len(items) == 2: return f"{prefix}{items[0]} and {prefix}{items[1]}"
    return ", ".join([f"{prefix}{i}" for i in items[:-1]]) + f" and {prefix}{items[-1]}"

class WorldState:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pools = cfg["pools"]
        self.rules = cfg["rules"]
        self.actor_loc = {a: random.choice(self.pools["locations"]) for a in self.pools["actors"]}
        self.item_loc = {i: random.choice(self.pools["locations"]) for i in self.pools["items"]}
        self.actor_state = {a: None for a in self.pools["actors"]}
        self.history = []

    def get_inventory(self, actor):
        return [i for i, h in self.item_loc.items() if h == actor]

def act_move(w, cfg):
    a = random.choice(w.pools["actors"])
    curr = w.actor_loc[a]
    new_loc = random.choice([l for l in w.pools["locations"] if l != curr])
    w.actor_loc[a] = new_loc
    w.history.append({"type": "move", "actor": a, "from": curr, "to": new_loc})
    tmpl = random.choice(cfg["actions"]["move"]["templates"])
    return tmpl.format(actor=a, location=new_loc)

def act_take(w, cfg):
    a = random.choice(w.pools["actors"])
    loc = w.actor_loc[a]
    floor = [i for i, place in w.item_loc.items() if place == loc]
    if not floor or len(w.get_inventory(a)) >= w.rules["max_inventory_per_actor"]:
        return None
    item = random.choice(floor)
    w.item_loc[item] = a
    w.history.append({"type": "take", "actor": a, "item": item, "loc": loc})
    tmpl = random.choice(cfg["actions"]["take"]["templates"])
    return tmpl.format(actor=a, item=item)

def act_drop(w, cfg):
    a = random.choice(w.pools["actors"])
    inv = w.get_inventory(a)
    if not inv: return None
    item = random.choice(inv)
    w.item_loc[item] = w.actor_loc[a]
    w.history.append({"type": "drop", "actor": a, "item": item, "loc": w.actor_loc[a]})
    tmpl = random.choice(cfg["actions"]["drop"]["templates"])
    return tmpl.format(actor=a, item=item)

def act_give(w, cfg):
    a = random.choice(w.pools["actors"])
    inv = w.get_inventory(a)
    peers = [p for p in w.pools["actors"] if p != a and w.actor_loc[p] == w.actor_loc[a]]
    if not inv or not peers: return None
    receiver, item = random.choice(peers), random.choice(inv)
    w.item_loc[item] = receiver
    w.history.append({"type": "give", "from": a, "to": receiver, "item": item})
    tmpl = random.choice(cfg["actions"]["give"]["templates"])
    return tmpl.format(actor=a, receiver=receiver, item=item)

def act_need(w, cfg):
    a = random.choice(w.pools["actors"])
    if w.actor_state[a] is not None: return None
    need = random.choice(cfg["states"]["needs"])
    w.actor_state[a] = need
    tmpl = random.choice(cfg["actions"]["need"]["templates"])
    return tmpl.format(actor=a, state=need["state"])

ACTION_HANDLERS = {
    "move": act_move,
    "take": act_take,
    "drop": act_drop,
    "give": act_give,
    "need": act_need
}


def resolve_actor_location(w, tmpl, actors, cfg):
    a = random.choice(actors)
    return tmpl.format(actor=a), w.actor_loc[a]

def resolve_item_location(w, tmpl, actors, cfg):
    i = random.choice(w.pools["items"])
    holder = w.item_loc[i]
    loc = w.actor_loc[holder] if holder in w.pools["actors"] else holder
    return tmpl.format(item=i), loc

def resolve_possession_list(w, tmpl, actors, cfg):
    a = random.choice(actors)
    inv = w.get_inventory(a)
    return tmpl.format(actor=a), format_item_list(inv, article="")

def resolve_possession_count(w, tmpl, actors, cfg):
    a = random.choice(actors)
    count_map = {0: "none", 1: "one", 2: "two"}
    return tmpl.format(actor=a), count_map.get(len(w.get_inventory(a)), "none")

def resolve_actor_location_polar(w, tmpl, actors, cfg):
    a = random.choice(actors)
    loc = w.actor_loc[a]
    if random.random() < 0.5:
        return tmpl.format(actor=a, location=loc), "yes"
    wrong_loc = random.choice([l for l in w.pools["locations"] if l != loc])
    return tmpl.format(actor=a, location=wrong_loc), "no"

def resolve_possession_polar(w, tmpl, actors, cfg):
    a = random.choice(actors)
    inv = w.get_inventory(a)
    if inv and random.random() < 0.5:
        return tmpl.format(actor=a, item=random.choice(inv)), "yes"
    not_held = [i for i in w.pools["items"] if i not in inv]
    return tmpl.format(actor=a, item=random.choice(not_held)), "no"

def resolve_past_location(w, tmpl, actors, cfg):
    moves = [h for h in w.history if h["type"] == "move"]
    if moves:
        m = random.choice(moves)
        return tmpl.format(actor=m["actor"], location=m["to"]), m["from"]
    a = random.choice(actors)
    return f"where is {a}", w.actor_loc[a]

def resolve_agent_motivation(w, tmpl, actors, cfg):
    a = random.choice(actors)
    st = w.actor_state[a]
    if st and "why" in tmpl:
        return tmpl.format(actor=a, location=st["target"]), st["state"]
    need_state = random.choice(cfg["states"]["needs"])["state"]
    matched = [act for act, s in w.actor_state.items() if s and s["state"] == need_state]
    return tmpl.format(state=need_state, actor="", location=""), matched[0] if matched else "nobody"

QUERY_RESOLVERS = {
    "actor_location": resolve_actor_location,
    "item_location": resolve_item_location,
    "possession_list": resolve_possession_list,
    "possession_count": resolve_possession_count,
    "actor_location_polar": resolve_actor_location_polar,
    "possession_polar": resolve_possession_polar,
    "past_location": resolve_past_location,
    "agent_motivation": resolve_agent_motivation
}


def generate_atomic_sample(cfg, target_category=None):
    w = WorldState(cfg)
    clauses = []
    
    active_actors = random.sample(cfg["pools"]["actors"], 2)
    for a in active_actors:
        clauses.append(random.choice(cfg["initial_templates"]).format(actor=a, location=w.actor_loc[a]))

    action_keys = list(cfg["actions"].keys())
    target_events = random.randint(cfg["rules"]["min_events"], cfg["rules"]["max_events"])
    events_run, attempts = 0, 0
    while events_run < target_events and attempts < 30:
        attempts += 1
        action = random.choice(action_keys)
        narrated = ACTION_HANDLERS[action](w, cfg)
        if narrated:
            if events_run > 0 and random.random() < cfg["rules"]["transition_probability"]:
                narrated = f"{random.choice(cfg['connectors'])} {narrated}"
            clauses.append(narrated)
            events_run += 1

    premise = " . ".join(clauses)

    cat = target_category if target_category else random.choice(list(cfg["queries"].keys()))
    q_meta = cfg["queries"][cat]
    tmpl = random.choice(q_meta["templates"])
    q_str, a_str = QUERY_RESOLVERS[q_meta["resolver"]](w, tmpl, active_actors, cfg)

    return format_qa_sequence(premise, q_str, a_str)

def generate_benchmark_suite(cfg, samples_per_category=400):
    dataset = {}
    for cat in cfg["queries"].keys():
        dataset[cat] = []
        seen = set()
        attempts = 0
        while len(dataset[cat]) < samples_per_category and attempts < samples_per_category * 12:
            attempts += 1
            seq = generate_atomic_sample(cfg, target_category=cat)
            p, t = split_qa(seq)
            if p not in seen:
                seen.add(p)
                dataset[cat].append({"q": p, "target": t})
    return dataset