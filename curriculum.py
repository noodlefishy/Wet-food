import json
import itertools
import random
import re

def normalize_text(text):
    t = text.strip()
    t = re.sub(r'\s*([?.!:,])\s*', r' \1 ', t)
    return re.sub(r'\s+', ' ', t).strip()

def format_qa_sequence(premise, question, answer):
    """
    Auto-formats raw text into strict engine tokens:
    [c] <premise> . [q] <question> ? [a] <answer> . <eos>
    """
    p = normalize_text(premise.strip().rstrip(". "))
    q = normalize_text(question.strip().rstrip("?.: "))
    a = normalize_text(answer.strip().rstrip(". "))

    return f"[c] {p} . [q] {q} ? [a] {a} . <eos>"

def expand_category_templates(category):
    """
    Expands Scenario Schema. Supports:
    - premise: str OR list[str] (Syntactic Premise Jitter)
    - q: str OR list[str]       (Syntactic Query Jitter)
    """
    raw_premise = category["premise"]
    premises = raw_premise if isinstance(raw_premise, list) else [raw_premise]
    
    templates = []
    
    for prem in premises:
        for item in category["queries"]:
            q_field = item["q"]
            queries = q_field if isinstance(q_field, list) else [q_field]
            for single_q in queries:
                templates.append(format_qa_sequence(prem, single_q, item["a"]))
                
    return templates

def get_template_placeholders(template):
    placeholders = re.findall(r'\{([a-z]+)(\d+)(?:_([a-z]+))?\}', template)
    groups = {}
    for p_type, p_idx, _ in placeholders:
        groups.setdefault(p_type, set()).add(p_idx)
    return placeholders, groups

def instantiate_random(template, pools, pair_pools):
    _, groups = get_template_placeholders(template)
    replacements = {}

    for p_type, indices in groups.items():
        if p_type in pools:
            sampled = random.sample(pools[p_type], len(indices))
            for idx, val in zip(sorted(list(indices)), sampled):
                replacements[f"{p_type}{idx}"] = val

    for p_type, indices in groups.items():
        if p_type in pair_pools:
            sampled = random.sample(pair_pools[p_type], len(indices))
            for idx, val_dict in zip(sorted(list(indices)), sampled):
                for sub_k, sub_v in val_dict.items():
                    replacements[f"{p_type}{idx}_{sub_k}"] = sub_v

    return template.format(**replacements)

def instantiate_all_permutations(template, pools, pair_pools):
    _, groups = get_template_placeholders(template)
    
    pool_perms = []
    entities_by_pool = []

    for p_type, indices in groups.items():
        vals = pools.get(p_type) or pair_pools.get(p_type)
        if not vals or len(indices) > len(vals):
            return []
        
        sorted_indices = sorted(list(indices))
        entities_by_pool.append((p_type, sorted_indices))
        perms = list(itertools.permutations(vals, len(indices)))
        pool_perms.append(perms)

    results = []
    for combo in itertools.product(*pool_perms):
        replacements = {}
        for (p_type, indices), assigned_vals in zip(entities_by_pool, combo):
            for idx, val in zip(indices, assigned_vals):
                if isinstance(val, dict):
                    for sub_k, sub_v in val.items():
                        replacements[f"{p_type}{idx}_{sub_k}"] = sub_v
                else:
                    replacements[f"{p_type}{idx}"] = val
        results.append(template.format(**replacements))
        
    return results

def split_qa(sequence_str):
    parts = sequence_str.split(" [a] ")
    prompt = parts[0].strip() + " [a]"
    target = parts[1].replace("<eos>", "").strip()
    return prompt, target