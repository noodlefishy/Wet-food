import json
import itertools
import re

print("=" * 65)
print("      GENERATING EXHAUSTIVE NON-COLLIDING STRESS DATASET      ")
print("=" * 65)


with open('config/logic.json', 'r') as f:
    config = json.load(f)

pools = config['pools']
pair_pools = config.get('pair_pools', {})
categories = config['categories']

def generate_dataset():
    dataset = {}
    
    for category in categories:
        cat_name = category['name']
        dataset[cat_name] = []
        seen_prompts = set()
        
        for template in category['templates']:
            # 1. Parse all placeholders (e.g., ['actor1', 'need1_state', 'actor2'])
            placeholders = re.findall(r'\{(\w+)\}', template)
            
            # 2. Group unique entities by their pool type
            # E.g., 'actor1' -> pool 'actor', 'need1_state' -> pool 'need'
            pool_entities = {}
            for p in placeholders:
                entity = re.sub(r'_[a-z]+$', '', p)   # 'need1_state' -> 'need1'
                pool_name = re.sub(r'\d+', '', entity) # 'need1' -> 'need'
                
                if pool_name not in pool_entities:
                    pool_entities[pool_name] = []
                if entity not in pool_entities[pool_name]:
                    pool_entities[pool_name].append(entity)

            # Ensure entities are sorted (e.g. actor1 before actor2)
            for p_name in pool_entities:
                pool_entities[p_name].sort()

            # 3. Generate distinct permutations per pool (NO REPLACEMENT)
            pool_perms = []
            valid_template = True

            for pool_name, entities in pool_entities.items():
                vals = pools.get(pool_name) or pair_pools.get(pool_name)
                if not vals or len(entities) > len(vals):
                    valid_template = False
                    break
                
                # Permutations guarantee actor1 != actor2 != actor3
                perms = list(itertools.permutations(vals, len(entities)))
                pool_perms.append((entities, perms))

            if not valid_template:
                continue

            entities_by_pool = [item[0] for item in pool_perms]
            perms_by_pool = [item[1] for item in pool_perms]

            for combo in itertools.product(*perms_by_pool):
                mapping = {}
                for entities, assigned_vals in zip(entities_by_pool, combo):
                    for entity, val in zip(entities, assigned_vals):
                        if isinstance(val, dict):
                            for sub_k, sub_v in val.items():
                                mapping[f"{entity}_{sub_k}"] = sub_v
                        else:
                            mapping[entity] = val

                try:
                    formatted = template.format(**mapping)
                    parts = re.split(r'\[q\]|\[a\]', formatted)
                    
                    # Ensure prompt ends cleanly with ' ? [a]'
                    context_q = parts[0].strip() + " [q] " + parts[1].strip()
                    if not context_q.endswith("[a]"):
                        context_q = context_q + " [a]"
                        
                    answer = parts[2].replace('<eos>', '').strip()
                    
                    if context_q not in seen_prompts:
                        seen_prompts.add(context_q)
                        dataset[cat_name].append({
                            "q": context_q,
                            "target": f"{answer}"
                        })
                except Exception:
                    continue

    return dataset

# Generate and save
data = generate_dataset()

with open('config/stress_test.json', 'w') as f:
    json.dump(data, f, indent=2)

total_count = 0
for cat, items in data.items():
    l = len(items)
    print(f"{cat:<34} : {l:5d} unique non-colliding tests")
    total_count += l

print("-" * 65)
print(f"Successfully generated {total_count} mathematically distinct tests!")
print("Saved to -> config/stress_test.json\n")