import json
import itertools
import re

# Load your template definition (the one from your previous prompt)
with open('config/logic.json', 'r') as f:
    config = json.load(f)

pools = config['pools']
pair_pools = config['pair_pools']
categories = config['categories']

def get_values(key):
    """Retrieves list of values or dicts from pools or pair_pools."""
    base_key = re.sub(r'\d+', '', key)
    if base_key in pools:
        return pools[base_key]
    if base_key in pair_pools:
        return pair_pools[base_key]
    return []

def generate_dataset():
    dataset = {}
    
    for category in categories:
        cat_name = category['name']
        dataset[cat_name] = []
        
        for template in category['templates']:
            # 1. Parse template to find placeholders (e.g., {actor1}, {need1_state})
            placeholders = re.findall(r'\{(\w+)\}', template)
            
            # 2. Group placeholders by their entity (e.g., actor1, need1)
            groups = {}
            for p in placeholders:
                # 'need1_state' -> 'need1'
                base = re.sub(r'_[a-z]+$', '', p)
                if base not in groups:
                    groups[base] = get_values(base)
            
            # 3. Create all combinations
            keys = list(groups.keys())
            for combination in itertools.product(*groups.values()):
                mapping = {}
                for i, key in enumerate(keys):
                    val = combination[i]
                    if isinstance(val, dict):
                        for sub_k, sub_v in val.items():
                            mapping[f"{key}_{sub_k}"] = sub_v
                    else:
                        mapping[key] = val
                
                # 4. Split template into context [c] and question [q] and answer [a]
                # Template format: [c] ... [q] ... [a] ... <eos>
                try:
                    parts = re.split(r'\[q\]|\[a\]', template.format(**mapping))
                    context_q = parts[0].strip() + " [q] " + parts[1].strip()
                    answer = parts[2].replace('<eos>', '').strip()
                    
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
questionsCount = 0
for questionCategory in data:
    l =  len(data[questionCategory])
    print(f'{questionCategory} - {l}')
    questionsCount += l
print(f"Successfully generated {questionsCount}")