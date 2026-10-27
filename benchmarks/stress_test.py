import os
import sys
import json

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from curriculum import expand_composition_templates, instantiate_all_permutations, split_qa

print("=" * 65)
print("      GENERATING EXHAUSTIVE NON-COLLIDING STRESS DATASET      ")
print("=" * 65)

with open("config/logic.json", "r") as f:
    log_cfg = json.load(f)

pools = log_cfg["pools"]
pair_pools = log_cfg.get("pair_pools", {})
clauses = log_cfg["clauses"]
compositions = log_cfg["compositions"]

dataset = {}
total_count = 0

for comp in compositions:
    comp_name = comp["name"]
    dataset[comp_name] = []
    seen = set()
    
    # 1. Expand templates via shared engine
    templates = expand_composition_templates(comp, clauses)
    
    for tmpl in templates:
        # 2. Get all distinct permutations without replacement
        permutations = instantiate_all_permutations(tmpl, pools, pair_pools)
        
        for seq in permutations:
            prompt, target = split_qa(seq)
            if prompt not in seen:
                seen.add(prompt)
                dataset[comp_name].append({"q": prompt, "target": target})

    count = len(dataset[comp_name])
    print(f"{comp_name:<34} : {count:5d} unique tests")
    total_count += count

output_file = "config/stress_test.json"
with open(output_file, "w") as f:
    json.dump(dataset, f, indent=2)

print("-" * 65)
print(f"Successfully generated {total_count} mathematically distinct tests!")
print(f"Saved to -> {output_file}\n")