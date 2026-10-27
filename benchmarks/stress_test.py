import os, sys, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from curriculum import generate_benchmark_suite

with open("config/world.json", "r") as f: log_cfg = json.load(f)
dataset = generate_benchmark_suite(log_cfg, samples_per_category=400)
with open("config/stress_test.json", "w") as f: json.dump(dataset, f, indent=2)