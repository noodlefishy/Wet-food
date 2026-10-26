import json
import os
import re
import random
from evaluator import run_model_end_to_end

print("=" * 76)
print("             RUNNING ADVERSARIAL STRESS TEST (UNSEEN DATA)            ")
print("=" * 76)

actors = ["alice", "bob", "meow", "miaou"]
locations = ["garden", "kitchen", "park", "room"]
items = ["milk", "fish", "ball", "box"]
colours = ["red", "blue", "yellow", "green", "pink", "white"]

tests = []

# 1. Test Inverted Clause Ordering for multi-colour binding
for _ in range(10):
    a1, a2 = random.sample(actors, 2)
    c1, c2 = random.sample(colours, 2)
    i1, i2 = random.sample(items, 2)
    
    # Query first item
    tests.append({
        "type": "Standard",
        "q": f"[c] {a1} has a {c1} {i1} . {a2} has a {c2} {i2} . [q] what colour is the {i1} ? [a]",
        "target": f"{c1} ."
    })
    # Query second item
    tests.append({
        "type": "Standard",
        "q": f"[c] {a1} has a {c1} {i1} . {a2} has a {c2} {i2} . [q] what colour is the {i2} ? [a]",
        "target": f"{c2} ."
    })
    # Adversarial: Missing period after clause 2
    tests.append({
        "type": "Missing Period (OOD)",
        "q": f"[c] {a1} has a {c1} {i1} . {a2} has a {c2} {i2} [q] what colour is the {i1} ? [a]",
        "target": f"{c1} ."
    })

passed_clean = 0
total_clean = 0
passed_noisy = 0
total_noisy = 0

for idx, t in enumerate(tests, 1):
    out, candidates = run_model_end_to_end(t["q"], max_tokens=10, greedy=True)
    clean_out = out.strip().lower()
    clean_target = t["target"].strip().lower()
    
    is_clean = t["type"] == "Standard"
    matched = (clean_out == clean_target)
    
    if is_clean:
        total_clean += 1
        if matched: passed_clean += 1
    else:
        total_noisy += 1
        if matched: passed_noisy += 1
        
    status = "PASS" if matched else "FAIL"
    print(f"[{idx:02d}] {status} | {t['type']:<22} | Expected: {clean_target:<10} | Got: {clean_out}")

print("-" * 76)
print(f"CLEAN SYNTAX ACCURACY : {passed_clean}/{total_clean} ({(passed_clean/total_clean)*100:.1f}%)")
print(f"NOISY SYNTAX ACCURACY : {passed_noisy}/{total_noisy} ({(passed_noisy/total_noisy)*100:.1f}%)")
print("=" * 76)