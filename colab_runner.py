"""
Unified Colab Runner for picoAI (Wet-food)
This single script runs on Google Colab to:
1. Environment setup and GPU detection
2. Train the 8-Layer RoPE Model (CUDA/MPS/CPU)
3. Quantize INT8 W8A32 weights for Meow (pico1) and Miaou (pico2)
4. Run Fast Evaluator (all 69,000+ tests across 24 categories)
5. Run Out-of-Distribution (OOD) Generalization Probes
6. Run Simulator test prompts
7. Package and offer automatic zip download of weights + firmware vocab
"""
import os, sys, subprocess, shutil

print("="*65)
print("     picoAI UNIFIED COLAB BUILD & TEST RUNNER")
print("="*65)

# Step 1: Detect Colab environment & install requirements
in_colab = False
try:
    import google.colab
    in_colab = True
    print("[✓] Running inside Google Colab environment.")
except ImportError:
    print("[*] Running in standalone local environment.")

if in_colab:
    repo_dir = "/content/Wet-food"
    if not os.path.exists(repo_dir) and not os.path.exists("config/model.json"):
        print("[*] Cloning repository...")
        subprocess.run(["git", "clone", "https://github.com/noodlefishy/Wet-food.git", repo_dir], check=True)
        os.chdir(repo_dir)
    elif os.path.exists(repo_dir):
        os.chdir(repo_dir)

print(f"[*] Working directory: {os.getcwd()}")

# Step 2: Run Training & Quantization
print("\n" + "="*65)
print(" STAGE 1: TRAINING & QUANTIZATION (training/local.py)")
print("="*65)
subprocess.run([sys.executable, "training/local.py"], check=True)

# Step 3: Run Fast Evaluator (Full 69k stress test)
print("\n" + "="*65)
print(" STAGE 2: FAST EVALUATION BENCHMARK (benchmarks/fast_evaluator.py)")
print("="*65)
subprocess.run([sys.executable, "benchmarks/fast_evaluator.py"], check=True)

# Step 4: Run Out-of-Distribution Generalization Probe
print("\n" + "="*65)
print(" STAGE 3: OUT-OF-DISTRIBUTION (OOD) GENERALIZATION")
print("="*65)
subprocess.run([sys.executable, "benchmarks/ood_evaluator.py"], check=True)

# Step 5: Run Simulator Smoke Test
print("\n" + "="*65)
print(" STAGE 4: SIMULATOR SMOKE TEST")
print("="*65)
test_prompts = [
    "[c] alice went to the kitchen . bob went to the garden . [q] where is alice ? [a]",
    "[c] alice went to the kitchen . bob went to the garden . [q] is alice in the kitchen ? [a]",
    "[c] bob had two book and gave two to alice . [q] how many book does bob have ? [a]"
]
sys.path.insert(0, os.getcwd())
from benchmarks.sim import generate

for tp in test_prompts:
    res = generate(tp)
    print(f"Q: {tp}\nA: {res}\n")

# Step 6: Package weights for download
print("="*65)
print(" STAGE 5: PACKAGING WEIGHTS & VOCAB")
print("="*65)
zip_filename = "pico_weights_latest"
files_to_pack = [
    ("models/local/pico1 weights.model", "pico1 weights.model"),
    ("models/local/pico2 weights.model", "pico2 weights.model"),
    ("models/local/vocab.py", "vocab.py"),
]

os.makedirs("export_pack", exist_ok=True)
for src, name in files_to_pack:
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join("export_pack", name))
        print(f"  [✓] Bundled {name} ({os.path.getsize(src)/1024:.1f} KB)")

shutil.make_archive(zip_filename, 'zip', "export_pack")
zip_path = f"{zip_filename}.zip"
print(f"\n[✓] Created {zip_path} ({os.path.getsize(zip_path)/1024:.1f} KB)")

if in_colab:
    try:
        from google.colab import files
        print("[*] Triggering automatic browser download of weights package...")
        files.download(zip_path)
    except Exception as e:
        print(f"[!] Could not trigger automatic download: {e}")
        print(f"    You can download {zip_path} from the left file sidebar.")

print("\n[✓] ALL BUILD & TEST STAGES COMPLETE!")
