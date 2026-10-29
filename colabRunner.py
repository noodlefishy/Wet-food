import os, sys, subprocess, shutil

print("=" * 65)
print("     picoAI UNIFIED COLAB BUILD & TEST RUNNER")
print("=" * 65)

import torch

inColab = False
try:
    import google.colab # pyright: ignore[reportMissingImports]
    inColab = True
except ImportError:
    pass

if not inColab:
    print("\n[!] ERROR: colabRunner.py must run inside a Google Colab GPU environment!")
    sys.exit(1)

if not torch.cuda.is_available():
    print("\n[!] WARNING: CUDA GPU not detected! Please switch runtime to T4 GPU.")
    sys.exit(1)

print(f"[✓] CUDA verified: {torch.cuda.get_device_name(0)}")
repoDir = "/content/Wet-food"
if not os.path.exists(repoDir) and not os.path.exists("config/model.json"):
    print("[*] Cloning repository...", flush=True)
    subprocess.run(["git", "clone", "https://github.com/noodlefishy/Wet-food.git", repoDir], check=True)
    os.chdir(repoDir)
elif os.path.exists(repoDir):
    os.chdir(repoDir)
    print("[*] Updating repository...", flush=True)
    subprocess.run(["git", "pull", "origin", "main"], check=False)

def runStreaming(cmd, envVars=None):
    e = os.environ.copy()
    e["PYTHONUNBUFFERED"] = "1"
    if envVars: e.update(envVars)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=e
    )
    if proc.stdout:
        for line in iter(proc.stdout.readline, ""):
            print(line, end="", flush=True)
        proc.stdout.close()
    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)

print("\n" + "=" * 65)
print(" STAGE 1: TRAINING & QUANTIZATION (training/trainModel.py)")
print("=" * 65, flush=True)
runStreaming(["python3", "-u", "training/trainModel.py"])

print("\n" + "=" * 65)
print(" STAGE 2: PACKAGING WEIGHTS")
print("=" * 65, flush=True)

zipTarget = "/content/pico_weights_latest.zip"
pkgDir = "/content/pico_pkg"
os.makedirs(pkgDir, exist_ok=True)
shutil.copy("models/local/pico1 weights.model", os.path.join(pkgDir, "pico1 weights.model"))
shutil.copy("models/local/pico2 weights.model", os.path.join(pkgDir, "pico2 weights.model"))
shutil.copy("models/local/vocab.py", os.path.join(pkgDir, "vocab.py"))
shutil.make_archive("/content/pico_weights_latest", "zip", pkgDir)
shutil.copy(zipTarget, "models/local/pico_weights_latest.zip")

print(f"[✓] Package created: {zipTarget} ({os.path.getsize(zipTarget)/1024:.1f} KB)")
print("\n" + "=" * 65)


print("\n" + "=" * 65)
print(" STAGE 3: GENERATING EXHAUSTIVE STRESS SUITE (100k+ TESTS)")
print("=" * 65, flush=True)
runStreaming(["python3", "-u", "benchmarks/stressTest.py"])

print("\n" + "=" * 65)
print(" STAGE 3: RUNNING FAST EVALUATION")
print("=" * 65, flush=True)
runStreaming(["python3", "-u", "benchmarks/fastEvaluator.py", "config/stressTest.json", "--failures"])

print(" STAGE 4: OUT-OF-DISTRIBUTION PROBES (benchmarks/oodEvaluator.py)")
print("=" * 65, flush=True)
runStreaming(["python3", "-u", "benchmarks/oodEvaluator.py"])

print("\n" + "=" * 65)
print(" STAGE 5: SIMULATOR SMOKE TEST")
print("=" * 65, flush=True)
simPy = """
from benchmarks.sim import generate
test_prompts = [
    "[c] alice went to the kitchen . bob went to the room . [q] where is alice ? [a]",
    "[c] alice has one cup . bob gave alice two cups . [q] how many items does alice have ? [a]",
    "once upon a time bob went into the green garden ."
]
for p in test_prompts:
    print(f"\\nPrompt: {p}\\nOutput: {generate(p, maxTokens=25)}")
"""
runStreaming(["python3", "-u", "-c", simPy])

print(" BUILD, TRAIN & EVALUATION PIPELINE COMPLETE!")
print("=" * 65)
