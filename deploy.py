import sys, os, subprocess, gzip

HTML_SRC = "firmware/index.html"
HTML_GZ = "firmware/index.html.gz"
if os.path.exists(HTML_SRC):
    with open(HTML_SRC, "rb") as fIn, gzip.open(HTML_GZ, "wb", 9) as fOut:
        fOut.writelines(fIn)

MEOW_FILES = [
    ("models/local/pico1 weights.model", "pico1 weights.model"),
    ("models/local/vocab.py", "vocab.py"),
    ("firmware/wifiConfig.py", "wifiConfig.py"),
    ("firmware/index.html.gz", "index.html.gz"),
    ("firmware/meow.py", "main.py")
]

MIAOU_FILES = [
    ("models/local/pico2 weights.model", "pico2 weights.model"),
    ("firmware/wifiConfig.py", "wifiConfig.py"),
    ("firmware/miaou.py", "main.py")
]

args = [a.lower() for a in sys.argv[1:]]
codeOnly = "--code" in args or "-c" in args
cleanArgs = [a for a in args if not a.startswith("-")]

node = cleanArgs[0] if len(cleanArgs) > 0 else None
if node not in ("meow", "miaou"):
    print("Usage: python3 deploy.py meow [--code] [port]")
    sys.exit(1)

portArg = ["connect", cleanArgs[1]] if len(cleanArgs) > 1 else []
allTargets = MEOW_FILES if node == "meow" else MIAOU_FILES
targets = [t for t in allTargets if not t[0].endswith(".model")] if codeOnly else allTargets

if node == "meow":
    subprocess.run(["mpremote"] + portArg + ["rm", ":index.html"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print(f"[*] Deploying {'(CODE ONLY)' if codeOnly else '(FULL)'} to {node.upper()}...")
for src, dst in targets:
    if not os.path.exists(src):
        print(f"[!] Missing {src}")
        sys.exit(1)
    localSize = os.path.getsize(src)
    print(f"  -> Uploading {src} ({localSize} bytes) as :{dst}...")
    subprocess.run(["mpremote"] + portArg + ["rm", f":{dst}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if subprocess.run(["mpremote"] + portArg + ["cp", src, f":{dst}"]).returncode != 0:
        print(f"[!] Upload failed for {src}")
        sys.exit(1)

print("\n[✓] Deployment Complete! Soft-rebooting board...")
subprocess.run(["mpremote"] + portArg + ["reset"])