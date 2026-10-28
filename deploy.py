import sys, os, subprocess

MEOW_FILES = [
    ("models/local/pico1 weights.model", "pico1 weights.model"),
    ("models/local/vocab.py", "vocab.py"),
    ("firmware/index.html", "index.html"),
    ("firmware/meow.py", "main.py")
]

MIAOU_FILES = [
    ("models/local/pico2 weights.model", "pico2 weights.model"),
    ("firmware/miaou.py", "main.py")
]

args = [a.lower() for a in sys.argv[1:]]
code_only = "--code" in args or "-c" in args
clean_args = [a for a in args if not a.startswith("-")]

node = clean_args[0] if len(clean_args) > 0 else None
if node not in ("meow", "miaou"):
    print("Usage: python3 deploy.py meow [--code] [port]")
    sys.exit(1)

port_arg = ["connect", clean_args[1]] if len(clean_args) > 1 else []
all_targets = MEOW_FILES if node == "meow" else MIAOU_FILES

# Filter to only .py / .html if --code is passed
targets = [t for t in all_targets if not t[0].endswith(".model")] if code_only else all_targets

print(f"[*] Deploying {'(CODE ONLY)' if code_only else '(FULL)'} to {node.upper()}...")
for src, dst in targets:
    if not os.path.exists(src):
        print(f"[!] Missing {src}")
        sys.exit(1)
    print(f"  -> Uploading {src} as :{dst}...")
    subprocess.run(["mpremote"] + port_arg + ["rm", f":{dst}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if subprocess.run(["mpremote"] + port_arg + ["cp", src, f":{dst}"]).returncode != 0:
        print(f"[!] Upload failed for {src}")
        sys.exit(1)

subprocess.run(["mpremote"] + port_arg + ["reset"])
print(f"[✓] {node.upper()} updated and rebooted!")