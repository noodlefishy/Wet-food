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

node = sys.argv[1].lower().replace("--", "") if len(sys.argv) > 1 else None
if node not in ("meow", "miaou"):
    print("Usage: python3 deploy.py meow  (or: python3 deploy.py miaou) [port]")
    sys.exit(1)

port_arg = ["connect", sys.argv[2]] if len(sys.argv) > 2 else []
targets = MEOW_FILES if node == "meow" else MIAOU_FILES
print(f"[*] Cleaning old files on {node.upper()}...")
for _, dst in targets:
    subprocess.run(["mpremote"] + port_arg + ["rm", f":{dst}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print(f"[*] Uploading fresh build to {node.upper()}...")
for src, dst in targets:
    if not os.path.exists(src): print(f"[!] Error: missing {src}. Run local.py first!"); sys.exit(1)
    print(f"  -> Uploading {src} as :{dst}...")
    cmd = ["mpremote"] + port_arg + ["cp", src, f":{dst}"]
    if subprocess.run(cmd).returncode != 0: print(f"[!] Upload failed for {src}. Check USB connection."); sys.exit(1)

print(f"[+] Resetting {node.upper()} to boot main.py...")
subprocess.run(["mpremote"] + port_arg + ["reset"])
print(f"[✓] {node.upper()} deployed and running successfully!")