import sys, os, subprocess, time, gzip

# Pre-compress index.html automatically on Mac/PC before uploading
HTML_SRC = "firmware/index.html"
HTML_GZ = "firmware/index.html.gz"
if os.path.exists(HTML_SRC):
    with open(HTML_SRC, "rb") as f_in, gzip.open(HTML_GZ, "wb", 9) as f_out:
        f_out.writelines(f_in)
    orig_kb = os.path.getsize(HTML_SRC) / 1024
    gz_kb = os.path.getsize(HTML_GZ) / 1024
    print(f"[*] Compressed {HTML_SRC}: {orig_kb:.1f} KB -> {gz_kb:.1f} KB (Saved {orig_kb - gz_kb:.1f} KB)")

MEOW_FILES = [
    ("models/local/pico1 weights.model", "pico1 weights.model"),
    ("models/local/vocab.py", "vocab.py"),
    ("firmware/wifi_config.py", "wifi_config.py"),
    ("firmware/index.html.gz", "index.html.gz"),
    ("firmware/meow.py", "main.py"),
    ("firmware/catalysts.json", "catalysts.json"),
    ("firmware/cortex.json", "cortex.json")
]

MIAOU_FILES = [
    ("models/local/pico2 weights.model", "pico2 weights.model"),
    ("firmware/wifi_config.py", "wifi_config.py"),
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
targets = [t for t in all_targets if not t[0].endswith(".model")] if code_only else all_targets

# Clean up old raw uncompressed index.html from Pico flash if switching to .gz
if node == "meow":
    subprocess.run(["mpremote"] + port_arg + ["rm", ":index.html"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print(f"[*] Deploying {'(CODE ONLY)' if code_only else '(FULL)'} to {node.upper()}...")
for src, dst in targets:
    if not os.path.exists(src):
        print(f"[!] Missing {src}")
        sys.exit(1)
        
    local_size = os.path.getsize(src)
    print(f"  -> Uploading {src} ({local_size} bytes) as :{dst}...")
    
    subprocess.run(["mpremote"] + port_arg + ["rm", f":{dst}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if subprocess.run(["mpremote"] + port_arg + ["cp", src, f":{dst}"]).returncode != 0:
        print(f"[!] Upload failed for {src}")
        sys.exit(1)
        
    res = subprocess.run(["mpremote"] + port_arg + ["exec", f"import os; print(os.stat('{dst}')[6])"], capture_output=True, text=True)
    try:
        remote_size = int(res.stdout.strip())
        if local_size != remote_size:
            print(f"[!] Size mismatch on {dst}: {remote_size} on Pico vs {local_size} local")
            sys.exit(1)
        else:
            print(f"     [✓] Verified ({remote_size} bytes)")
    except Exception:
        print(f"     [?] Upload completed.")

print(f"[+] Resetting {node.upper()}...")
subprocess.run(["mpremote"] + port_arg + ["reset"])
print(f"[✓] {node.upper()} deployed successfully!")