import sys, os, subprocess, time

MEOW_FILES = [
    ("models/local/pico1 weights.model", "pico1 weights.model"),
    ("models/local/vocab.py", "vocab.py"),
    ("firmware/wifi_config.py", "wifi_config.py"),
    ("firmware/index.html", "index.html"),
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

print(f"[*] Deploying {'(CODE ONLY)' if code_only else '(FULL)'} to {node.upper()}...")
for src, dst in targets:
    if not os.path.exists(src):
        print(f"[!] Missing {src}")
        sys.exit(1)
        
    local_size = os.path.getsize(src)
    print(f"  -> Uploading {src} ({local_size} bytes) as :{dst}...")
    
    # Delete old file to prevent fragmentation
    subprocess.run(["mpremote"] + port_arg + ["rm", f":{dst}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Upload new file
    if subprocess.run(["mpremote"] + port_arg + ["cp", src, f":{dst}"]).returncode != 0:
        print(f"[!] Upload failed for {src}")
        sys.exit(1)
        
    # VERIFY BYTE COUNT ON DEVICE
    res = subprocess.run(["mpremote"] + port_arg + ["exec", f"import os; print(os.stat('{dst}')[6])"], capture_output=True, text=True)
    try:
        remote_size = int(res.stdout.strip())
        if local_size != remote_size:
            print(f"\n[!] VERIFICATION FAILED! {dst} is {remote_size} bytes on Pico, but should be {local_size} bytes.")
            print("[!] The Pico filesystem is full or the USB transfer dropped. You need to wipe the flash.")
            sys.exit(1)
        else:
            print(f"     [✓] Verified perfectly ({remote_size} bytes)")
    except Exception:
        print(f"     [?] Could not verify size, but upload completed.")

print(f"[+] Resetting {node.upper()}...")
subprocess.run(["mpremote"] + port_arg + ["reset"])

if node == "meow":
    print("[*] Waiting for Meow to connect to Wi-Fi...")
    time.sleep(3)
    try:
        res = subprocess.run(
            ["mpremote"] + port_arg + ["exec", "import network, time; w = network.WLAN(network.STA_IF); print(w.ifconfig()[0] if w.isconnected() else 'AP')"],
            capture_output=True, text=True, timeout=8
        )
        ip = res.stdout.strip()
        if ip and ip != "AP":
            print(f"\n[★] MEOW IS ONLINE ON YOUR HOME NETWORK:")
            print(f"    -> http://{ip}/")
            print(f"    -> http://meow/ (if supported by your router)\n")
    except Exception as e:
        print(e)


print(f"[✓] {node.upper()} deployed successfully!")