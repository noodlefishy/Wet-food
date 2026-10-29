# meow.py - Host (Master) node for distributed transformer
import network, socket, struct, math, gc, time, machine, random, _thread, ure as re # pyright: ignore[reportMissingImports]
from array import array; from machine import Pin # pyright: ignore[reportMissingImports]
machine.freq(240_000_000)
led = Pin("LED", Pin.OUT); led.off()

EMBED_DIM = 96; BLOCK_SIZE = 64; NUM_HEADS = 4; HEAD_DIM = EMBED_DIM // NUM_HEADS; VOCAB_SIZE = 1024; BUF = bytearray(320 * 4)
ROW_B = 4 + EMBED_DIM; L0_OFF = VOCAB_SIZE * ROW_B
SINK_SIZE = 4; WINDOW_SIZE = 32; LAYERS = 3

try: import vocab; VOCAB = vocab.VOCAB
except: VOCAB = ["<unk>", "<pad>", "<eos>"] + ["w" + str(i) for i in range(1021)]
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}

try: import wifiConfig; WIFI_SSID = wifiConfig.SSID; WIFI_PASS = wifiConfig.PASSWORD
except: WIFI_SSID = ""; WIFI_PASS = ""

def read_vec(f, c): return struct.unpack(f'{c}f', f.read(c * 4))

def read_emb(f, idx):
    f.seek(idx * ROW_B)
    s = struct.unpack('f', f.read(4))[0]
    row = struct.unpack(f'{EMBED_DIM}b', f.read(EMBED_DIM))
    return [v * s for v in row]

def unquote(s):
    res = s.replace('+', ' ')
    parts = res.split('%')
    if len(parts) == 1: return res
    out = [parts[0]]
    for p in parts[1:]:
        if len(p) >= 2:
            try: out.append(chr(int(p[:2], 16)) + p[2:])
            except: out.append('%' + p)
        else: out.append('%' + p)
    return "".join(out)

def recv_all(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet: return None
        data.extend(packet)
    return data

@micropython.native
def stream_linear(f, r, c, x):
    out, bv = [0.0]*r, memoryview(BUF)[:c]
    for i in range(r):
        scale = struct.unpack('f', f.read(4))[0]
        f.readinto(bv); row = struct.unpack(f'{c}b', bv)
        out[i] = sum(row[j]*x[j] for j in range(c)) * scale
    bi = 0
    while bi < r:
        s = min(64, r - bi); f.readinto(memoryview(BUF)[:s*4]); b = struct.unpack(f'{s}f', memoryview(BUF)[:s*4])
        for i in range(s): out[bi+i] += b[i]
        bi += s
    return out

def layer_norm(x, w, b):
    m = sum(x)/len(x); v = sum((i-m)**2 for i in x)/len(x); s = math.sqrt(v+1e-5)
    return [((x[i]-m)/s)*w[i]+b[i] for i in range(len(x))]

# Precompute RoPE table for positions 0..63
inv_freq = [1.0 / (10000.0 ** (2.0 * i / HEAD_DIM)) for i in range(HEAD_DIM // 2)]
cos_tab = [[math.cos(p * inv_freq[i]) for i in range(HEAD_DIM // 2)] for p in range(BLOCK_SIZE)]
sin_tab = [[math.sin(p * inv_freq[i]) for i in range(HEAD_DIM // 2)] for p in range(BLOCK_SIZE)]

def apply_rope(vec, pos):
    c, s = cos_tab[min(pos, BLOCK_SIZE - 1)], sin_tab[min(pos, BLOCK_SIZE - 1)]
    for i in range(10):
        i2 = i * 2
        x0, x1 = vec[i2], vec[i2 + 1]
        vec[i2] = x0 * c[i] - x1 * s[i]
        vec[i2 + 1] = x0 * s[i] + x1 * c[i]

def sample_top5(ranks):
    r = random.random()
    if r < 0.52: return ranks[0]
    elif r < 0.76: return ranks[1]
    elif r < 0.88: return ranks[2]
    elif r < 0.95: return ranks[3]
    return ranks[4]

f_w = open("pico1 weights.model", "rb")
kv = [{'k':[], 'v':[]} for _ in range(LAYERS)]  # 3 Layers on Meow

def run_layer(f, idx, x, pos):
    nx = layer_norm(x, read_vec(f, EMBED_DIM), read_vec(f, EMBED_DIM))
    qkv = stream_linear(f, EMBED_DIM + 2*HEAD_DIM, EMBED_DIM, nx)
    q = qkv[:EMBED_DIM]
    k = qkv[EMBED_DIM : EMBED_DIM + HEAD_DIM]
    v = qkv[EMBED_DIM + HEAD_DIM :]
    
    # RoPE on Query heads and Key
    c_tab, s_tab = cos_tab[min(pos, BLOCK_SIZE-1)], sin_tab[min(pos, BLOCK_SIZE-1)]
    for h in range(NUM_HEADS):
        ho = h * HEAD_DIM
        for i in range(HEAD_DIM // 2):
            i2 = ho + i * 2
            x0, x1 = q[i2], q[i2+1]
            q[i2]   = x0 * c_tab[i] - x1 * s_tab[i]
            q[i2+1] = x0 * s_tab[i] + x1 * c_tab[i]
    apply_rope(k, pos)

    kv[idx]['k'].append(array('f', k))
    kv[idx]['v'].append(array('f', v))
    
    # Sliding window KV cache with Attention Sinks
    if len(kv[idx]['k']) > SINK_SIZE + WINDOW_SIZE:
        kv[idx]['k'].pop(SINK_SIZE)
        kv[idx]['v'].pop(SINK_SIZE)
        
    ao, sc = [0.0]*EMBED_DIM, 1.0/math.sqrt(HEAD_DIM)
    seq_len = len(kv[idx]['k'])
    avg_attn = [0.0] * seq_len
    for h in range(NUM_HEADS):
        ho = h * HEAD_DIM; qh = q[ho : ho + HEAD_DIM]
        scs = [sum(qh[i]*kv[idx]['k'][t][i] for i in range(HEAD_DIM))*sc for t in range(seq_len)]
        ms = max(scs); wts = [math.exp(s-ms) for s in scs]; s_wts = sum(wts); wts = [e/s_wts for e in wts]
        for t in range(seq_len):
            avg_attn[t] += wts[t] / NUM_HEADS
            for i in range(HEAD_DIM): ao[ho+i] += wts[t] * kv[idx]['v'][t][i]
            
    proj = stream_linear(f, EMBED_DIM, EMBED_DIM, ao)
    x = [x[i] + proj[i] for i in range(EMBED_DIM)]
    nx = layer_norm(x, read_vec(f, EMBED_DIM), read_vec(f, EMBED_DIM))
    h1 = stream_linear(f, 4*EMBED_DIM, EMBED_DIM, nx)
    h2 = stream_linear(f, EMBED_DIM, 4*EMBED_DIM, [max(0.0, val) for val in h1])
    return [x[i] + h2[i] for i in range(EMBED_DIM)], avg_attn

print("\n" + "="*50)
print("[MEOW] Initializing Dual-Core Engine (3L RoPE + Attention Telemetry)...")

try: network.hostname('meow')
except: pass

sta = network.WLAN(network.STA_IF)
sta.active(True)
if WIFI_SSID:
    sta.connect(WIFI_SSID, WIFI_PASS)
    t_end = time.time() + 12
    while not sta.isconnected() and time.time() < t_end:
        led.toggle(); time.sleep(0.2)

if sta.isconnected():
    my_ip = sta.ifconfig()[0]
    print(f"[MEOW] Connected to Home Wi-Fi! IP: {my_ip}")
else:
    ap = network.WLAN(network.AP_IF); ap.active(True); ap.config(ssid='Meow-Net', security=0)
    my_ip = ap.ifconfig()[0]

led.off()

ai_sock = socket.socket(); ai_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
ai_sock.bind(('0.0.0.0', 5000)); ai_sock.listen(1)

web_sock = socket.socket(); web_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
web_sock.bind(('0.0.0.0', 80)); web_sock.listen(2)

beacon_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
beacon_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# =====================================================================
# DUAL-CORE INTER-CORE FIFO PIPELINE
# =====================================================================
job_queue = []          # [prompt_string]
telemetry_fifo = []     # [json_line_string, ...]
client_active = [False]

def core1_http_daemon():
    """Core 1 handles Web HTTP server & async streaming without halting Core 0 math."""
    while True:
        try:
            cl, _ = web_sock.accept()
            cl.settimeout(4.0)
            req = cl.recv(1024).decode('utf-8', 'ignore')
            
            if "GET /gen?p=" in req:
                p_raw = req.split("p=")[1].split(" ")[0].split("&")[0]
                prompt = unquote(p_raw).lower()
                cl.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n")
                
                # Signal Core 0 to ingest job
                client_active[0] = True
                telemetry_fifo.clear()
                job_queue.append(prompt)
                
                # Stream telemetry lines from Core 0 to browser
                while client_active[0]:
                    if telemetry_fifo:
                        line = telemetry_fifo.pop(0)
                        if line == "DONE": break
                        cl.sendall(line.encode())
                    else:
                        time.sleep_ms(6)
                cl.close()
            elif "GET / " in req or "GET /index.html" in req:
                try:
                    with open("index.html.gz", "rb") as f_gz:
                        cl.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Encoding: gzip\r\nConnection: close\r\n\r\n")
                        while True:
                            b = f_gz.read(1024)
                            if not b: break
                            cl.sendall(b)
                except:
                    cl.sendall(b"HTTP/1.1 404 Not Found\r\n\r\nindex.html missing")
                cl.close()
            else:
                cl.sendall(b"HTTP/1.1 404 Not Found\r\n\r\n")
                cl.close()
        except:
            client_active[0] = False
            time.sleep_ms(20)

try:
    _thread.start_new_thread(core1_http_daemon, ())
    print("[MEOW] Core 1 HTTP daemon initialized successfully.")
except Exception as e:
    print("[MEOW] Core 1 fallback warning:", e)

# =====================================================================
# CORE 0: NEURAL INFERENCE ENGINE & MIAOU LINK
# =====================================================================
pb = None
last_beacon = 0

while True:
    if pb is None:
        if time.time() - last_beacon >= 2:
            try: beacon_sock.sendto(b"MEOW_BEACON", ('255.255.255.255', 5001))
            except: pass
            last_beacon = time.time()
            
        ai_sock.settimeout(0.5)
        try:
            pb, addr = ai_sock.accept()
            pb.settimeout(10.0)
            print(f"[MEOW] -> Miaou paired from {addr}! System Ready.")
            led.on(); time.sleep(0.1); led.off()
        except: pass
        continue

    # Core 0: check if Core 1 queued a prompt
    if job_queue:
        if pb is None:
            telemetry_fifo.append('{"stage":"error","msg":"Miaou worker not connected"}\n')
            telemetry_fifo.append("DONE")
            job_queue.pop(0)
            continue

        p = job_queue.pop(0)
        CHUNKS = [
            ("once upon a time", "once_upon_a_time"),
            ("who went to the",  "who_went_to_the"),
            ("who is in the",    "who_is_in_the"),
            ("went to the",      "went_to_the"),
            ("walked to the",    "walked_to_the"),
            ("is in the",        "is_in_the"),
            ("was in the",       "was_in_the"),
            ("does not have",    "does_not_have"),
            ("where is",         "where_is"),
            ("how many",         "how_many"),
            ("one day",          "one_day")
        ]
        p_clean = p.lower()
        for phrase, chunk_tok in CHUNKS:
            p_clean = p_clean.replace(phrase, f" {chunk_tok} ")
        for char in [".", "?", "!", ",", ":"]:
            p_clean = p_clean.replace(char, f" {char} ")
        p_clean = p_clean.replace("'s", " 's ")

        is_logic = any(t in p_clean for t in ("[c]", "[q]", "[a]")) or any(p_clean.strip().startswith(w) for w in ("who", "what", "where", "is", "does", "why", "how", "question"))
        if is_logic:
            if "[c]" not in p_clean and not p_clean.startswith("question"): p_clean = f"[c] {p_clean}"
            if "[q]" in p_clean and not p_clean.endswith("[a]"): p_clean = p_clean.rstrip("?.: ") + " ? [a]"
            
        tks = [WORD_TO_ID.get(w, 0) for w in p_clean.split() if w]
        if not tks:
            continue

        for c_kv in kv: c_kv['k'].clear(); c_kv['v'].clear()
        
        # Reset signal to Miaou
        try:
            pb.sendall(struct.pack(f'{EMBED_DIM}f', *([-999.0] + [0.0]*(EMBED_DIM-1))))
            ack = recv_all(pb, 2)
            if ack != b"OK":
                pb = None
                telemetry_fifo.append('{"stage":"error","msg":"Miaou reset failed"}\n')
                telemetry_fifo.append("DONE")
                continue
        except Exception:
            pb = None
            telemetry_fifo.append('{"stage":"error","msg":"Miaou reset socket error"}\n')
            telemetry_fifo.append("DONE")
            continue
        
        t_start = time.ticks_ms()
        cur = tks[0]
        ct_ranks = [0, 0, 0, 0, 0]
        aborted = False
        
        # Phase 1: Prefill prompt tokens into KV cache with RoPE & Attention Capture
        for stp in range(len(tks)):
            if not client_active[0]: aborted = True; break
            led.on()
            tid = tks[stp]
            x = read_emb(f_w, tid)
            
            last_attn = []
            f_w.seek(L0_OFF)
            for l in range(LAYERS):
                x, last_attn = run_layer(f_w, l, x, stp)
            
            try:
                pb.sendall(struct.pack(f'{EMBED_DIM}f', *x))
                rep = recv_all(pb, 10)
                if not rep:
                    pb = None
                    aborted = True
                    telemetry_fifo.append('{"stage":"error","msg":"Miaou worker dropped"}\n')
                    break
            except Exception:
                pb = None
                aborted = True
                telemetry_fifo.append('{"stage":"error","msg":"Miaou link lost"}\n')
                break
            
            ct_ranks = struct.unpack('5H', rep)
            cur = ct_ranks[0] if is_logic else sample_top5(ct_ranks)
            
            lat = time.ticks_diff(time.ticks_ms(), t_start); t_start = time.ticks_ms()
            ranks = [VOCAB[t] for t in ct_ranks]
            ranks_json = f'["{ranks[0]}","{ranks[1]}","{ranks[2]}","{ranks[3]}","{ranks[4]}"]'
            v_str = "[" + ",".join([f"{v:.2f}" for v in x]) + "]"
            attn_str = "[" + ",".join([f"{a:.2f}" for a in last_attn]) + "]"
            
            telemetry_fifo.append(f'{{"stage":"prefill","cur":{stp+1},"tot":{len(tks)},"tok":"{VOCAB[tid]}","ms":{lat},"ram":{gc.mem_free()},"v":{v_str},"ranks":{ranks_json},"attn":{attn_str}}}\n')
            led.off()

        # Phase 2: Generation loop
        if not aborted:
            for gen_stp in range(50):
                if not client_active[0]: break
                led.on()
                word = VOCAB[cur]
                lat = time.ticks_diff(time.ticks_ms(), t_start); t_start = time.ticks_ms()
                v_str = "[" + ",".join([f"{v:.2f}" for v in x]) + "]"
                ranks = [VOCAB[t] for t in ct_ranks]
                ranks_json = f'["{ranks[0]}","{ranks[1]}","{ranks[2]}","{ranks[3]}","{ranks[4]}"]'
                attn_str = "[" + ",".join([f"{a:.2f}" for a in last_attn]) + "]"

                word_display = VOCAB[cur].replace("_", " ")
                telemetry_fifo.append(f'{{"stage":"gen","w":"{word}","ms":{lat},"ram":{gc.mem_free()},"v":{v_str},"ranks":{ranks_json},"attn":{attn_str}}}\n')
                
                if (is_logic and word == ".") or word in ("<eos>", "<pad>"): led.off(); break
                
                pos = len(tks) + gen_stp
                tid = cur
                x = read_emb(f_w, tid)
                
                f_w.seek(L0_OFF)
                for l in range(3):
                    x, last_attn = run_layer(f_w, l, x, pos)
                
                try:
                    pb.sendall(struct.pack(f'{EMBED_DIM}f', *x))
                    rep = recv_all(pb, 10)
                    if not rep: break
                except Exception: break
                
                ct_ranks = struct.unpack('5H', rep)
                cur = ct_ranks[0] if is_logic else sample_top5(ct_ranks)
                led.off()

        telemetry_fifo.append("DONE")
        gc.collect()
    else:
        time.sleep_ms(10)