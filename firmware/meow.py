# meow.py - Host (Master) node for distributed transformer (96-D / 4-Head MHA)
import network, socket, struct, math, gc, time, machine, random, _thread, ure as re
from array import array; from machine import Pin
machine.freq(240_000_000)
led = Pin("LED", Pin.OUT); led.off()

EMBED_DIM = 96
BLOCK_SIZE = 64
NUM_HEADS = 4
HEAD_DIM = EMBED_DIM // NUM_HEADS  # Exactly 24
VOCAB_SIZE = 1024
ROW_B = 4 + EMBED_DIM
L0_OFF = VOCAB_SIZE * ROW_B
SINK_SIZE = 2
WINDOW_SIZE = 10                   # 12 tokens max in RAM (27.6 KB, safely below 36.5 KB heap)
LAYERS = 3

try: import vocab; VOCAB = vocab.VOCAB
except: VOCAB = ["<unk>", "<pad>", "<eos>"] + ["w" + str(i) for i in range(1021)]
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}

try: import wifiConfig; WIFI_SSID = wifiConfig.SSID; WIFI_PASS = wifiConfig.PASSWORD
except: WIFI_SSID = ""; WIFI_PASS = ""

# =====================================================================
# ZERO-ALLOCATION STATIC BUFFERS
# =====================================================================
BUF_nx    = [0.0] * EMBED_DIM
BUF_qkv   = [0.0] * (3 * EMBED_DIM)
BUF_ao    = [0.0] * EMBED_DIM
BUF_proj  = [0.0] * EMBED_DIM
BUF_mlp1  = [0.0] * (4 * EMBED_DIM)
BUF_mlp2  = [0.0] * EMBED_DIM
BUF_ln_w  = [0.0] * EMBED_DIM
BUF_ln_b  = [0.0] * EMBED_DIM
BUF_attn  = [0.0] * BLOCK_SIZE
ROW_EMB   = bytearray(EMBED_DIM)
BUF_LIN   = bytearray(4 * EMBED_DIM * 4)

def read_emb(f, idx, out):
    f.seek(idx * ROW_B)
    s = struct.unpack('f', f.read(4))[0]
    f.readinto(ROW_EMB)
    for i in range(EMBED_DIM):
        v = ROW_EMB[i]
        out[i] = (v if v < 128 else v - 256) * s

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
def stream_linear_into(f, r, c, x, out):
    bv = memoryview(BUF_LIN)[:c]
    for i in range(r):
        scale = struct.unpack('f', f.read(4))[0]
        f.readinto(bv)
        s = 0.0
        for j in range(c):
            v = bv[j]
            s += (v if v < 128 else v - 256) * x[j]
        out[i] = s * scale
    bi = 0
    mv_b = memoryview(BUF_LIN)
    while bi < r:
        s = min(64, r - bi)
        f.readinto(mv_b[:s * 4])
        b = struct.unpack(f'{s}f', mv_b[:s * 4])
        for i in range(s):
            out[bi + i] += b[i]
        bi += s

def layer_norm_into(x, w, b, out):
    n = len(x)
    m = sum(x) / n
    v = sum((x[i] - m) ** 2 for i in range(n)) / n
    s = math.sqrt(v + 1e-5)
    for i in range(n):
        out[i] = ((x[i] - m) / s) * w[i] + b[i]

def read_vec_into(f, c, target):
    raw = f.read(c * 4)
    vals = struct.unpack(f'{c}f', raw)
    for i in range(c):
        target[i] = vals[i]

half_head = HEAD_DIM // 2
inv_freq = [1.0 / (10000.0 ** (2.0 * i / HEAD_DIM)) for i in range(half_head)]
cos_tab = [[math.cos(p * inv_freq[i]) for i in range(half_head)] for p in range(BLOCK_SIZE)]
sin_tab = [[math.sin(p * inv_freq[i]) for i in range(half_head)] for p in range(BLOCK_SIZE)]

def sample_top5(ranks):
    r = random.random()
    if r < 0.52: return ranks[0]
    elif r < 0.76: return ranks[1]
    elif r < 0.88: return ranks[2]
    elif r < 0.95: return ranks[3]
    return ranks[4]

f_w = open("pico1 weights.model", "rb")
kv = [{'k':[], 'v':[]} for _ in range(LAYERS)]
cur_x = [0.0] * EMBED_DIM

def run_layer(f, idx, x, pos):
    read_vec_into(f, EMBED_DIM, BUF_ln_w)
    read_vec_into(f, EMBED_DIM, BUF_ln_b)
    layer_norm_into(x, BUF_ln_w, BUF_ln_b, BUF_nx)

    stream_linear_into(f, 3 * EMBED_DIM, EMBED_DIM, BUF_nx, BUF_qkv)
    
    c_tab = cos_tab[min(pos, BLOCK_SIZE - 1)]
    s_tab = sin_tab[min(pos, BLOCK_SIZE - 1)]
    for h in range(NUM_HEADS):
        ho = h * HEAD_DIM
        for i in range(half_head):
            i2 = ho + i * 2
            q0, q1 = BUF_qkv[i2], BUF_qkv[i2 + 1]
            BUF_qkv[i2]     = q0 * c_tab[i] - q1 * s_tab[i]
            BUF_qkv[i2 + 1] = q0 * s_tab[i] + q1 * c_tab[i]
            k0, k1 = BUF_qkv[EMBED_DIM + i2], BUF_qkv[EMBED_DIM + i2 + 1]
            BUF_qkv[EMBED_DIM + i2]     = k0 * c_tab[i] - k1 * s_tab[i]
            BUF_qkv[EMBED_DIM + i2 + 1] = k0 * s_tab[i] + k1 * c_tab[i]

    kv[idx]['k'].append(array('f', BUF_qkv[EMBED_DIM : 2 * EMBED_DIM]))
    kv[idx]['v'].append(array('f', BUF_qkv[2 * EMBED_DIM : 3 * EMBED_DIM]))
    
    if len(kv[idx]['k']) > SINK_SIZE + WINDOW_SIZE:
        kv[idx]['k'].pop(SINK_SIZE)
        kv[idx]['v'].pop(SINK_SIZE)
        
    for i in range(EMBED_DIM): BUF_ao[i] = 0.0
    seq_len = len(kv[idx]['k'])
    for t in range(seq_len): BUF_attn[t] = 0.0
    sc = 1.0 / math.sqrt(HEAD_DIM)
    
    for h in range(NUM_HEADS):
        ho = h * HEAD_DIM
        ms = -1e9
        for t in range(seq_len):
            k_t = kv[idx]['k'][t]
            dot = 0.0
            for i in range(HEAD_DIM):
                dot += BUF_qkv[ho + i] * k_t[ho + i]
            score = dot * sc
            BUF_mlp1[t] = score
            if score > ms: ms = score
            
        sum_exp = 0.0
        for t in range(seq_len):
            ew = math.exp(BUF_mlp1[t] - ms)
            BUF_mlp1[t] = ew
            sum_exp += ew
            
        inv_sum = 1.0 / (sum_exp if sum_exp > 1e-12 else 1.0)
        for t in range(seq_len):
            w = BUF_mlp1[t] * inv_sum
            BUF_attn[t] += w / NUM_HEADS
            v_t = kv[idx]['v'][t]
            for i in range(HEAD_DIM):
                BUF_ao[ho + i] += w * v_t[ho + i]
            
    stream_linear_into(f, EMBED_DIM, EMBED_DIM, BUF_ao, BUF_proj)
    for i in range(EMBED_DIM): x[i] += BUF_proj[i]
    
    read_vec_into(f, EMBED_DIM, BUF_ln_w)
    read_vec_into(f, EMBED_DIM, BUF_ln_b)
    layer_norm_into(x, BUF_ln_w, BUF_ln_b, BUF_nx)
    
    stream_linear_into(f, 4 * EMBED_DIM, EMBED_DIM, BUF_nx, BUF_mlp1)
    for j in range(4 * EMBED_DIM):
        if BUF_mlp1[j] < 0.0: BUF_mlp1[j] = 0.0
        
    stream_linear_into(f, EMBED_DIM, 4 * EMBED_DIM, BUF_mlp1, BUF_mlp2)
    for i in range(EMBED_DIM): x[i] += BUF_mlp2[i]
    
    return BUF_attn[:seq_len]

def canonicalize_to_qca(raw):
    p = raw.strip()
    is_logic = any(t in p for t in ("[c]", "[q]", "[a]")) or any(
        p.lower().startswith(w) for w in ("who", "what", "where", "is", "does", "why", "how", "question")
    ) or ("?" in p)
    
    if not is_logic:
        return p, False

    if "[q]" in p and "[c]" in p:
        q_pos = p.find("[q]")
        c_pos = p.find("[c]")
        if q_pos < c_pos:
            if not p.endswith("[a]"): p = p.rstrip("?.: ") + " [a]"
            return p, True
        else:
            c_part = p[c_pos + 3 : q_pos].strip().rstrip(". ")
            rem = p[q_pos + 3 :].strip()
            q_part = rem.split("[a]")[0].strip().rstrip("?.: ")
            return f"[q] {q_part} ? [c] {c_part} . [a]", True

    if "?" in p:
        parts = p.split("?")
        q_words = ("who", "what", "where", "is", "are", "does", "did", "why", "how")
        first = parts[0].strip()
        second = parts[1].replace("[a]", "").strip().rstrip(". ")
        if any(first.lower().startswith(w) for w in q_words):
            if second: return f"[q] {first} ? [c] {second} . [a]", True
            else: return f"[q] {first} ? [a]", True
        else:
            return f"[q] {second} ? [c] {first} . [a]", True

    if not p.startswith("[q]"): p = f"[q] {p}"
    if not p.endswith("[a]"): p = p.rstrip("?.: ") + " ? [a]"
    return p, True

print("\n" + "="*50)
print(f"[MEOW] Initializing 96-D / 4-Head Master Engine...")

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

job_queue = []
telemetry_fifo = []
client_active = [False]

def send_chunk(cl, payload_bytes):
    cl.sendall(f"{len(payload_bytes):X}\r\n".encode() + payload_bytes + b"\r\n")

def core1_http_daemon():
    while True:
        try:
            cl, _ = web_sock.accept()
            cl.settimeout(4.0)
            req = cl.recv(1024).decode('utf-8', 'ignore')
            
            if "GET /gen?p=" in req:
                p_raw = req.split("p=")[1].split(" ")[0].split("&")[0]
                user_prompt = unquote(p_raw)
                qca_prompt, is_logic = canonicalize_to_qca(user_prompt)
                
                cl.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: application/x-ndjson\r\n"
                    b"Transfer-Encoding: chunked\r\n"
                    b"Cache-Control: no-cache, no-transform\r\n"
                    b"X-Content-Type-Options: nosniff\r\n"
                    b"Connection: close\r\n\r\n"
                )
                
                client_active[0] = True
                telemetry_fifo.clear()
                job_queue.append((user_prompt, qca_prompt, is_logic))
                
                while client_active[0]:
                    if telemetry_fifo:
                        line = telemetry_fifo.pop(0)
                        if line == "DONE": break
                        send_chunk(cl, line.encode())
                    else:
                        time.sleep_ms(6)
                send_chunk(cl, b"")
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
    print("[MEOW] Core 1 HTTP daemon initialized.")
except Exception as e:
    print("[MEOW] Core 1 fallback warning:", e)

# =====================================================================
# CORE 0: INFERENCE LOOP
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
            pb.settimeout(12.0)
            print(f"[MEOW] -> Miaou paired from {addr}! System Ready.")
            led.on(); time.sleep(0.1); led.off()
        except: pass
        continue

    if job_queue:
        if pb is None:
            telemetry_fifo.append('{"stage":"error","msg":"Miaou worker disconnected"}\n')
            telemetry_fifo.append("DONE")
            job_queue.pop(0)
            continue

        raw_prompt, qca_prompt, is_logic = job_queue.pop(0)
        gc.collect()

        p_clean = qca_prompt.lower()
        for char in [".", "?", "!", ",", ":"]:
            p_clean = p_clean.replace(char, f" {char} ")
        p_clean = p_clean.replace("'s", " 's ")

        words_qca = [w for w in p_clean.split() if w]
        canonical_tks = [WORD_TO_ID.get(w, 0) for w in words_qca]
        
        # User typed words for ghost slot matching
        p_user = raw_prompt.lower()
        for char in [".", "?", "!", ",", ":"]:
            p_user = p_user.replace(char, f" {char} ")
        p_user = p_user.replace("'s", " 's ")
        words_user = [w for w in p_user.split() if w]

        if not canonical_tks:
            telemetry_fifo.append("DONE")
            continue

        for c_kv in kv: c_kv['k'].clear(); c_kv['v'].clear()
        
        # Reset signal to Miaou (mode 2)
        try:
            pb.sendall(b'\x02' + struct.pack(f'{EMBED_DIM}f', *([0.0] * EMBED_DIM)))
            ack = recv_all(pb, 2)
            if ack != b"OK":
                pb = None
                telemetry_fifo.append('{"stage":"error","msg":"Miaou reset failed"}\n')
                telemetry_fifo.append("DONE")
                continue
        except Exception:
            pb = None
            telemetry_fifo.append('{"stage":"error","msg":"Miaou socket error"}\n')
            telemetry_fifo.append("DONE")
            continue
        
        qca_json = "[" + ",".join([f'"{w}"' for w in words_qca]) + "]"
        user_json = "[" + ",".join([f'"{w}"' for w in words_user]) + "]"
        telemetry_fifo.append(f'{{"stage":"init","user_prompt":"{raw_prompt}","qca_tokens":{qca_json},"user_tokens":{user_json},"is_logic":{str(is_logic).lower()}}}\n')
        
        t_start = time.ticks_ms()
        cur = canonical_tks[0]
        ct_ranks = [0, 0, 0, 0, 0]
        aborted = False
        
        # Phase 1: Prefill prompt tokens
        for stp in range(len(canonical_tks)):
            if not client_active[0]: aborted = True; break
            led.on()
            tid = canonical_tks[stp]
            read_emb(f_w, tid, cur_x)
            
            last_attn = []
            f_w.seek(L0_OFF)
            for l in range(LAYERS):
                last_attn = run_layer(f_w, l, cur_x, stp)
            
            is_last = (stp == len(canonical_tks) - 1)
            mode_byte = b'\x01' if is_last else b'\x00'
            
            try:
                pb.sendall(mode_byte + struct.pack(f'{EMBED_DIM}f', *cur_x))
                if is_last:
                    rep = recv_all(pb, 10)
                    if not rep:
                        pb = None; aborted = True
                        telemetry_fifo.append('{"stage":"error","msg":"Miaou dropped during head"}\n')
                        break
                    ct_ranks = struct.unpack('5H', rep)
                    cur = ct_ranks[0] if is_logic else sample_top5(ct_ranks)
                else:
                    rep = recv_all(pb, 2)
                    if rep != b"OK":
                        pb = None; aborted = True
                        telemetry_fifo.append('{"stage":"error","msg":"Miaou sync lost"}\n')
                        break
            except Exception:
                pb = None; aborted = True
                telemetry_fifo.append('{"stage":"error","msg":"Miaou connection lost"}\n')
                break
            
            lat = time.ticks_diff(time.ticks_ms(), t_start); t_start = time.ticks_ms()
            ranks = [VOCAB[t] for t in ct_ranks]
            ranks_json = f'["{ranks[0]}","{ranks[1]}","{ranks[2]}","{ranks[3]}","{ranks[4]}"]'
            v_str = "[" + ",".join([f"{v:.2f}" for v in cur_x[:80]]) + "]"
            attn_str = "[" + ",".join([f"{a:.2f}" for a in last_attn]) + "]"
            
            telemetry_fifo.append(
                f'{{"stage":"prefill","cur":{stp+1},"tot":{len(canonical_tks)},"tok":"{VOCAB[tid]}","ms":{lat},"ram":{gc.mem_free()},"v":{v_str},"ranks":{ranks_json},"attn":{attn_str}}}\n'
            )
            led.off()

        # Phase 2: Autoregressive Generation
        if not aborted:
            for gen_stp in range(35):
                if not client_active[0]: break
                word = VOCAB[cur]
                if word in ("<eos>", "<pad>"):
                    led.off()
                    break

                led.on()
                lat = time.ticks_diff(time.ticks_ms(), t_start); t_start = time.ticks_ms()
                v_str = "[" + ",".join([f"{v:.2f}" for v in cur_x[:80]]) + "]"
                ranks = [VOCAB[t] for t in ct_ranks]
                ranks_json = f'["{ranks[0]}","{ranks[1]}","{ranks[2]}","{ranks[3]}","{ranks[4]}"]'
                attn_str = "[" + ",".join([f"{a:.2f}" for a in last_attn]) + "]"

                telemetry_fifo.append(
                    f'{{"stage":"gen","w":"{word}","ms":{lat},"ram":{gc.mem_free()},"v":{v_str},"ranks":{ranks_json},"attn":{attn_str}}}\n'
                )
                
                pos = len(canonical_tks) + gen_stp
                read_emb(f_w, cur, cur_x)
                
                f_w.seek(L0_OFF)
                for l in range(3):
                    last_attn = run_layer(f_w, l, cur_x, pos)
                
                try:
                    pb.sendall(b'\x01' + struct.pack(f'{EMBED_DIM}f', *cur_x))
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