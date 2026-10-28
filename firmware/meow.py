import network, socket, select, struct, math, gc, time, machine
from array import array; from machine import Pin
machine.freq(240_000_000)
led = Pin("LED", Pin.OUT); led.off()
EMBED_DIM = 80; BLOCK_SIZE = 64; NUM_HEADS = 4; HEAD_DIM = EMBED_DIM // NUM_HEADS; VOCAB_SIZE = 1024; BUF = bytearray(320 * 4)
try: import vocab; VOCAB = vocab.VOCAB
except: VOCAB = ["<unk>", "<pad>", "<eos>"] + ["w"+str(i) for i in range(1021)]
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}; L0_OFF = (VOCAB_SIZE*EMBED_DIM + BLOCK_SIZE*EMBED_DIM)*4
def read_vec(f, c): return struct.unpack(f'{c}f', f.read(c * 4))

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
    bi=0
    while bi<r:
        s=min(64, r-bi); f.readinto(memoryview(BUF)[:s*4]); b=struct.unpack(f'{s}f', memoryview(BUF)[:s*4])
        for i in range(s): out[bi+i]+=b[i]
        bi+=s
    return out

def layer_norm(x, w, b):
    m = sum(x)/len(x); v = sum((i-m)**2 for i in x)/len(x); s = math.sqrt(v+1e-5)
    return [((x[i]-m)/s)*w[i]+b[i] for i in range(len(x))]

f_w = open("pico1 weights.model", "rb")
kv = [{'k':[], 'v':[]} for _ in range(3)]

def run_layer(f, idx, x):
    nx = layer_norm(x, read_vec(f, EMBED_DIM), read_vec(f, EMBED_DIM))
    qkv = stream_linear(f, EMBED_DIM + 2*HEAD_DIM, EMBED_DIM, nx)
    q, k, v = qkv[:EMBED_DIM], qkv[EMBED_DIM:EMBED_DIM+HEAD_DIM], qkv[EMBED_DIM+HEAD_DIM:]
    kv[idx]['k'].append(array('f', k)); kv[idx]['v'].append(array('f', v))
    if len(kv[idx]['k'])>BLOCK_SIZE: kv[idx]['k'].pop(0); kv[idx]['v'].pop(0)
    ao, sc = [0.0]*EMBED_DIM, 1.0/math.sqrt(HEAD_DIM)
    for h in range(NUM_HEADS):
        ho = h*HEAD_DIM; qh = q[ho:ho+HEAD_DIM]
        scs = [sum(qh[i]*kv[idx]['k'][t][i] for i in range(HEAD_DIM))*sc for t in range(len(kv[idx]['k']))]
        ms = max(scs); wts = [math.exp(s-ms) for s in scs]; s_wts = sum(wts); wts = [e/s_wts for e in wts]
        for t in range(len(wts)):
            for i in range(HEAD_DIM): ao[ho+i] += wts[t] * kv[idx]['v'][t][i]
            
    proj = stream_linear(f, EMBED_DIM, EMBED_DIM, ao)
    x = [x[i] + proj[i] for i in range(EMBED_DIM)]
    nx = layer_norm(x, read_vec(f, EMBED_DIM), read_vec(f, EMBED_DIM))
    h1 = stream_linear(f, 4*EMBED_DIM, EMBED_DIM, nx)
    h2 = stream_linear(f, EMBED_DIM, 4*EMBED_DIM, [max(0.0, v) for v in h1])
    return [x[i] + h2[i] for i in range(EMBED_DIM)]

print("\n" + "="*50)
print("[MEOW] Initializing Meow-Net Access Point...")
ap = network.WLAN(network.AP_IF); ap.active(True); ap.config(ssid='Meow-Net', security=0); time.sleep(1)
print(f"[MEOW] AP Online at {ap.ifconfig()[0]}")

ai_sock = socket.socket(); ai_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
ai_sock.bind(('0.0.0.0', 5000)); ai_sock.listen(1)

web_sock = socket.socket(); web_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
web_sock.bind(('0.0.0.0', 80)); web_sock.listen(2)

pb = None; socks = [web_sock, ai_sock]
print("[MEOW] Waiting for Miaou (Pico 2) to connect over Wi-Fi...")

while True:
    r, _, _ = select.select(socks, [], [], 1.0)
    for s in r:
        if s == ai_sock:
            if pb: pb.close()
            pb, addr = ai_sock.accept()
            pb.settimeout(10.0)
            print(f"[MEOW] -> Miaou paired from {addr}! System Ready.")
            led.on(); time.sleep(0.1); led.off()
        elif s == web_sock:
            c, _ = web_sock.accept()
            c.settimeout(5.0); req = c.recv(1024).decode('utf-8', 'ignore')
            
            if "GET /gen?p=" in req and pb:
                raw_p = req.split("GET /gen?p=")[1].split(" ")[0]
                p = unquote(raw_p).lower()
                print(f"\n[MEOW] Prompt: '{p}'")
                
                is_logic = any(t in p for t in ("[c]", "[q]", "[a]")) or any(p.startswith(w) for w in ("who", "what", "where", "is", "does", "why", "how", "question"))
                if is_logic:
                    if "[c]" not in p and not p.startswith("question"): p = f"[c] {p}"
                    if "[q]" in p and not p.endswith("[a]"): p = p.rstrip("?.: ") + " ? [a]"
                tks = [WORD_TO_ID.get(w, 0) for w in p.split()]
                for c_kv in kv: c_kv['k'].clear(); c_kv['v'].clear()
                
                try:
                    pb.sendall(struct.pack(f'{EMBED_DIM}f', *([-999.0, 1.0 if is_logic else 0.0] + [0.0]*(EMBED_DIM-2))))
                    ack = recv_all(pb, 2)
                    if ack != b"OK":
                        c.close(); continue
                except Exception:
                    c.close(); continue

                c.sendall(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n"); cur = tks[-1]
                t_start = time.ticks_ms()
                for stp in range(len(tks) + 30):
                    led.on()
                    tid = tks[stp] if stp < len(tks) else cur
                    
                    f_w.seek(tid * EMBED_DIM * 4); tv = read_vec(f_w, EMBED_DIM)
                    f_w.seek((VOCAB_SIZE * EMBED_DIM + min(stp, BLOCK_SIZE-1) * EMBED_DIM) * 4); pv = read_vec(f_w, EMBED_DIM)
                    x = [tv[i] + pv[i] for i in range(EMBED_DIM)]; f_w.seek(L0_OFF)
                    for l in range(3): x = run_layer(f_w, l, x)
                    
                    try:
                        pb.sendall(struct.pack(f'{EMBED_DIM}f', *x))
                        rep = recv_all(pb, 10)
                        if not rep: break
                    except Exception as e:
                        print(f"[MEOW] Wi-Fi lost: {e}"); break
                    
                    ct_ranks = struct.unpack('5H', rep)
                    cur = ct_ranks[0]
                    lat = time.ticks_diff(time.ticks_ms(), t_start); t_start = time.ticks_ms()
                    v_str = f"[{x[0]:.2f},{x[1]:.2f},{x[2]:.2f},{x[3]:.2f}]"
                    
                    # STREAM TELEMETRY DURING PREFILL
                    if stp < len(tks) - 1:
                        tok_str = VOCAB[tid]
                        print(f" -> Prefilling ({stp+1}/{len(tks)}): '{tok_str}' ({lat}ms)")
                        c.sendall(f'{{"stage":"prefill","cur":{stp+1},"tot":{len(tks)},"tok":"{tok_str}","ms":{lat},"ram":{gc.mem_free()},"v":{v_str}}}\n'.encode())
                    else:
                        # STREAM GENERATED ANSWER TOKENS
                        word = VOCAB[cur]
                        ranks = [VOCAB[t] for t in ct_ranks]
                        ranks_json = f'["{ranks[0]}","{ranks[1]}","{ranks[2]}","{ranks[3]}","{ranks[4]}"]'
                        print(f" -> Generated: [{word}] ({lat}ms)")
                        c.sendall(f'{{"stage":"gen","w":"{word}","ms":{lat},"ram":{gc.mem_free()},"v":{v_str},"ranks":{ranks_json}}}\n'.encode())
                        if (is_logic and word == ".") or word in ("<eos>", "<pad>"): 
                            led.off(); break
                    led.off()
            
            elif "GET / " in req or "GET /index.html" in req:
                c.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n")
                try:
                    with open("index.html", "r") as html_f:
                        while True:
                            chunk = html_f.read(1024)
                            if not chunk: break
                            c.sendall(chunk)
                except Exception as e:
                    c.sendall(b"index.html missing")
            else: 
                c.sendall(b"HTTP/1.1 404 Not Found\r\n\r\n")
            c.close()
    gc.collect()