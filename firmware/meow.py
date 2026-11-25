# meow.py - Master node for distributed transformer
import _thread, gc, machine, network, random, socket, struct, time
from machine import Pin
from pico_engine import PicoEngine, json_escape, recv_all

machine.freq(240_000_000)
led = Pin("LED", Pin.OUT)
led.off()

eng = PicoEngine("pico1 weights.model", has_emb=True, default_layers=3)
E = eng.embed_dim

try:
    import vocab
    VOCAB = vocab.VOCAB
except Exception:
    VOCAB = ["<unk>", "<pad>", "<eos>"] + ["w" + str(i) for i in range(max(0, eng.vocab_size - 3))]
while len(VOCAB) < eng.vocab_size:
    VOCAB.append("<unk>")
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}

try:
    import wifiConfig
    WIFI_SSID = wifiConfig.SSID
    WIFI_PASS = wifiConfig.PASSWORD
except Exception:
    WIFI_SSID = ""
    WIFI_PASS = ""

def unquote(s):
    res = s.replace("+", " ")
    parts = res.split("%")
    if len(parts) == 1:
        return res
    out = [parts[0]]
    for p in parts[1:]:
        if len(p) >= 2:
            try:
                out.append(chr(int(p[:2], 16)) + p[2:])
            except Exception:
                out.append("%" + p)
        else:
            out.append("%" + p)
    return "".join(out)

def canonicalize_to_qca(raw):
    p = raw.strip()
    is_logic = (
        any(t in p for t in ("[c]", "[q]", "[a]"))
        or any(p.lower().startswith(w) for w in ("who", "what", "where", "is", "does", "why", "how", "question"))
        or ("?" in p)
    )
    if not is_logic:
        return p, False

    if "[q]" in p and "[c]" in p:
        q_pos = p.find("[q]")
        c_pos = p.find("[c]")
        if q_pos < c_pos:
            if not p.endswith("[a]"):
                p = p.rstrip("?.: ") + " [a]"
            return p, True
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
            if second:
                return f"[q] {first} ? [c] {second} . [a]", True
            return f"[q] {first} ? [a]", True
        return f"[q] {second} ? [c] {first} . [a]", True

    if not p.startswith("[q]"):
        p = f"[q] {p}"
    if not p.endswith("[a]"):
        p = p.rstrip("?.: ") + " ? [a]"
    return p, True

def parse_query_params(req):
    q_str = req.split(" ")[1].split("?", 1)[1] if "?" in req.split(" ")[1] else ""
    params = {}
    for pair in q_str.split("&"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            params[k] = unquote(v)
    return params

def sample_top5(ranks):
    r = random.random()
    if r < 0.52:
        return ranks[0]
    if r < 0.76:
        return ranks[1]
    if r < 0.88:
        return ranks[2]
    if r < 0.95:
        return ranks[3]
    return ranks[4]

def pick_next_token(ct_ranks, is_logic, temp, top_k):
    if temp <= 0.02 or top_k <= 1 or is_logic:
        return ct_ranks[0]
    return sample_top5(ct_ranks)

print("\n" + "=" * 50)
print(f"[MEOW] {E}-D / {eng.num_heads}H / {eng.layers}L | KV cap={eng.sink_size}+{eng.window_size}")

try:
    network.hostname("meow")
except Exception:
    pass

sta = network.WLAN(network.STA_IF)
sta.active(True)
if WIFI_SSID:
    sta.connect(WIFI_SSID, WIFI_PASS)
    t_end = time.time() + 12
    while not sta.isconnected() and time.time() < t_end:
        led.toggle()
        time.sleep(0.2)

if sta.isconnected():
    print(f"[MEOW] Connected to Wi-Fi! IP: {sta.ifconfig()[0]}")
else:
    ap = network.WLAN(network.AP_IF)
    ap.active(True)
    ap.config(ssid="Meow-Net", security=0)
    print(f"[MEOW] AP Mode active! IP: {ap.ifconfig()[0]}")

led.off()

ai_sock = socket.socket()
ai_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
ai_sock.bind(("0.0.0.0", 5000))
ai_sock.listen(1)

web_sock = socket.socket()
web_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
web_sock.bind(("0.0.0.0", 80))
web_sock.listen(2)

beacon_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
beacon_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

fifo_lock = _thread.allocate_lock()
job_queue = []
telemetry_fifo = []
client_active = [False]

def send_chunk(cl, payload_bytes):
    cl.sendall(f"{len(payload_bytes):X}\r\n".encode() + payload_bytes + b"\r\n")

def push_telemetry(msg):
    with fifo_lock:
        telemetry_fifo.append(msg)

def core1_http_daemon():
    while True:
        try:
            cl, _ = web_sock.accept()
            cl.settimeout(4.0)
            req = cl.recv(1024).decode("utf-8", "ignore")

            if "GET /gen?" in req:
                params = parse_query_params(req)
                user_prompt = params.get("p", "")
                qca_prompt, is_logic = canonicalize_to_qca(user_prompt)
                try:
                    temp = float(params.get("t", "0.0" if is_logic else "0.75"))
                except Exception:
                    temp = 0.0 if is_logic else 0.75
                try:
                    top_k = int(params.get("k", "1" if is_logic else "5"))
                except Exception:
                    top_k = 1 if is_logic else 5

                cl.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: application/x-ndjson\r\n"
                    b"Transfer-Encoding: chunked\r\n"
                    b"Cache-Control: no-cache, no-transform\r\n"
                    b"X-Content-Type-Options: nosniff\r\n"
                    b"Connection: close\r\n\r\n"
                )

                with fifo_lock:
                    client_active[0] = True
                    telemetry_fifo.clear()
                    job_queue.append((user_prompt, qca_prompt, is_logic, temp, top_k))

                while True:
                    line = None
                    with fifo_lock:
                        if not client_active[0]:
                            break
                        if telemetry_fifo:
                            line = telemetry_fifo.pop(0)
                    if line is not None:
                        if line == "DONE":
                            break
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
                            if not b:
                                break
                            cl.sendall(b)
                except Exception:
                    cl.sendall(b"HTTP/1.1 404 Not Found\r\n\r\nindex.html missing")
                cl.close()
            else:
                cl.sendall(b"HTTP/1.1 404 Not Found\r\n\r\n")
                cl.close()
        except Exception:
            with fifo_lock:
                client_active[0] = False
            time.sleep_ms(20)

_thread.start_new_thread(core1_http_daemon, ())
print("[MEOW] Core 1 HTTP daemon initialized.")

cur_x = [0.0] * E
tx_buf = bytearray(1 + E * 4)
reset_frame = b"\x02" + bytes(E * 4)
pb = None
last_beacon = 0

def split_words(txt):
    s = txt.lower()
    for ch in (".", "?", "!", ",", ":"):
        s = s.replace(ch, f" {ch} ")
    s = s.replace("'s", " 's ")
    return [w for w in s.split() if w]

def send_x(sock, mode_byte, vec):
    tx_buf[0] = mode_byte
    struct.pack_into(f"{E}f", tx_buf, 1, *vec)
    sock.sendall(tx_buf)

while True:
    if pb is None:
        if time.time() - last_beacon >= 2:
            try:
                beacon_sock.sendto(b"MEOW_BEACON", ("255.255.255.255", 5001))
            except Exception:
                pass
            last_beacon = time.time()

        ai_sock.settimeout(0.5)
        try:
            pb, addr = ai_sock.accept()
            pb.settimeout(12.0)
            print(f"[MEOW] -> Miaou paired from {addr}! System Ready.")
            led.on()
            time.sleep(0.1)
            led.off()
        except Exception:
            pass
        continue

    job = None
    with fifo_lock:
        if job_queue:
            job = job_queue.pop(0)

    if job is not None:
        raw_prompt, qca_prompt, is_logic, temp, top_k = job
        gc.collect()

        words_qca = split_words(qca_prompt)
        canonical_tks = [WORD_TO_ID.get(w, 0) for w in words_qca]
        words_user = split_words(raw_prompt)

        if not canonical_tks:
            push_telemetry("DONE")
            continue

        eng.clear_kv()

        try:
            pb.sendall(reset_frame)
            if recv_all(pb, 2) != b"OK":
                pb = None
                push_telemetry('{"stage":"error","msg":"Miaou reset failed"}\n')
                push_telemetry("DONE")
                continue
        except Exception:
            pb = None
            push_telemetry('{"stage":"error","msg":"Miaou socket error"}\n')
            push_telemetry("DONE")
            continue

        qca_json = "[" + ",".join([f'"{json_escape(w)}"' for w in words_qca]) + "]"
        user_json = "[" + ",".join([f'"{json_escape(w)}"' for w in words_user]) + "]"
        push_telemetry(
            f'{{"stage":"init","user_prompt":"{json_escape(raw_prompt)}","qca_tokens":{qca_json},"user_tokens":{user_json},"is_logic":{str(is_logic).lower()}}}\n'
        )

        t_start = time.ticks_ms()
        cur = canonical_tks[0]
        ct_ranks = [0, 0, 0, 0, 0]
        aborted = False
        last_attn = []
        v_vis_dim = 96 if E >= 96 else E

        for stp in range(len(canonical_tks)):
            with fifo_lock:
                if not client_active[0]:
                    aborted = True
                    break
            led.on()
            tid = canonical_tks[stp]
            eng.read_emb(tid, cur_x)
            last_attn = eng.run_all_layers(cur_x, stp, return_attn=True)
            is_last = stp == (len(canonical_tks) - 1)

            try:
                send_x(pb, 1 if is_last else 0, cur_x)
                if is_last:
                    rep = recv_all(pb, 10)
                    if not rep:
                        pb = None
                        aborted = True
                        push_telemetry('{"stage":"error","msg":"Miaou dropped during head"}\n')
                        break
                    ct_ranks = struct.unpack("5H", rep)
                    cur = pick_next_token(ct_ranks, is_logic, temp, top_k)
                else:
                    if recv_all(pb, 2) != b"OK":
                        pb = None
                        aborted = True
                        push_telemetry('{"stage":"error","msg":"Miaou sync lost"}\n')
                        break
            except Exception:
                pb = None
                aborted = True
                push_telemetry('{"stage":"error","msg":"Miaou connection lost"}\n')
                break

            lat = time.ticks_diff(time.ticks_ms(), t_start)
            t_start = time.ticks_ms()
            ranks_json = "[" + ",".join([f'"{json_escape(VOCAB[t])}"' for t in ct_ranks]) + "]"
            v_str = "[" + ",".join([f"{v:.2f}" for v in cur_x[:v_vis_dim]]) + "]"
            attn_str = "[" + ",".join([f"{a:.2f}" for a in last_attn]) + "]"

            push_telemetry(
                f'{{"stage":"prefill","cur":{stp+1},"tot":{len(canonical_tks)},"tok":"{json_escape(VOCAB[tid])}","ms":{lat},"ram":{gc.mem_free()},"v":{v_str},"ranks":{ranks_json},"attn":{attn_str}}}\n'
            )
            led.off()

        if not aborted:
            for gen_stp in range(35):
                with fifo_lock:
                    if not client_active[0]:
                        break
                word = VOCAB[cur]
                if word in ("<eos>", "<pad>"):
                    led.off()
                    break

                led.on()
                lat = time.ticks_diff(time.ticks_ms(), t_start)
                t_start = time.ticks_ms()
                v_str = "[" + ",".join([f"{v:.2f}" for v in cur_x[:v_vis_dim]]) + "]"
                ranks_json = "[" + ",".join([f'"{json_escape(VOCAB[t])}"' for t in ct_ranks]) + "]"
                attn_str = "[" + ",".join([f"{a:.2f}" for a in last_attn]) + "]"

                push_telemetry(
                    f'{{"stage":"gen","w":"{json_escape(word)}","ms":{lat},"ram":{gc.mem_free()},"v":{v_str},"ranks":{ranks_json},"attn":{attn_str}}}\n'
                )

                pos = len(canonical_tks) + gen_stp
                eng.read_emb(cur, cur_x)
                last_attn = eng.run_all_layers(cur_x, pos, return_attn=True)

                try:
                    send_x(pb, 1, cur_x)
                    rep = recv_all(pb, 10)
                    if not rep:
                        break
                except Exception:
                    break

                ct_ranks = struct.unpack("5H", rep)
                cur = pick_next_token(ct_ranks, is_logic, temp, top_k)
                led.off()

        push_telemetry("DONE")
        gc.collect()
    else:
        time.sleep_ms(10)