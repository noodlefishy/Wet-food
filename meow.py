# ---
# MEOW (Pico WH) - Master Host (64-D, 4-Layer)
# ---
import network
import socket
import select
import struct
import math
import gc
import time
import random
import machine
from array import array
from machine import Pin

# 240 MHz: Rock-solid stability without thermal/voltage sag
machine.freq(240_000_000)

led = Pin("LED", Pin.OUT)
for _ in range(3):
    led.on(); time.sleep(0.08); led.off(); time.sleep(0.08)
led.on()

EMBED_DIM = 64
BLOCK_SIZE = 64
NUM_HEADS = 4
HEAD_DIM = EMBED_DIM // NUM_HEADS
VOCAB_SIZE = 1024
BUF = bytearray(256 * 4)  # 256 floats for MLP

try:
    import vocab
    VOCAB = vocab.VOCAB
except ImportError:
    VOCAB = ["<unk>", "<pad>", "<eos>"] + ["word" + str(i) for i in range(1021)]
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}

def url_decode(s):
    s = s.replace('+', ' ')
    parts = s.split('%')
    res = parts[0]
    for part in parts[1:]:
        if len(part) >= 2:
            try:
                c = chr(int(part[:2], 16))
                res += c + part[2:]
            except:
                res += '%' + part
        else:
            res += '%' + part
    return res

def read_vec(f, count):
    return struct.unpack(f'{count}f', f.read(count * 4))

@micropython.native
def stream_linear(f, rows, cols, x):
    out = [0.0] * rows
    row_bytes = cols * 4
    buf_view = memoryview(BUF)[:row_bytes]
    for r in range(rows):
        f.readinto(buf_view)
        row = struct.unpack(f'{cols}f', buf_view)
        out[r] = sum(row[c] * x[c] for c in range(cols))
        
    b_idx = 0
    while b_idx < rows:
        step = min(64, rows - b_idx)
        step_bytes = step * 4
        f.readinto(memoryview(BUF)[:step_bytes])
        b_chunk = struct.unpack(f'{step}f', memoryview(BUF)[:step_bytes])
        for i in range(step):
            out[b_idx + i] += b_chunk[i]
        b_idx += step
    return out

def layer_norm(x, w, b):
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    std = math.sqrt(var + 1e-5)
    return [((x[i] - mean) / std) * w[i] + b[i] for i in range(len(x))]

def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk: return None
        buf.extend(chunk)
    return buf

f_weights = open("pico1 weights.model", "rb")
kv_caches = [{'k': [], 'v': []}, {'k': [], 'v': []}]

def run_layer(f, layer_idx, x):
    ln1_w = read_vec(f, EMBED_DIM)
    ln1_b = read_vec(f, EMBED_DIM)
    norm_x = layer_norm(x, ln1_w, ln1_b)
    qkv = stream_linear(f, 3 * EMBED_DIM, EMBED_DIM, norm_x)
    q, k, v = qkv[:EMBED_DIM], qkv[EMBED_DIM:2*EMBED_DIM], qkv[2*EMBED_DIM:]

    cache = kv_caches[layer_idx]
    cache['k'].append(array('f', k))
    cache['v'].append(array('f', v))
    if len(cache['k']) > BLOCK_SIZE:
        cache['k'].pop(0); cache['v'].pop(0)

    attn_out = [0.0] * EMBED_DIM
    scale = 1.0 / math.sqrt(HEAD_DIM)
    seq_len = len(cache['k'])

    for h in range(NUM_HEADS):
        h_off = h * HEAD_DIM
        q_h = q[h_off:h_off + HEAD_DIM]
        scores = [sum(q_h[i] * cache['k'][t][h_off + i] for i in range(HEAD_DIM)) * scale for t in range(seq_len)]
        max_s = max(scores)
        exp_s = [math.exp(s - max_s) for s in scores]
        sum_exp = sum(exp_s)
        weights = [e / sum_exp for e in exp_s]

        for t in range(seq_len):
            w_t = weights[t]
            v_vec = cache['v'][t]
            for i in range(HEAD_DIM):
                attn_out[h_off + i] += w_t * v_vec[h_off + i]

    proj = stream_linear(f, EMBED_DIM, EMBED_DIM, attn_out)
    x = [x[i] + proj[i] for i in range(EMBED_DIM)]

    ln2_w = read_vec(f, EMBED_DIM)
    ln2_b = read_vec(f, EMBED_DIM)
    norm_x = layer_norm(x, ln2_w, ln2_b)
    h1 = stream_linear(f, 4 * EMBED_DIM, EMBED_DIM, norm_x)
    h1_relu = [max(0.0, v) for v in h1]
    h2 = stream_linear(f, EMBED_DIM, 4 * EMBED_DIM, h1_relu)
    return [x[i] + h2[i] for i in range(EMBED_DIM)]

with open('index.html', 'r') as f:
    HTML_PAGE = f.read()

def clean_tokenize(text):
    t = text.strip().lower()
    
    if "[c]" in t and "[q]" in t and "[a]" not in t:
        t = t.rstrip("?.: ") + " ? [a]"
    elif not t.startswith("question") and not t.startswith("text"):
        q_words = ("who", "what", "where", "why", "how", "when", "is", "are", "can", "do")
        if t.endswith("?") or any(t.startswith(w) for w in q_words):
            core = t.rstrip("?.: ")
            t = f"question : {core} ? answer :"

    t = t.replace(":", " : ").replace("?", " ? ").replace(".", " . ")
    raw_words = t.split()
    return [w for w in raw_words if w in WORD_TO_ID]

def generate_story_stream(ai_sock, web_client, prompt_text, max_words=50):
    try:
        kv_caches[0]['k'].clear(); kv_caches[0]['v'].clear()
        kv_caches[1]['k'].clear(); kv_caches[1]['v'].clear()

        words = clean_tokenize(prompt_text)
        if not words: words = ["meow", "and", "miaou"]

        is_comprehension = "[a]" in prompt_text or "question" in prompt_text
        reset_temp_flag = 1.0 if is_comprehension else 0.0

        # Send reset packet (1 float + 1 float + 62 zeros = 64 floats)
        reset_packet = struct.pack('64f', -999.0, reset_temp_flag, *([0.0] * 62))
        ai_sock.sendall(reset_packet)
        ack = ai_sock.recv(2)
        if ack != b"OK":
            web_client.sendall(b'{"phase":"err","w":"[Miaou sync error]","ms":0,"v":[],"ram":0,"ranks":[]}\n')
            return False

        tokens = [WORD_TO_ID[w] for w in words]
        current_tok = tokens[-1]
        
        # Proper offset for 64-D model
        LAYER0_OFFSET = (VOCAB_SIZE * EMBED_DIM + BLOCK_SIZE * EMBED_DIM) * 4

        prompt_words = [VOCAB[t] for t in tokens]
        json_prompt_words = '["' + '","'.join(prompt_words) + '"]'
        web_client.sendall(f'{{"phase":"init","tokens":{json_prompt_words}}}\n'.encode('utf-8'))

        SUPPRESS_TOKENS = {":", "?", "answer", "question", "<unk>", "<pad>"}
        content_started = False
        sentence_count = 0
        gen_tokens_count = 0

        for step in range(len(tokens) + max_words):
            t_start = time.ticks_ms()
            led.toggle()
            pos = min(step, BLOCK_SIZE - 1)
            
            is_prompt_input = step < len(tokens)
            tok_id = tokens[step] if is_prompt_input else current_tok

            f_weights.seek(tok_id * EMBED_DIM * 4)
            tok_v = read_vec(f_weights, EMBED_DIM)

            f_weights.seek((VOCAB_SIZE * EMBED_DIM + pos * EMBED_DIM) * 4)
            pos_v = read_vec(f_weights, EMBED_DIM)

            x = [tok_v[i] + pos_v[i] for i in range(EMBED_DIM)]
            f_weights.seek(LAYER0_OFFSET)
            
            # Run Layers 0 and 1
            x = run_layer(f_weights, 0, x)
            x = run_layer(f_weights, 1, x)

            try:
                # Transmit 64 floats to Miaou
                ai_sock.sendall(struct.pack('64f', *x))
                
                reply = recv_exact(ai_sock, 10)
                if not reply:
                    web_client.sendall(b'{"phase":"err","w":"[Miaou disconnected]","ms":0,"v":[],"ram":0,"ranks":[]}\n')
                    return False
                if reply.startswith(b"ERR:"):
                    err_type = reply[4:].decode('utf-8', 'ignore').strip()
                    web_client.sendall(f'{{"phase":"err","w":"[Miaou crash: {err_type}]","ms":0,"v":[],"ram":0,"ranks":[]}}\n'.encode('utf-8'))
                    return False

                cand_tokens = struct.unpack('5H', reply)
                current_tok = cand_tokens[0]
            except Exception as e:
                web_client.sendall(f'{{"phase":"err","w":"[Error: {e}]","ms":0,"v":[],"ram":0,"ranks":[]}\n'.encode('utf-8'))
                return False

            t_elapsed = time.ticks_diff(time.ticks_ms(), t_start)
            v_prev = [round(x[0], 2), round(x[1], 2), round(x[2], 2), round(x[3], 2)]
            cand_words = [VOCAB[t] for t in cand_tokens]
            json_ranks = '["' + '","'.join(cand_words) + '"]'

            # PREFILL
            if step < len(tokens) - 1:
                prefill_json = f'{{"phase":"prefill","idx":{step},"w":"{prompt_words[step]}","ms":{t_elapsed},"v":{v_prev},"ram":{gc.mem_free()},"ranks":{json_ranks}}}\n'
                web_client.sendall(prefill_json.encode('utf-8'))

            # GENERATION
            else:
                word = VOCAB[current_tok]

                if not content_started:
                    if word in SUPPRESS_TOKENS:
                        continue
                    content_started = True

                # Autonomous End-Of-Sequence
                if word in ["<eos>", "[c]", "[q]"]:
                    break

                gen_tokens_count += 1
                telemetry = f'{{"phase":"gen","w":"{word}","ms":{t_elapsed},"v":{v_prev},"ram":{gc.mem_free()},"ranks":{json_ranks}}}\n'
                web_client.sendall(telemetry.encode('utf-8'))

                # Q&A Stop
                if is_comprehension and word == ".":
                    break

                # Story Stop
                if not is_comprehension and word == ".":
                    sentence_count += 1
                    if sentence_count >= 2 or gen_tokens_count >= 18:
                        break

                if len(tokens) + gen_tokens_count >= BLOCK_SIZE - 2:
                    break

            gc.collect()

        return True

    except Exception as e:
        try:
            web_client.sendall(f'{{"phase":"err","w":"[CRITICAL: {e}]","ms":0,"v":[],"ram":0,"ranks":[]}\n'.encode('utf-8'))
        except: pass
        return False

# AP Setup
ap = network.WLAN(network.AP_IF)
ap.active(False)
time.sleep(0.5)
ap.active(True)
ap.config(ssid='Meow-Net', security=0)
try: ap.config(pm = 0xa11140)
except: pass
time.sleep(1.0)

ai_server = socket.socket()
ai_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
ai_server.bind(('0.0.0.0', 5000))
ai_server.listen(1)

web_server = socket.socket()
web_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
web_server.bind(('0.0.0.0', 80))
web_server.listen(2)

print("Meow is ready on Meow-Net.")
led.off()
pico_b_sock = None
sockets_list = [web_server, ai_server]

while True:
    try:
        r, _, _ = select.select(sockets_list, [], [], 1.0)
        
        for s in r:
            if s == ai_server:
                try:
                    if pico_b_sock:
                        try: pico_b_sock.close()
                        except: pass
                    pico_b_sock, _ = ai_server.accept()
                    pico_b_sock.settimeout(180.0)
                    led.on(); time.sleep(0.1); led.off()
                    print("Miaou connected.")
                except Exception:
                    pico_b_sock = None
                
            elif s == web_server:
                client = None
                try:
                    client, _ = web_server.accept()
                    client.settimeout(3.0)
                    req = client.recv(1024).decode('utf-8', 'ignore')

                    if "GET /gen?p=" in req:
                        if not pico_b_sock:
                            client.sendall(b'HTTP/1.1 200 OK\r\n\r\n{"phase":"err","w":"[Miaou is offline]","ms":0,"v":[],"ram":0,"ranks":[]}\n')
                        else:
                            try:
                                raw_p = req.split("GET /gen?p=")[1].split(" ")[0]
                                prompt_arg = url_decode(raw_p)
                            except:
                                prompt_arg = "meow"
                            
                            client.settimeout(180.0)
                            client.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n")
                            
                            success = generate_story_stream(pico_b_sock, client, prompt_arg)
                            if not success:
                                try: pico_b_sock.close()
                                except: pass
                                pico_b_sock = None
                            led.off()
                    else:
                        client.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n" + HTML_PAGE.encode('utf-8'))
                except Exception:
                    pass
                finally:
                    if client:
                        try: client.close()
                        except: pass

        gc.collect()

    except Exception:
        time.sleep(0.5)
        gc.collect()