import network
import socket
import select
import struct
import math
import random
import gc
import time
import sys
import machine
from machine import Pin
from array import array

machine.freq(240_000_000)

led = Pin("LED", Pin.OUT)
for _ in range(3):
    led.on(); time.sleep(0.08); led.off(); time.sleep(0.08)

EMBED_DIM = 80
BLOCK_SIZE = 64
NUM_HEADS = 4
HEAD_DIM = EMBED_DIM // NUM_HEADS
VOCAB_SIZE = 1024
BUF = bytearray(256 * 4)

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

def sample_top_k_ranked(logits, k=5, temperature=0.7):
    logits[0] = -1e9
    logits[1] = -1e9

    top_idx = [2] * k
    top_val = [-1e9] * k

    for i in range(2, VOCAB_SIZE):
        val = logits[i]
        if val > top_val[0]:
            top_val[0] = val
            top_idx[0] = i
            j = 0
            while j < k - 1 and top_val[j] > top_val[j + 1]:
                top_val[j], top_val[j + 1] = top_val[j + 1], top_val[j]
                top_idx[j], top_idx[j + 1] = top_idx[j + 1], top_idx[j]
                j += 1

    ranked_idx = [top_idx[k - 1 - i] for i in range(k)]
    ranked_vals = [top_val[k - 1 - i] for i in range(k)]

    if temperature < 0.15:
        chosen = ranked_idx[0]
    else:
        max_v = ranked_vals[0] / temperature
        exps = [math.exp((ranked_vals[i] / temperature) - max_v) for i in range(k)]
        sum_e = sum(exps)
        probs = [e / sum_e for e in exps]
        r = random.random()
        acc = 0.0
        chosen = ranked_idx[0]
        for i in range(k):
            acc += probs[i]
            if r <= acc:
                chosen = ranked_idx[i]
                break

    result = [chosen] + [t for t in ranked_idx if t != chosen]
    while len(result) < 5:
        result.append(ranked_idx[0])
    return result[:5]

def run_layer(f, layer_idx, x, kv_caches):
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

sta = network.WLAN(network.STA_IF)

while True:
    client = None
    f_weights = None
    try:
        sta.active(False)
        time.sleep(0.5)
        sta.active(True)
        
        while not sta.isconnected():
            try: sta.connect('Meow-Net')
            except: pass
            for _ in range(8):
                led.toggle()
                time.sleep(0.3)
                if sta.isconnected(): break

        try: sta.config(pm = 0xa11140)
        except: pass

        led.on(); time.sleep(0.4); led.off()
        client = socket.socket()
        client.settimeout(180.0)
        client.connect(('192.168.4.1', 5000))
        f_weights = open("pico2 weights.model", "rb")
        kv_caches = [{'k': [], 'v': []}, {'k': [], 'v': []},{'k': [], 'v': []}]
        temp = 0.7

        while True:
            ready, _, _ = select.select([client], [], [], 2.0)
            if not ready:
                led.off()
                continue
                
            led.on()
            data = recv_exact(client, EMBED_DIM * 4)
            if not data: break

            floats = struct.unpack(f'{EMBED_DIM}f', data)
            
            if floats[0] == -999.0:
                # Clear all 3 caches:
                for c in kv_caches:
                    c['k'].clear(); c['v'].clear()
                temp = 0.05 if floats[1] == 1.0 else 0.7
                client.sendall(b"OK")
                gc.collect()
                continue

            gc.collect()
            x = list(floats)
            f_weights.seek(0)
            
            x = run_layer(f_weights, 0, x, kv_caches)
            x = run_layer(f_weights, 1, x, kv_caches)
            x = run_layer(f_weights, 2, x, kv_caches)


            ln_f_w = read_vec(f_weights, EMBED_DIM)
            ln_f_b = read_vec(f_weights, EMBED_DIM)
            norm_x = layer_norm(x, ln_f_w, ln_f_b)

            logits = stream_linear(f_weights, VOCAB_SIZE, EMBED_DIM, norm_x)
            ranked_tokens = sample_top_k_ranked(logits, k=5, temperature=temp)

            client.sendall(struct.pack('5H', *ranked_tokens))

    except Exception as e:
        print("MIAOU CRASHED:", e)
        sys.print_exception(e)
        try:
            err_name = (type(e).__name__[:6]).ljust(6)
            client.sendall(b"ERR:" + err_name.encode('utf-8'))
        except: pass
        for _ in range(10):
            led.toggle(); time.sleep(0.05)
    finally:
        if client:
            try: client.close()
            except: pass
        if f_weights:
            try: f_weights.close()
            except: pass
        time.sleep(1)
