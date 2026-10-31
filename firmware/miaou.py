# miaou.py - Worker (Slave) node for distributed transformer
import network, socket, struct, math, gc, time, machine
from machine import Pin; from array import array
machine.freq(240_000_000); led = Pin("LED", Pin.OUT); led.on()

EMBED_DIM = 120
BLOCK_SIZE = 64
NUM_HEADS = 6
HEAD_DIM = EMBED_DIM // NUM_HEADS
VOCAB_SIZE = 1024
ROW_B = 4 + EMBED_DIM
L0_OFF = VOCAB_SIZE * ROW_B
SINK_SIZE = 4
WINDOW_SIZE = 20
LAYERS = 3

try: import wifiConfig; WIFI_SSID = wifiConfig.SSID; WIFI_PASS = wifiConfig.PASSWORD
except: WIFI_SSID = ""; WIFI_PASS = ""

# Zero-allocation buffers
BUF_nx    = [0.0] * EMBED_DIM
BUF_qkv   = [0.0] * (3 * EMBED_DIM)
BUF_ao    = [0.0] * EMBED_DIM
BUF_proj  = [0.0] * EMBED_DIM
BUF_mlp1  = [0.0] * (4 * EMBED_DIM)
BUF_mlp2  = [0.0] * EMBED_DIM
BUF_ln_w  = [0.0] * EMBED_DIM
BUF_ln_b  = [0.0] * EMBED_DIM
BUF_lg    = [0.0] * VOCAB_SIZE
ROW_BUF   = bytearray(4 + 4 * EMBED_DIM)
BIAS_BUF  = bytearray(64 * 4)

def recv_all(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet: return None
        data.extend(packet)
    return data

@micropython.native
def stream_linear_into(f, r, c, x, out):
    row_bytes = 4 + c
    mv = memoryview(ROW_BUF)[:row_bytes]
    for i in range(r):
        f.readinto(mv)
        scale = struct.unpack_from('f', mv, 0)[0]
        s = 0.0
        for j in range(c):
            v = mv[4 + j]
            s += (v if v < 128 else v - 256) * x[j]
        out[i] = s * scale
    bi = 0
    mv_b = memoryview(BIAS_BUF)
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
    mv = memoryview(BIAS_BUF)[:c * 4]
    f.readinto(mv)
    vals = struct.unpack(f'{c}f', mv)
    for i in range(c):
        target[i] = vals[i]

half_head = HEAD_DIM // 2
inv_freq = [1.0 / (10000.0 ** (2.0 * i / HEAD_DIM)) for i in range(half_head)]
cos_tab = [[math.cos(p * inv_freq[i]) for i in range(half_head)] for p in range(BLOCK_SIZE)]
sin_tab = [[math.sin(p * inv_freq[i]) for i in range(half_head)] for p in range(BLOCK_SIZE)]

print("\n" + "="*50)
print("[MIAOU] Initializing 3-Layer Worker (Layers 3-5, Zero-Allocation)...")

try: network.hostname('miaou')
except: pass

sta = network.WLAN(network.STA_IF)
sta.active(True)
if WIFI_SSID:
    print(f"[MIAOU] Connecting to {WIFI_SSID}...")
    sta.connect(WIFI_SSID, WIFI_PASS)
    t_end = time.time() + 12
    while not sta.isconnected() and time.time() < t_end:
        led.toggle(); time.sleep(0.2)

is_home_wifi = sta.isconnected()
if is_home_wifi:
    print(f"[MIAOU] Connected to Home Wi-Fi! IP: {sta.ifconfig()[0]}")
    beacon_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    beacon_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    beacon_sock.bind(('0.0.0.0', 5001))
    meow_ip = None
    while not meow_ip:
        led.toggle()
        data, addr = beacon_sock.recvfrom(32)
        if b"MEOW_BEACON" in data:
            meow_ip = addr[0]
            print(f"[MIAOU] Discovered Meow at {meow_ip}!")
    beacon_sock.close()
else:
    print("[MIAOU] Connecting to Meow-Net...")
    sta.connect('Meow-Net')
    while not sta.isconnected(): led.toggle(); time.sleep(0.25)
    meow_ip = '192.168.4.1'

c = socket.socket()
connected = False
while not connected:
    led.toggle()
    try:
        c.connect((meow_ip, 5000))
        c.settimeout(None)
        connected = True
        print(f"[MIAOU] Paired with Meow at {meow_ip}:5000!")
    except: time.sleep(0.2)

led.off()

f_w = open("pico2 weights.model", "rb")
kv = [{'k':[], 'v':[]} for _ in range(LAYERS)]
token_counter = [0]
x_vec = [0.0] * EMBED_DIM

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
    sc = 1.0 / math.sqrt(HEAD_DIM)
    seq_len = len(kv[idx]['k'])
    
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
            
        inv_sum = 1.0 / sum_exp
        for t in range(seq_len):
            w = BUF_mlp1[t] * inv_sum
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

while True:
    try:
        header = recv_all(c, 1 + EMBED_DIM * 4)
        if not header: break
        
        mode = header[0]
        data = header[1:]
        
        # Mode 2: Reset
        if mode == 2:
            for c_kv in kv: c_kv['k'].clear(); c_kv['v'].clear()
            token_counter[0] = 0
            c.sendall(b"OK")
            gc.collect()
            continue
            
        led.on()
        floats = struct.unpack(f'{EMBED_DIM}f', data)
        for i in range(EMBED_DIM): x_vec[i] = floats[i]
        
        pos = token_counter[0]
        token_counter[0] += 1
        
        f_w.seek(0)
        for l in range(LAYERS): run_layer(f_w, l, x_vec, pos)
        
        # Mode 0: Intermediate prefill token (KV cache updated; skip LM head)
        if mode == 0:
            c.sendall(b"OK")
            led.off()
            continue
            
        # Mode 1: Final prompt token or generation token (compute logits)
        read_vec_into(f_w, EMBED_DIM, BUF_ln_w)
        read_vec_into(f_w, EMBED_DIM, BUF_ln_b)
        layer_norm_into(x_vec, BUF_ln_w, BUF_ln_b, BUF_nx)
        
        stream_linear_into(f_w, VOCAB_SIZE, EMBED_DIM, BUF_nx, BUF_lg)
        BUF_lg[0] = BUF_lg[1] = -1e9
        
        tops = [2]*5; tvs = [-1e9]*5
        for i in range(2, VOCAB_SIZE):
            if BUF_lg[i] > tvs[0]:
                tvs[0], tops[0], j = BUF_lg[i], i, 0
                while j < 4 and tvs[j] > tvs[j+1]:
                    tvs[j], tvs[j+1], tops[j], tops[j+1] = tvs[j+1], tvs[j], tops[j+1], tops[j]
                    j += 1
                    
        c.sendall(struct.pack('5H', tops[4], tops[3], tops[2], tops[1], tops[0]))
        led.off()
    except Exception as e:
        print(f"[MIAOU] Connection error: {e}")
        led.off()
        break