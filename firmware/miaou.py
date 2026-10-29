# miaou.py - Worker (Slave) node for distributed transformer
import network, socket, struct, math, gc, time, machine
from machine import Pin; from array import array
machine.freq(240_000_000); led = Pin("LED", Pin.OUT); led.on()

EMBED_DIM = 80; BLOCK_SIZE = 64; NUM_HEADS = 4; HEAD_DIM = EMBED_DIM // NUM_HEADS; VOCAB_SIZE = 1024; BUF = bytearray(320 * 4)
SINK_SIZE = 4; WINDOW_SIZE = 32; LAYERS = 3

try: import wifiConfig; WIFI_SSID = wifiConfig.SSID; WIFI_PASS = wifiConfig.PASSWORD
except: WIFI_SSID = ""; WIFI_PASS = ""

def read_vec(f, c): return struct.unpack(f'{c}f', f.read(c * 4))

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

print("\n" + "="*50)
print("[MIAOU] Initializing 5-Layer Worker (Layers 3-7)...")

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
kv = [{'k':[], 'v':[]} for _ in range(LAYERS)]  # 5 Layers on Miaou
token_counter = [0]

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
    for h in range(NUM_HEADS):
        ho = h * HEAD_DIM; qh = q[ho : ho + HEAD_DIM]
        scs = [sum(qh[i]*kv[idx]['k'][t][i] for i in range(HEAD_DIM))*sc for t in range(len(kv[idx]['k']))]
        ms = max(scs); wts = [math.exp(s-ms) for s in scs]; s_wts = sum(wts); wts = [e/s_wts for e in wts]
        for t in range(len(wts)):
            for i in range(HEAD_DIM): ao[ho+i] += wts[t] * kv[idx]['v'][t][i]
            
    proj = stream_linear(f, EMBED_DIM, EMBED_DIM, ao)
    x = [x[i] + proj[i] for i in range(EMBED_DIM)]
    nx = layer_norm(x, read_vec(f, EMBED_DIM), read_vec(f, EMBED_DIM))
    h1 = stream_linear(f, 4*EMBED_DIM, EMBED_DIM, nx)
    h2 = stream_linear(f, EMBED_DIM, 4*EMBED_DIM, [max(0.0, val) for val in h1])
    return [x[i] + h2[i] for i in range(EMBED_DIM)]

while True:
    try:
        data = recv_all(c, EMBED_DIM * 4)
        if not data: break
        
        led.on()
        x = list(struct.unpack(f'{EMBED_DIM}f', data))
        
        # Reset signal from Meow
        if x[0] == -999.0:
            for c_kv in kv: c_kv['k'].clear(); c_kv['v'].clear()
            token_counter[0] = 0
            c.sendall(b"OK")
            gc.collect()
            led.off()
            continue
        
        pos = token_counter[0]
        token_counter[0] += 1
        
        f_w.seek(0)
        # Compute 5 layers (Layers 3, 4, 5, 6, 7)
        for l in range(LAYERS): x = run_layer(f_w, l, x, pos)
        
        nx = layer_norm(x, read_vec(f_w, EMBED_DIM), read_vec(f_w, EMBED_DIM))
        lg = stream_linear(f_w, VOCAB_SIZE, EMBED_DIM, nx)
        lg[0] = lg[1] = -1e9  # Mask <unk> and <pad>
        
        # Top 5 extraction
        tops = [2]*5; tvs = [-1e9]*5
        for i in range(2, VOCAB_SIZE):
            if lg[i] > tvs[0]:
                tvs[0], tops[0], j = lg[i], i, 0
                while j < 4 and tvs[j] > tvs[j+1]:
                    tvs[j], tvs[j+1], tops[j], tops[j+1] = tvs[j+1], tvs[j], tops[j+1], tops[j]
                    j += 1
                    
        c.sendall(struct.pack('5H', tops[4], tops[3], tops[2], tops[1], tops[0]))
        led.off()
    except Exception as e:
        print(f"[MIAOU] Connection closed: {e}")
        led.off()
        break
