import network, socket, select, struct, math, random, gc, time, machine
from machine import Pin; from array import array
machine.freq(240_000_000); led = Pin("LED", Pin.OUT); led.on()
EMBED_DIM = 96; BLOCK_SIZE = 64; NUM_HEADS = 4; HEAD_DIM = EMBED_DIM // NUM_HEADS; VOCAB_SIZE = 1024; BUF = bytearray(256 * 4)
def read_vec(f, c): return struct.unpack(f'{c}f', f.read(c * 4))

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

sta = network.WLAN(network.STA_IF); sta.active(True)
while not sta.isconnected():
    try: sta.connect('Meow-Net')
    except: pass
    time.sleep(1)
c = socket.socket(); c.connect(('192.168.4.1', 5000)); led.off()
f_w = open("pico2 weights.model", "rb"); kv = [{'k':[], 'v':[]} for _ in range(3)]; temp = 0.7

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
    x = [x[i] + stream_linear(f, EMBED_DIM, EMBED_DIM, ao)[i] for i in range(EMBED_DIM)]
    nx = layer_norm(x, read_vec(f, EMBED_DIM), read_vec(f, EMBED_DIM))
    return [x[i] + stream_linear(f, EMBED_DIM, 4*EMBED_DIM, [max(0.0, v) for v in stream_linear(f, 4*EMBED_DIM, EMBED_DIM, nx)])[i] for i in range(EMBED_DIM)]

while True:
    r, _, _ = select.select([c], [], [], 2.0)
    if not r: continue
    data = c.recv(EMBED_DIM * 4)
    if not data: break
    x = list(struct.unpack(f'{EMBED_DIM}f', data))
    if x[0] == -999.0:
        for c_kv in kv: c_kv['k'].clear(); c_kv['v'].clear()
        temp = 0.05 if x[1] == 1.0 else 0.7; c.sendall(b"OK"); gc.collect(); continue
    f_w.seek(0)
    for l in range(3): x = run_layer(f_w, l, x)
    nx = layer_norm(x, read_vec(f_w, EMBED_DIM), read_vec(f_w, EMBED_DIM))
    lg = stream_linear(f_w, VOCAB_SIZE, EMBED_DIM, nx)
    lg[0] = lg[1] = -1e9; tops = [2]*5; tvs = [-1e9]*5
    
    if temp < 0.1:
        bst = 2
        for i in range(2, VOCAB_SIZE):
            if lg[i] > lg[bst]: bst = i
        c.sendall(struct.pack('5H', bst, bst, bst, bst, bst))
    else:
        scaled = [v/temp for v in lg]
        ms = max(scaled); exps = [math.exp(v-ms) for v in scaled]
        probs = [e/sum(exps) for e in exps]; r_val, acc, curr = random.random(), 0.0, len(probs)-1
        for i, pr in enumerate(probs):
            acc += pr
            if r_val <= acc: curr = i; break
        c.sendall(struct.pack('5H', curr, curr, curr, curr, curr))