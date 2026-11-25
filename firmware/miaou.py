# miaou.py - Worker node for distributed transformer
import gc, machine, network, socket, struct, time
from machine import Pin
from pico_engine import PicoEngine, recv_all

machine.freq(240_000_000)
led = Pin("LED", Pin.OUT)
led.on()

eng = PicoEngine("pico2 weights.model", has_emb=False, default_layers=3)
E = eng.embed_dim
V = eng.vocab_size
BUF_lg = [0.0] * V

try:
    import wifiConfig
    WIFI_SSID = wifiConfig.SSID
    WIFI_PASS = wifiConfig.PASSWORD
except Exception:
    WIFI_SSID = ""
    WIFI_PASS = ""

print("\n" + "=" * 50)
print(f"[MIAOU] {E}-D / {eng.num_heads}H / {eng.layers}L | KV cap={eng.sink_size}+{eng.window_size}")

try:
    network.hostname("miaou")
except Exception:
    pass

sta = network.WLAN(network.STA_IF)
sta.active(True)
if WIFI_SSID:
    print(f"[MIAOU] Connecting to {WIFI_SSID}...")
    sta.connect(WIFI_SSID, WIFI_PASS)
    t_end = time.time() + 12
    while not sta.isconnected() and time.time() < t_end:
        led.toggle()
        time.sleep(0.2)

if sta.isconnected():
    print(f"[MIAOU] Connected to Wi-Fi! IP: {sta.ifconfig()[0]}")
    beacon_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    beacon_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    beacon_sock.bind(("0.0.0.0", 5001))
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
    sta.connect("Meow-Net")
    while not sta.isconnected():
        led.toggle()
        time.sleep(0.25)
    meow_ip = "192.168.4.1"

c = socket.socket()
connected = False
while not connected:
    led.toggle()
    try:
        c.connect((meow_ip, 5000))
        c.settimeout(None)
        connected = True
        print(f"[MIAOU] Paired with Meow at {meow_ip}:5000!")
    except Exception:
        time.sleep(0.2)

led.off()

token_counter = 0
x_vec = [0.0] * E
pkt_size = 1 + E * 4

while True:
    try:
        header = recv_all(c, pkt_size)
        if not header:
            break

        mode = header[0]
        if mode == 2:
            eng.clear_kv()
            token_counter = 0
            c.sendall(b"OK")
            gc.collect()
            continue

        led.on()
        floats = struct.unpack_from(f"{E}f", header, 1)
        for i in range(E):
            x_vec[i] = floats[i]

        pos = token_counter
        token_counter += 1
        eng.run_all_layers(x_vec, pos, return_attn=False)

        if mode == 0:
            c.sendall(b"OK")
            led.off()
            continue

        eng.read_vec_into(E, eng.ln_w)
        eng.read_vec_into(E, eng.ln_b)
        eng.layer_norm_into(x_vec, eng.ln_w, eng.ln_b, eng.nx)
        eng.stream_linear_into(V, E, eng.nx, BUF_lg)
        BUF_lg[0] = BUF_lg[1] = -1e9

        tops = [2] * 5
        tvs = [-1e9] * 5
        for i in range(2, V):
            if BUF_lg[i] > tvs[0]:
                tvs[0], tops[0], j = BUF_lg[i], i, 0
                while j < 4 and tvs[j] > tvs[j + 1]:
                    tvs[j], tvs[j + 1], tops[j], tops[j + 1] = tvs[j + 1], tvs[j], tops[j + 1], tops[j]
                    j += 1

        c.sendall(struct.pack("5H", tops[4], tops[3], tops[2], tops[1], tops[0]))
        led.off()
    except Exception as e:
        print(f"[MIAOU] Connection error: {e}")
        led.off()
        break