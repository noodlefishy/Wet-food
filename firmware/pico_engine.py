# pico_engine.py - Shared zero-alloc INT8 transformer engine for RP2040 & RP2350
import gc, math, os, struct
from array import array

HEADER_MAGIC = b"PICO"
HEADER_FMT = "<4s7I"
HEADER_SIZE = 32

def recv_all(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def json_escape(s):
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        else:
            out.append(ch)
    return "".join(out)

@micropython.viper
def dot_i8_i32(w_buf, x_buf, n: int) -> int:
    wp: ptr8 = ptr8(w_buf)
    xp: ptr32 = ptr32(x_buf)
    acc: int = 0
    for j in range(n):
        w: int = wp[j]
        if w >= 128:
            w -= 256
        acc += w * xp[j]
    return acc

class PicoEngine:
    def __init__(self, weights_path, has_emb=True, default_layers=3):
        self.f = open(weights_path, "rb")
        raw = self.f.read(HEADER_SIZE)
        if len(raw) == HEADER_SIZE and raw[:4] == HEADER_MAGIC:
            _, _, v_size, e_dim, n_heads, _, node_l, b_size = struct.unpack(HEADER_FMT, raw)
            self.data_offset = HEADER_SIZE
        else:
            self.f.seek(0)
            v_size, e_dim, n_heads, node_l, b_size = 1024, 96, 4, default_layers, 64
            self.data_offset = 0

        self.vocab_size = v_size
        self.embed_dim = e_dim
        self.num_heads = n_heads
        self.head_dim = e_dim // n_heads
        self.layers = node_l
        self.block_size = b_size
        self.row_b = 4 + e_dim
        self.l0_offset = self.data_offset + (v_size * self.row_b if has_emb else 0)

        # Static zero-allocation inference buffers
        self.nx = [0.0] * e_dim
        self.qkv = [0.0] * (3 * e_dim)
        self.ao = [0.0] * e_dim
        self.proj = [0.0] * e_dim
        self.mlp1 = [0.0] * (4 * e_dim)
        self.mlp2 = [0.0] * e_dim
        self.ln_w = [0.0] * e_dim
        self.ln_b = [0.0] * e_dim
        self.attn = [0.0] * b_size
        self.scale_buf = bytearray(4)

        max_cols = 4 * e_dim
        self.row_buf = bytearray(4 + max_cols)
        self.mv_row = memoryview(self.row_buf)
        self.bias_buf = bytearray(64 * 4)
        self.mv_bias = memoryview(self.bias_buf)
        self.x_q = array("i", [0] * max_cols)

        # Precompute RoPE tables
        half_head = self.head_dim // 2
        self.half_head = half_head
        inv_freq = [1.0 / (10000.0 ** (2.0 * i / self.head_dim)) for i in range(half_head)]
        self.cos_tab = [[math.cos(p * inv_freq[i]) for i in range(half_head)] for p in range(b_size)]
        self.sin_tab = [[math.sin(p * inv_freq[i]) for i in range(half_head)] for p in range(b_size)]

        # Auto-size KV cache for Pico 1 (RP2040) vs Pico 2 (RP2350)
        gc.collect()
        free_ram = gc.mem_free()
        mach = os.uname().machine
        is_rp2350 = ("RP2350" in mach) or ("Pico 2" in mach) or (free_ram > 220000)

        bytes_per_tok = self.layers * 2 * e_dim * 4
        safety_margin = 48000 if is_rp2350 else 10000
        max_tokens = (free_ram - safety_margin) // bytes_per_tok

        if is_rp2350:
            self.sink_size = 4
            cap = min(b_size, max(16, max_tokens))
        else:
            self.sink_size = 2
            cap = min(14, max(8, max_tokens))
        self.window_size = max(1, cap - self.sink_size)
        self.kv = [{"k": [], "v": []} for _ in range(self.layers)]

    def clear_kv(self):
        for c_kv in self.kv:
            c_kv["k"].clear()
            c_kv["v"].clear()

    def read_emb(self, idx, out):
        self.f.seek(self.data_offset + idx * self.row_b)
        self.f.readinto(self.mv_row[: self.row_b])
        s = struct.unpack_from("f", self.mv_row, 0)[0]
        rb = self.row_buf
        for i in range(self.embed_dim):
            v = rb[4 + i]
            out[i] = (v if v < 128 else v - 256) * s

    def read_vec_into(self, c, target):
        raw = self.f.read(c * 4)
        vals = struct.unpack(f"{c}f", raw)
        for i in range(c):
            target[i] = vals[i]

    def layer_norm_into(self, x, w, b, out):
        n = self.embed_dim
        m = sum(x) / n
        v = sum((x[i] - m) ** 2 for i in range(n)) / n
        s = math.sqrt(v + 1e-5)
        for i in range(n):
            out[i] = ((x[i] - m) / s) * w[i] + b[i]

    @micropython.native
    def stream_linear_into(self, r, c, x, out):
        # Quantize activation x once per linear call to avoid 31-bit small-int overflow
        max_abs = 1e-6
        for j in range(c):
            av = x[j]
            if av < 0.0:
                av = -av
            if av > max_abs:
                max_abs = av
        x_scale = max_abs / 4096.0
        inv_xs = 1.0 / x_scale
        xq = self.x_q
        for j in range(c):
            xq[j] = int(x[j] * inv_xs)

        f = self.f
        row_sz = 4 + c
        mv = self.mv_row[:row_sz]
        w_mv = self.mv_row[4:row_sz]

        for i in range(r):
            f.readinto(mv)
            w_scale = struct.unpack_from("f", mv, 0)[0]
            acc = dot_i8_i32(w_mv, xq, c)
            out[i] = acc * (w_scale * x_scale)

        bi = 0
        mv_b = self.mv_bias
        while bi < r:
            chunk = 64 if (r - bi) > 64 else (r - bi)
            f.readinto(mv_b[: chunk * 4])
            b = struct.unpack(f"{chunk}f", mv_b[: chunk * 4])
            for i in range(chunk):
                out[bi + i] += b[i]
            bi += chunk

    def run_layer(self, idx, x, pos, return_attn=False):
        E = self.embed_dim
        H = self.num_heads
        D = self.head_dim
        hh = self.half_head

        self.read_vec_into(E, self.ln_w)
        self.read_vec_into(E, self.ln_b)
        self.layer_norm_into(x, self.ln_w, self.ln_b, self.nx)
        self.stream_linear_into(3 * E, E, self.nx, self.qkv)

        p_clamped = pos if pos < self.block_size else self.block_size - 1
        c_tab = self.cos_tab[p_clamped]
        s_tab = self.sin_tab[p_clamped]
        qkv = self.qkv
        for h in range(H):
            ho = h * D
            for i in range(hh):
                i2 = ho + i * 2
                q0, q1 = qkv[i2], qkv[i2 + 1]
                qkv[i2] = q0 * c_tab[i] - q1 * s_tab[i]
                qkv[i2 + 1] = q0 * s_tab[i] + q1 * c_tab[i]
                k0, k1 = qkv[E + i2], qkv[E + i2 + 1]
                qkv[E + i2] = k0 * c_tab[i] - k1 * s_tab[i]
                qkv[E + i2 + 1] = k0 * s_tab[i] + k1 * c_tab[i]

        k_cache = self.kv[idx]["k"]
        v_cache = self.kv[idx]["v"]
        k_cache.append(array("f", qkv[E : 2 * E]))
        v_cache.append(array("f", qkv[2 * E : 3 * E]))

        if len(k_cache) > self.sink_size + self.window_size:
            k_cache.pop(self.sink_size)
            v_cache.pop(self.sink_size)

        ao = self.ao
        for i in range(E):
            ao[i] = 0.0
        seq_len = len(k_cache)
        attn_out = self.attn
        if return_attn:
            for t in range(seq_len):
                attn_out[t] = 0.0

        sc = 1.0 / math.sqrt(D)
        scores = self.mlp1
        for h in range(H):
            ho = h * D
            ms = -1e9
            for t in range(seq_len):
                k_t = k_cache[t]
                dot = 0.0
                for i in range(D):
                    dot += qkv[ho + i] * k_t[ho + i]
                score = dot * sc
                scores[t] = score
                if score > ms:
                    ms = score

            sum_exp = 0.0
            for t in range(seq_len):
                ew = math.exp(scores[t] - ms)
                scores[t] = ew
                sum_exp += ew

            inv_sum = 1.0 / (sum_exp if sum_exp > 1e-12 else 1.0)
            for t in range(seq_len):
                w = scores[t] * inv_sum
                if return_attn:
                    attn_out[t] += w / H
                v_t = v_cache[t]
                for i in range(D):
                    ao[ho + i] += w * v_t[ho + i]

        self.stream_linear_into(E, E, ao, self.proj)
        for i in range(E):
            x[i] += self.proj[i]

        self.read_vec_into(E, self.ln_w)
        self.read_vec_into(E, self.ln_b)
        self.layer_norm_into(x, self.ln_w, self.ln_b, self.nx)

        mlp1 = self.mlp1
        self.stream_linear_into(4 * E, E, self.nx, mlp1)
        for j in range(4 * E):
            if mlp1[j] < 0.0:
                mlp1[j] = 0.0

        self.stream_linear_into(E, 4 * E, mlp1, self.mlp2)
        for i in range(E):
            x[i] += self.mlp2[i]

        return attn_out[:seq_len] if return_attn else None

    def run_all_layers(self, x, pos, return_attn=False):
        self.f.seek(self.l0_offset)
        last_attn = []
        for l in range(self.layers):
            res = self.run_layer(l, x, pos, return_attn=(return_attn and l == self.layers - 1))
            if res is not None:
                last_attn = res
        return last_attn