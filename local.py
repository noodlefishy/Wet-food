# local.py - High-speed local Mac runner with multi-head attention extraction & TCP_NODELAY streaming
import os, sys, json, re, time, math, socket, urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch, torch.nn.functional as F

projectRoot = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, projectRoot)

from benchmarks.fastEvaluator import (
    model, VOCAB, WORD_TO_ID, BLOCK_SIZE, padId, device, 
    NUM_HEADS, HEAD_DIM, EMBED_DIM, cosC, sinC, applyRope, NUM_LAYERS,VOCAB_SIZE
)

model.eval()
print("=" * 65)
print(f"[*] Loaded PyTorch Model ({EMBED_DIM}-D, {NUM_HEADS} Heads, dk={HEAD_DIM}) on {str(device).upper()}")
print("[*] Starting Local Neon Cyber-Terminal at http://localhost:8080")
print("=" * 65)

TOKEN_REGEX = r"\[[cqa]\]|<eos>|<unk>|<pad>|'s|\w+|[.,?!:]"

def tokenize(s):
    for c in [".", "?", "!", ",", ":"]: s = s.replace(c, f" {c} ")
    s = s.replace("'s", " 's ")
    return [WORD_TO_ID.get(w, 0) for w in re.findall(TOKEN_REGEX, s.lower())]

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

activation_cache = {
    "thought_vector": None,
    "layers_attn": [None] * NUM_LAYERS
}

def make_attn_hook(l_idx):
    def hook(module, inp, out):
        x = inp[0]
        B, T, C = x.shape
        qkv = module.qkvProj(x)
        q, k, _ = qkv.chunk(3, dim=-1)
        q = q.view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        q = applyRope(q, cosC, sinC)
        k = applyRope(k, cosC, sinC)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        attn_matrix = F.softmax(scores, dim=-1) # Shape: (B, NUM_HEADS, T, T)
        
        # Cache full multi-head tensor (NUM_HEADS, T, T)
        activation_cache["layers_attn"][l_idx] = attn_matrix[0].detach().cpu()
    return hook

def layer2_block_hook(module, inp, out):
    activation_cache["thought_vector"] = out[0].detach()

model.layers[2].register_forward_hook(layer2_block_hook)
for idx in range(NUM_LAYERS):
    model.layers[idx].sa.register_forward_hook(make_attn_hook(idx))

class SimulatorHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        try:
            self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        if parsed.path in ("/", "/index.html"):
            html_path = os.path.join(projectRoot, "firmware", "index.html")
            with open(html_path, "rb") as f: content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if parsed.path == "/gen":
            query = urllib.parse.parse_qs(parsed.query)
            raw_prompt = query.get("p", [""])[0]
            qca_prompt, is_logic = canonicalize_to_qca(raw_prompt)

            default_temp = 0.0 if is_logic else 0.75
            default_top_k = 1 if is_logic else 25
            try: temp_param = float(query.get("t", [default_temp])[0])
            except (ValueError, TypeError): temp_param = default_temp

            try: top_k_param = int(query.get("k", [default_top_k])[0])
            except (ValueError, TypeError): top_k_param = default_top_k

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()

            def send_chunk(obj):
                chunk = (json.dumps(obj) + "\n").encode("utf-8")
                try:
                    self.wfile.write(f"{len(chunk):X}\r\n".encode("utf-8") + chunk + b"\r\n")
                    self.wfile.flush()
                    return True
                except (BrokenPipeError, ConnectionResetError):
                    return False

            def extract_all_layers_attn(pos, max_len):
                all_l = []
                for l_idx in range(NUM_LAYERS):
                    mat = activation_cache["layers_attn"][l_idx]
                    if mat is not None and pos < mat.shape[1]:
                        avg_row = mat.mean(dim=0)[pos, :max_len].tolist()
                        head_rows = [mat[h, pos, :max_len].tolist() for h in range(NUM_HEADS)]
                        all_l.append({
                            "all": [round(w, 3) for w in avg_row],
                            "heads": [[round(w, 3) for w in h_row] for h_row in head_rows]
                        })
                    else:
                        all_l.append({
                            "all": [0.0] * max_len,
                            "heads": [[0.0] * max_len for _ in range(NUM_HEADS)]
                        })
                return all_l

            canonical_tks = tokenize(qca_prompt)
            user_tks = tokenize(raw_prompt)
            if not canonical_tks:
                send_chunk({"stage": "error", "msg": "Empty prompt"})
                try: self.wfile.write(b"0\r\n\r\n")
                except Exception: pass
                return

            qca_words = [VOCAB[t] for t in canonical_tks]
            user_words = [VOCAB[t] for t in user_tks]

            if not send_chunk({
                "stage": "init",
                "user_prompt": raw_prompt,
                "qca_tokens": qca_words,
                "user_tokens": user_words,
                "is_logic": is_logic
            }):
                return

            curr_tokens = list(canonical_tks)
            
            # Phase 1: Prefill prompt tokens
            t_prev = time.perf_counter()
            for stp in range(len(canonical_tks)):
                prefix = curr_tokens[:stp + 1]
                padded = prefix + [padId] * (BLOCK_SIZE - len(prefix))
                x = torch.tensor([padded[:BLOCK_SIZE]], dtype=torch.long, device=device)
                
                with torch.no_grad():
                    logits = model(x)
                
                lat_ms = int((time.perf_counter() - t_prev) * 1000)
                t_prev = time.perf_counter()

                tv = activation_cache.get("thought_vector", None)
                v_slice = tv[stp, :96].tolist() if tv is not None else [0.0] * 96

                lg = logits[0, stp].clone()
                lg[WORD_TO_ID.get("<unk>", 0)] = lg[WORD_TO_ID.get("<pad>", 1)] = -1e9
                top_indices = torch.topk(lg, 5).indices.tolist()
                ranks = [VOCAB[idx] for idx in top_indices]

                if not send_chunk({
                    "stage": "prefill",
                    "cur": stp + 1,
                    "tot": len(canonical_tks),
                    "tok": VOCAB[canonical_tks[stp]],
                    "ms": max(1, lat_ms),
                    "ram": 140000,
                    "v": v_slice,
                    "ranks": ranks,
                    "attn_layers": extract_all_layers_attn(stp, stp + 1)
                }):
                    return
                time.sleep(0.08)

            # Phase 2: Autoregressive Generation
            generated_history = []
            max_gen = 30 if is_logic else 60

            for gen_stp in range(max_gen):
                pos = len(curr_tokens) - 1
                padded = curr_tokens + [padId] * (BLOCK_SIZE - len(curr_tokens))
                x = torch.tensor([padded[:BLOCK_SIZE]], dtype=torch.long, device=device)
                
                with torch.no_grad():
                    logits = model(x)
                    
                lg = logits[0, pos].clone()
                lg[WORD_TO_ID.get("<unk>", 0)] = lg[WORD_TO_ID.get("<pad>", 1)] = -1e9

                if temp_param <= 0.02 or top_k_param <= 1:
                    pred_id = lg.argmax().item()
                else:
                    for past_tok in set(curr_tokens[-16:] + generated_history[-10:]):
                        lg[past_tok] -= 1.8
                    
                    if not is_logic and gen_stp < 14:
                        lg[WORD_TO_ID.get("<eos>", 2)] = -1e9
                        lg[WORD_TO_ID.get(".", 3)] -= 0.8

                    probs = F.softmax(lg / max(0.05, temp_param), dim=-1)
                    k_val = min(max(1, top_k_param), VOCAB_SIZE)
                    top_k_probs, top_k_indices = torch.topk(probs, k_val)
                    sample_idx = torch.multinomial(top_k_probs, 1).item()
                    pred_id = top_k_indices[sample_idx].item()

                word = VOCAB[pred_id]
                if word in ("<eos>", "<pad>"):
                    break

                lat_ms = int((time.perf_counter() - t_prev) * 1000)
                t_prev = time.perf_counter()

                tv = activation_cache.get("thought_vector", None)
                v_slice = tv[pos, :96].tolist() if tv is not None else [0.0] * 96

                top_indices = torch.topk(lg, 5).indices.tolist()
                ranks = [VOCAB[idx] for idx in top_indices]

                if not send_chunk({
                    "stage": "gen",
                    "w": word,
                    "ms": max(1, lat_ms),
                    "ram": 140000,
                    "v": v_slice,
                    "ranks": ranks,
                    "attn_layers": extract_all_layers_attn(pos, len(curr_tokens))
                }):
                    return

                curr_tokens.append(pred_id)
                generated_history.append(pred_id)
                if len(curr_tokens) >= BLOCK_SIZE:
                    break

                time.sleep(0.16)

            try: self.wfile.write(b"0\r\n\r\n")
            except Exception: pass

    def log_message(self, format, *args):
        return

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8080), SimulatorHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[!] Simulator server stopped.")