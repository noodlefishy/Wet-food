import os, sys, json, re, time, math, urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch, torch.nn.functional as F

projectRoot = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, projectRoot)

from benchmarks.fastEvaluator import model, VOCAB, WORD_TO_ID, BLOCK_SIZE, padId, device, NUM_HEADS, HEAD_DIM, EMBED_DIM, cosC, sinC, applyRope

model.eval()
print("=" * 65)
print(f"[*] Loaded PyTorch Model ({EMBED_DIM}-D, {NUM_HEADS} Heads) on {str(device).upper()}")
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

# Real Neural Extraction Hooks on Layer 2
activation_cache = {}

def layer2_block_hook(module, inp, out):
    # Hidden Thought Vector (80-D slice)
    activation_cache["thought_vector"] = out[0].detach()

def layer2_attention_hook(module, inp, out):
    # Extract real multi-head self-attention weights
    x = inp[0]  # (B, T, C)
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
    attn_matrix = F.softmax(scores, dim=-1)  # (B, NUM_HEADS, T, T)
    
    # Average across all 4 attention heads for holistic visualization
    avg_head_attn = attn_matrix[0].mean(dim=0)  # (T, T)
    activation_cache["real_attn_matrix"] = avg_head_attn.detach().cpu()

model.layers[2].register_forward_hook(layer2_block_hook)
model.layers[2].sa.register_forward_hook(layer2_attention_hook)

class SimulatorHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        if parsed.path in ("/", "/index.html"):
            html_path = os.path.join(projectRoot, "firmware", "index.html")
            with open(html_path, "rb") as f:
                content = f.read()
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

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def send_chunk(obj):
                chunk = (json.dumps(obj) + "\n").encode("utf-8")
                self.wfile.write(f"{len(chunk):X}\r\n".encode("utf-8") + chunk + b"\r\n")
                self.wfile.flush()

            tks = tokenize(qca_prompt)
            if not tks:
                send_chunk({"stage": "error", "msg": "Empty prompt"})
                self.wfile.write(b"0\r\n\r\n")
                return

            tokens_str = [VOCAB[t] for t in tks]
            send_chunk({"stage": "init", "user_prompt": raw_prompt, "qca_tokens": tokens_str})

            curr_tokens = list(tks)
            
            # Phase 1: Prefill prompt tokens
            t_prev = time.perf_counter()
            for stp in range(len(tks)):
                prefix = curr_tokens[:stp + 1]
                padded = prefix + [padId] * (BLOCK_SIZE - len(prefix))
                x = torch.tensor([padded[:BLOCK_SIZE]], dtype=torch.long, device=device)
                
                with torch.no_grad():
                    logits = model(x)
                
                lat_ms = int((time.perf_counter() - t_prev) * 1000)
                t_prev = time.perf_counter()

                # Extract 80-D Thought Vector
                tv = activation_cache.get("thought_vector", None)
                v_slice = tv[stp, :80].tolist() if tv is not None else [0.0] * 80

                # Extract Top 5 candidate word rankings
                lg = logits[0, stp].clone()
                lg[WORD_TO_ID.get("<unk>", 0)] = lg[WORD_TO_ID.get("<pad>", 1)] = -1e9
                top_indices = torch.topk(lg, 5).indices.tolist()
                ranks = [VOCAB[idx] for idx in top_indices]

                # Extract REAL authentic attention weights for token at index stp
                attn_mat = activation_cache.get("real_attn_matrix", None)
                if attn_mat is not None and stp < attn_mat.shape[0]:
                    attn_row = attn_mat[stp, :stp + 1].tolist()
                    attn_weights = [round(w, 3) for w in attn_row]
                else:
                    attn_weights = [1.0]

                send_chunk({
                    "stage": "prefill",
                    "cur": stp + 1,
                    "tot": len(tks),
                    "tok": VOCAB[tks[stp]],
                    "ms": max(1, lat_ms),
                    "ram": 140000,
                    "v": v_slice,
                    "ranks": ranks,
                    "attn": attn_weights
                })
                # Rhythmic cyber pacing for prefill
                time.sleep(0.08)

            # Phase 2: Autoregressive Generation
            pred_id = top_indices[0] if is_logic else top_indices[0]
            for gen_stp in range(40):
                word = VOCAB[pred_id]
                lat_ms = int((time.perf_counter() - t_prev) * 1000)
                t_prev = time.perf_counter()

                pos = len(curr_tokens) - 1
                tv = activation_cache.get("thought_vector", None)
                v_slice = tv[pos, :80].tolist() if tv is not None else [0.0] * 80

                # Extract REAL attention for generated token
                attn_mat = activation_cache.get("real_attn_matrix", None)
                if attn_mat is not None and pos < attn_mat.shape[0]:
                    attn_row = attn_mat[pos, :pos + 1].tolist()
                    attn_weights = [round(w, 3) for w in attn_row]
                else:
                    attn_weights = [0.2] * len(curr_tokens)

                send_chunk({
                    "stage": "gen",
                    "w": word,
                    "ms": max(1, lat_ms),
                    "ram": 140000,
                    "v": v_slice,
                    "ranks": ranks,
                    "attn": attn_weights
                })

                if word in ("<eos>", "<pad>"):
                    break

                curr_tokens.append(pred_id)
                if len(curr_tokens) >= BLOCK_SIZE:
                    break

                padded = curr_tokens + [padId] * (BLOCK_SIZE - len(curr_tokens))
                x = torch.tensor([padded[:BLOCK_SIZE]], dtype=torch.long, device=device)
                
                with torch.no_grad():
                    logits = model(x)
                    
                lg = logits[0, len(curr_tokens) - 1].clone()
                lg[WORD_TO_ID.get("<unk>", 0)] = lg[WORD_TO_ID.get("<pad>", 1)] = -1e9
                top_indices = torch.topk(lg, 5).indices.tolist()
                ranks = [VOCAB[idx] for idx in top_indices]
                pred_id = top_indices[0]
                
                # Rhythmic generation pacing (~7 TPS)
                time.sleep(0.14)

            self.wfile.write(b"0\r\n\r\n")

    def log_message(self, format, *args):
        return

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8080), SimulatorHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[!] Simulator server stopped.")