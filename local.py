# mac_server.py - Dual-Mode (Logic QCA + Open Narrative Story) High-Speed Server
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

activation_cache = {}

def layer2_block_hook(module, inp, out):
    activation_cache["thought_vector"] = out[0].detach()

def layer2_attention_hook(module, inp, out):
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
    attn_matrix = F.softmax(scores, dim=-1)
    
    avg_head_attn = attn_matrix[0].mean(dim=0)
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

            canonical_tks = tokenize(qca_prompt)
            user_tks = tokenize(raw_prompt)
            if not canonical_tks:
                send_chunk({"stage": "error", "msg": "Empty prompt"})
                self.wfile.write(b"0\r\n\r\n")
                return

            qca_words = [VOCAB[t] for t in canonical_tks]
            user_words = [VOCAB[t] for t in user_tks]

            send_chunk({
                "stage": "init",
                "user_prompt": raw_prompt,
                "qca_tokens": qca_words,
                "user_tokens": user_words,
                "is_logic": is_logic
            })

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
                v_slice = tv[stp, :80].tolist() if tv is not None else [0.0] * 80

                lg = logits[0, stp].clone()
                lg[WORD_TO_ID.get("<unk>", 0)] = lg[WORD_TO_ID.get("<pad>", 1)] = -1e9
                top_indices = torch.topk(lg, 5).indices.tolist()
                ranks = [VOCAB[idx] for idx in top_indices]

                attn_mat = activation_cache.get("real_attn_matrix", None)
                if attn_mat is not None and stp < attn_mat.shape[0]:
                    attn_row = attn_mat[stp, :stp + 1].tolist()
                    attn_weights = [round(w, 3) for w in attn_row]
                else:
                    attn_weights = [1.0]

                send_chunk({
                    "stage": "prefill",
                    "cur": stp + 1,
                    "tot": len(canonical_tks),
                    "tok": VOCAB[canonical_tks[stp]],
                    "ms": max(1, lat_ms),
                    "ram": 140000,
                    "v": v_slice,
                    "ranks": ranks,
                    "attn": attn_weights
                })
                time.sleep(0.06)

            # Phase 2: Autoregressive Generation
            generated_history = []
            max_gen = 35 if is_logic else 55

            for gen_stp in range(max_gen):
                pos = len(curr_tokens) - 1
                padded = curr_tokens + [padId] * (BLOCK_SIZE - len(curr_tokens))
                x = torch.tensor([padded[:BLOCK_SIZE]], dtype=torch.long, device=device)
                
                with torch.no_grad():
                    logits = model(x)
                    
                lg = logits[0, pos].clone()
                lg[WORD_TO_ID.get("<unk>", 0)] = lg[WORD_TO_ID.get("<pad>", 1)] = -1e9

                if is_logic:
                    pred_id = lg.argmax().item()
                else:
                    # STORY MODE CONTROLS: Temperature, Repetition Penalty & Min-Length Guard
                    for past_tok in set(curr_tokens[-12:] + generated_history[-10:]):
                        lg[past_tok] -= 1.8  # Repetition Penalty prevents `. he . he .` loops
                    
                    if gen_stp < 12:
                        lg[WORD_TO_ID.get("<eos>", 2)] = -1e9  # Keep story going for at least 12 words

                    # Temperature scaling & Top-K (K=25) sampling
                    probs = F.softmax(lg / 0.75, dim=-1)
                    top_k_probs, top_k_indices = torch.topk(probs, 25)
                    sample_idx = torch.multinomial(top_k_probs, 1).item()
                    pred_id = top_k_indices[sample_idx].item()

                word = VOCAB[pred_id]
                lat_ms = int((time.perf_counter() - t_prev) * 1000)
                t_prev = time.perf_counter()

                tv = activation_cache.get("thought_vector", None)
                v_slice = tv[pos, :80].tolist() if tv is not None else [0.0] * 80

                attn_mat = activation_cache.get("real_attn_matrix", None)
                if attn_mat is not None and pos < attn_mat.shape[0]:
                    attn_row = attn_mat[pos, :pos + 1].tolist()
                    attn_weights = [round(w, 3) for w in attn_row]
                else:
                    attn_weights = [0.1] * len(curr_tokens)

                top_indices = torch.topk(lg, 5).indices.tolist()
                ranks = [VOCAB[idx] for idx in top_indices]

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
                generated_history.append(pred_id)
                if len(curr_tokens) >= BLOCK_SIZE:
                    break

                time.sleep(0.12)

            self.wfile.write(b"0\r\n\r\n")

    def log_message(self, format, *args):
        return

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8080), SimulatorHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[!] Simulator server stopped.")