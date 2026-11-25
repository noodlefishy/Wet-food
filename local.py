import json, math, os, socket, sys, time, urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
import torch
import torch.nn.functional as F

projectRoot = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, projectRoot)

from pico.model import apply_rope, load_distributed_model
from pico.tokenizer import canonicalize_to_qca, tokenize

ctx = load_distributed_model(projectRoot)
model = ctx["model"]
VOCAB = ctx["vocab"]
WORD_TO_ID = ctx["word_to_id"]
VOCAB_SIZE = ctx["vocab_size"]
EMBED_DIM = ctx["embed_dim"]
BLOCK_SIZE = ctx["block_size"]
NUM_HEADS = ctx["num_heads"]
HEAD_DIM = ctx["head_dim"]
NUM_LAYERS = ctx["num_layers"]
MEOW_LAYERS = ctx["meow_layers"]
device = ctx["device"]
padId = WORD_TO_ID.get("<pad>", 1)

print("=" * 65)
print(f"[*] Loaded PyTorch Model ({EMBED_DIM}-D, {NUM_LAYERS}L, {NUM_HEADS} Heads, dk={HEAD_DIM}) on {str(device).upper()}")
print("[*] Starting Local Neon Cyber-Terminal at http://localhost:8080")
print("=" * 65)

activation_cache = {
    "thought_vector": None,
    "layers_attn": [None] * NUM_LAYERS,
}

def make_attn_hook(l_idx):
    def hook(module, inp, out):
        x = inp[0]
        B, T, _ = x.shape
        q, k, _ = module.qkvProj(x).chunk(3, dim=-1)
        q = apply_rope(q.view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2), module.cos_cached, module.sin_cached)
        k = apply_rope(k.view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2), module.cos_cached, module.sin_cached)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        activation_cache["layers_attn"][l_idx] = F.softmax(scores, dim=-1)[0].detach().cpu()
    return hook

def meow_boundary_hook(module, inp, out):
    activation_cache["thought_vector"] = out[0].detach()

model.layers[MEOW_LAYERS - 1].register_forward_hook(meow_boundary_hook)
for idx in range(NUM_LAYERS):
    model.layers[idx].sa.register_forward_hook(make_attn_hook(idx))

def compute_attention_rollout(layers_attn_list, seq_len):
    valid_layers = [l for l in layers_attn_list if l is not None]
    if not valid_layers or seq_len <= 0:
        return torch.zeros((seq_len, seq_len))

    dev = valid_layers[0].device
    averaged_layers = [
        (l.mean(dim=0) if l.ndim == 3 else l)[:seq_len, :seq_len].to(dev)
        for l in valid_layers
    ]
    peak_matrix, _ = torch.stack(averaged_layers, dim=0).max(dim=0)
    peak_matrix = peak_matrix * torch.tril(torch.ones((seq_len, seq_len), device=dev))
    return (peak_matrix / peak_matrix.sum(dim=-1, keepdim=True).clamp(min=1e-9)).detach().cpu()

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

            default_temp = 0.0 if is_logic else 0.75
            default_top_k = 1 if is_logic else 25
            try:
                temp_param = float(query.get("t", [default_temp])[0])
            except (ValueError, TypeError):
                temp_param = default_temp
            try:
                top_k_param = int(query.get("k", [default_top_k])[0])
            except (ValueError, TypeError):
                top_k_param = default_top_k

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
                            "heads": [[round(w, 3) for w in h_row] for h_row in head_rows],
                        })
                    else:
                        all_l.append({
                            "all": [0.0] * max_len,
                            "heads": [[0.0] * max_len for _ in range(NUM_HEADS)],
                        })
                return all_l

            canonical_tks = tokenize(qca_prompt, WORD_TO_ID)
            user_tks = tokenize(raw_prompt, WORD_TO_ID)
            if not canonical_tks:
                send_chunk({"stage": "error", "msg": "Empty prompt"})
                try:
                    self.wfile.write(b"0\r\n\r\n")
                except Exception:
                    pass
                return

            if not send_chunk({
                "stage": "init",
                "user_prompt": raw_prompt,
                "qca_tokens": [VOCAB[t] for t in canonical_tks],
                "user_tokens": [VOCAB[t] for t in user_tks],
                "is_logic": is_logic,
            }):
                return

            curr_tokens = list(canonical_tks)
            t_prev = time.perf_counter()

            for stp in range(len(canonical_tks)):
                prefix = curr_tokens[: stp + 1]
                padded = prefix + [padId] * (BLOCK_SIZE - len(prefix))
                x = torch.tensor([padded[:BLOCK_SIZE]], dtype=torch.long, device=device)

                with torch.no_grad():
                    logits = model(x)

                lat_ms = int((time.perf_counter() - t_prev) * 1000)
                t_prev = time.perf_counter()

                tv = activation_cache.get("thought_vector")
                v_slice = tv[stp, : min(96, EMBED_DIM)].tolist() if tv is not None else [0.0] * min(96, EMBED_DIM)

                lg = logits[0, stp].clone()
                lg[WORD_TO_ID.get("<unk>", 0)] = lg[WORD_TO_ID.get("<pad>", 1)] = -1e9
                ranks = [VOCAB[idx] for idx in torch.topk(lg, 5).indices.tolist()]

                curr_len = stp + 1
                R_mat = compute_attention_rollout(activation_cache["layers_attn"], curr_len)
                rollout_row = (R_mat[-1, :curr_len] if R_mat.ndim == 2 else R_mat).tolist()

                if not send_chunk({
                    "stage": "prefill",
                    "cur": stp + 1,
                    "tot": len(canonical_tks),
                    "tok": VOCAB[canonical_tks[stp]],
                    "ms": max(1, lat_ms),
                    "ram": 140000,
                    "v": v_slice,
                    "ranks": ranks,
                    "attn_layers": extract_all_layers_attn(stp, stp + 1),
                    "rollout": rollout_row,
                }):
                    return
                time.sleep(0.08)

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
                    top_k_probs, top_k_indices = torch.topk(probs, min(max(1, top_k_param), VOCAB_SIZE))
                    pred_id = top_k_indices[torch.multinomial(top_k_probs, 1).item()].item()

                word = VOCAB[pred_id]
                if word in ("<eos>", "<pad>"):
                    break

                lat_ms = int((time.perf_counter() - t_prev) * 1000)
                t_prev = time.perf_counter()

                tv = activation_cache.get("thought_vector")
                v_slice = tv[pos, : min(96, EMBED_DIM)].tolist() if tv is not None else [0.0] * min(96, EMBED_DIM)
                ranks = [VOCAB[idx] for idx in torch.topk(lg, 5).indices.tolist()]

                curr_len = len(curr_tokens)
                R_mat = compute_attention_rollout(activation_cache["layers_attn"], curr_len)
                rollout_row = (R_mat[-1, :curr_len] if R_mat.ndim == 2 else R_mat).tolist()

                if not send_chunk({
                    "stage": "gen",
                    "w": word,
                    "ms": max(1, lat_ms),
                    "ram": 140000,
                    "v": v_slice,
                    "ranks": ranks,
                    "attn_layers": extract_all_layers_attn(pos, len(curr_tokens)),
                    "rollout": rollout_row,
                }):
                    return

                curr_tokens.append(pred_id)
                generated_history.append(pred_id)
                if len(curr_tokens) >= BLOCK_SIZE:
                    break
                time.sleep(0.16)

            try:
                self.wfile.write(b"0\r\n\r\n")
            except Exception:
                pass

    def log_message(self, format, *args):
        return

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8080), SimulatorHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[!] Simulator server stopped.")