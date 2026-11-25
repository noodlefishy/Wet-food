import json, math, os, re, sys
from collections import Counter

projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, projectRoot)

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "to", "of", "from",
    "and", "or", "not", "it", "then", "if", "who", "what", "where", "when",
    "why", "how", "does", "did", "do", "many", "items", "have", "has", "had",
    "now", "first", "?", ".", ",", ":", "!"
}

TOKEN_REGEX = r"\[[cqa]\]|<eos>|<unk>|<pad>|'s|\w+|[.,?!:]"

def tokenize_words(s):
    s = s.lower()
    for c in (".", "?", "!", ",", ":"):
        s = s.replace(c, f" {c} ")
    s = s.replace("'s", " 's ")
    return re.findall(TOKEN_REGEX, s)

def stem_term(w):
    if len(w) > 3 and w.endswith("s") and w not in ("is", "was", "has", "does"):
        return w[:-1]
    return w

def content_terms(text):
    return [stem_term(w) for w in tokenize_words(text) if w not in STOPWORDS and w.isalpha()]

class MicroBM25:
    def __init__(self, facts):
        self.facts = [f.strip().rstrip(". ") for f in facts if f.strip()]
        self.docs = [content_terms(f) for f in self.facts]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.N)
        self.df = Counter()
        for d in self.docs:
            for w in set(d):
                self.df[w] += 1

    def score(self, query_terms, doc_terms, k1=1.5, b=0.75):
        if not doc_terms:
            return 0.0
        tf = Counter(doc_terms)
        dl = len(doc_terms)
        s = 0.0
        for q in query_terms:
            if q not in tf:
                continue
            df_q = self.df[q]
            idf = math.log((self.N - df_q + 0.5) / (df_q + 0.5) + 1.0)
            freq = tf[q]
            s += idf * (freq * (k1 + 1.0)) / (freq + k1 * (1.0 - b + b * dl / max(1.0, self.avgdl)))
        return s

    def retrieve(self, question, top_k=2, two_hop=True):
        q_words = tokenize_words(question)
        q_terms = content_terms(question)
        base_scores = [self.score(q_terms, d) for d in self.docs]
        ranked = sorted(range(self.N), key=lambda i: base_scores[i], reverse=True)

        chosen = []
        if ranked and base_scores[ranked[0]] > 0:
            chosen.append(ranked[0])

        # Fallback: if no query content term matched anywhere in facts.txt,
        # pull 2 unrelated background facts so the model enters abstention mode.
        if not chosen:
            return [self.facts[i] for i in range(min(2, self.N))]

        if two_hop and top_k > 1:
            bridge_terms = [w for w in self.docs[chosen[0]] if w not in q_terms]
            is_where_q = "where" in q_words and len(q_terms) == 1
            best_hop2_idx, best_hop2_score = -1, 0.0

            for idx in range(self.N):
                if idx in chosen:
                    continue
                q_sc = base_scores[idx]
                b_sc = self.score(bridge_terms, self.docs[idx])
                raw_doc = self.facts[idx]
                has_logic_rule = any(w in raw_doc for w in ("things", "if ", "cat"))
                is_spatial_rel = any(w in raw_doc for w in ("north", "south", "bigger", "smaller", "left of"))

                if q_sc > 0 or (b_sc > 0 and (has_logic_rule or (is_spatial_rel and not is_where_q))):
                    comb = 2.0 * q_sc + b_sc
                    if comb > best_hop2_score:
                        best_hop2_score = comb
                        best_hop2_idx = idx
            if best_hop2_idx != -1:
                chosen.append(best_hop2_idx)

        return [self.facts[i] for i in chosen[:top_k]]

def build_rag_prompt(question, retriever, word_to_id=None, block_size=64, reserve_gen=14):
    q_clean = question.strip().rstrip("?.: ")
    q_toks = tokenize_words(f"[q] {q_clean} ?")
    max_ctx_tokens = block_size - len(q_toks) - 3 - reserve_gen

    facts = retriever.retrieve(q_clean, top_k=2, two_hop=True)
    kept_facts = []
    used_tokens = 0
    for fact in facts:
        ftoks = tokenize_words(fact + " .")
        if used_tokens + len(ftoks) <= max_ctx_tokens:
            kept_facts.append(fact)
            used_tokens += len(ftoks)

    context_str = " . ".join(kept_facts) if kept_facts else "nobody"
    prompt = f"[q] {q_clean} ? [c] {context_str} . [a]"

    unks = [w for w in tokenize_words(prompt) if word_to_id and w not in word_to_id]
    return prompt, kept_facts, unks

BENCH_CASES = [
    ("where is alice", "kitchen"),
    ("where is bob", "garden"),
    ("where is lucy", "kitchen"),
    ("where is max", "bedroom"),
    ("where is felix", "forest"),
    ("where is the coin", "basement"),
    ("where is the rock", "yard"),
    ("what colour is the spoon", "green"),
    ("who likes the toy", "simba"),
    ("is the cup hot", "yes"),
    ("is the plate cold", "yes"),
    ("is the jar closed", "yes"),
    ("what is north of the shed", "castle"),
    ("where is charlie", "nowhere"),
    ("who has the pencil", "nobody"),
]

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    run_bench = "--bench" in sys.argv

    facts_path = os.path.join(projectRoot, "facts.txt")
    with open(facts_path, "r") as f:
        facts = f.readlines()

    with open(os.path.join(projectRoot, "config", "model.json")) as f:
        cfg = json.load(f)
    block_size = cfg["architecture"]["block_size"]

    if dry_run:
        from models.local import vocab
        WORD_TO_ID = {w: i for i, w in enumerate(vocab.VOCAB)}
        model = None
        print(f"[*] Loaded {len(facts)} facts from facts.txt (--dry-run mode).")
    else:
        import torch
        from benchmarks.fastEvaluator import model, VOCAB, WORD_TO_ID, BLOCK_SIZE, padId, eosId, device
        block_size = BLOCK_SIZE
        model.eval()
        print(f"[*] Loaded model on {device.upper()} | {len(facts)} facts from facts.txt.")

    retriever = MicroBM25(facts)

    def answer_question(q):
        prompt, picked, unks = build_rag_prompt(q, retriever, WORD_TO_ID, block_size=block_size)
        if dry_run or model is None:
            return None, None, prompt, picked, unks
        import torch
        toks = [WORD_TO_ID.get(w, 0) for w in tokenize_words(prompt)]
        padded = toks + [padId] * (block_size - len(toks))
        x = torch.tensor([padded[:block_size]], dtype=torch.long, device=device)
        out = []
        with torch.no_grad():
            for step in range(16):
                pos = len(toks) + step - 1
                if pos >= block_size - 1:
                    break
                lg = model(x)[0, pos].clone()
                lg[WORD_TO_ID["<unk>"]] = lg[WORD_TO_ID["<pad>"]] = -1e9
                pred = lg.argmax().item()
                if pred in (eosId, padId):
                    break
                out.append(VOCAB[pred])
                x[0, pos + 1] = pred
        raw = " ".join(out)
        clean = raw.strip().rstrip(". ")
        final = clean.split(":")[-1].strip().rstrip(". ") if ":" in clean else clean
        return final, raw, prompt, picked, unks

    if run_bench:
        passed = 0
        print("\n" + "=" * 68)
        print("            END-TO-END RAG BENCHMARK (facts.txt)")
        print("=" * 68)
        for q, expected in BENCH_CASES:
            ans, raw, prompt, picked, unks = answer_question(q)
            if dry_run:
                print(f"Q: {q:<26} -> {picked} | unks={unks}")
            else:
                ok = ans == expected
                passed += int(ok)
                mark = "✓" if ok else "✗"
                print(f"  [{mark}] Q: {q:<24} | Exp: {expected:<8} | Got: {ans:<8} | Raw: {raw}")
        if not dry_run:
            print("=" * 68)
            print(f" RAG END-TO-END SCORE: {passed}/{len(BENCH_CASES)} ({passed/len(BENCH_CASES)*100:.1f}%)")
            print("=" * 68)
    else:
        for q, _ in BENCH_CASES[:6]:
            ans, raw, prompt, picked, unks = answer_question(q)
            print(f"\nQ: {q}")
            print(f"  Retrieved : {picked}")
            print(f"  Prompt    : {prompt}")
            if unks:
                print(f"  [!] OOV words: {unks}")
            if raw is not None:
                print(f"  Answer    : {raw}")