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

def content_terms(text):
    return [w for w in tokenize_words(text) if w not in STOPWORDS and w.isalpha()]

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
        q_terms = content_terms(question)
        base_scores = [self.score(q_terms, d) for d in self.docs]
        ranked = sorted(range(self.N), key=lambda i: base_scores[i], reverse=True)

        chosen = []
        if ranked and base_scores[ranked[0]] > 0:
            chosen.append(ranked[0])

        # Only expand Hop-2 when a 2nd fact shares a query term OR bridges a rule/relation
        if two_hop and chosen and top_k > 1:
            bridge_terms = [w for w in self.docs[chosen[0]] if w not in q_terms]
            best_hop2_idx, best_hop2_score = -1, 0.0
            for idx in range(self.N):
                if idx in chosen:
                    continue
                q_sc = base_scores[idx]
                b_sc = self.score(bridge_terms, self.docs[idx])
                # Require either direct query overlap OR strong bridge + rule keyword
                has_rule = any(w in ("things", "if", "north", "south", "bigger", "smaller", "left", "right", "cats") for w in self.docs[idx])
                is_direct_where = q_terms and len(q_terms) == 1 and "where" in tokenize_words(question)
                if q_sc > 0 or (b_sc > 0 and has_rule and not is_direct_where):
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
    max_ctx_tokens = block_size - len(q_toks) - 3 - reserve_gen  # [c], ., [a]

    facts = retriever.retrieve(q_clean, top_k=3, two_hop=True)
    kept_facts = []
    used_tokens = 0
    for fact in facts:
        ftoks = tokenize_words(fact + " .")
        if used_tokens + len(ftoks) <= max_ctx_tokens:
            kept_facts.append(fact)
            used_tokens += len(ftoks)

    context_str = " . ".join(kept_facts) if kept_facts else "nobody"
    prompt = f"[q] {q_clean} ? [c] {context_str} . [a]"

    unks = []
    if word_to_id is not None:
        unks = [w for w in tokenize_words(prompt) if w not in word_to_id]
    return prompt, kept_facts, unks

# Sample bAbI-native knowledge base (expand or move to a facts.txt file)
SAMPLE_FACTS = [
    "alice went to the kitchen",
    "bob went to the garden",
    "felix is a cat",
    "cats like the ball",
    "alice gave the cup to bob",
    "the cup is red",
    "red things are hot",
    "if the ball is in the garden then it is wet",
    "bob left the ball in the garden",
    "the kitchen is north of the park",
    "the park is north of the bedroom",
]

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    with open(os.path.join(projectRoot, "config", "model.json")) as f:
        cfg = json.load(f)
    block_size = cfg["architecture"]["block_size"]

    if dry_run:
        from models.local import vocab
        WORD_TO_ID = {w: i for i, w in enumerate(vocab.VOCAB)}
        model = None
        print("[*] Running RAG in --dry-run mode (no weights loaded).")
    else:
        import torch
        from benchmarks.fastEvaluator import model, VOCAB, WORD_TO_ID, BLOCK_SIZE, padId, device
        block_size = BLOCK_SIZE
        model.eval()
        print(f"[*] Loaded model on {device.upper()} (block_size={block_size}).")

    retriever = MicroBM25(SAMPLE_FACTS)
    test_questions = [
        "where is alice",
        "what does felix like",
        "is the cup hot",
        "is the ball wet",
        "what is north of the bedroom",
    ]

    for q in test_questions:
        prompt, picked, unks = build_rag_prompt(q, retriever, WORD_TO_ID, block_size=block_size)
        print(f"\nQ: {q}")
        print(f"  Retrieved : {picked}")
        print(f"  Prompt    : {prompt} ({len(tokenize_words(prompt))} toks)")
        if unks:
            print(f"  [!] OOV words (-> <unk>): {unks}")

        if not dry_run and model is not None:
            import torch
            toks = [WORD_TO_ID.get(w, 0) for w in tokenize_words(prompt)]
            padded = toks + [padId] * (block_size - len(toks))
            x = torch.tensor([padded[:block_size]], dtype=torch.long, device=device)
            out = []
            with torch.no_grad():
                for step in range(14):
                    pos = len(toks) + step - 1
                    if pos >= block_size - 1:
                        break
                    lg = model(x)[0, pos].clone()
                    lg[WORD_TO_ID["<unk>"]] = lg[WORD_TO_ID["<pad>"]] = -1e9
                    pred = lg.argmax().item()
                    w = VOCAB[pred]
                    if w in ("<eos>", "<pad>"):
                        break
                    out.append(w)
                    x[0, pos + 1] = pred
            print(f"  Answer    : {' '.join(out)}")