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

        if not chosen:
            return []

        top_fact = self.facts[chosen[0]]
        is_where_q = "where" in q_words and len(q_terms) == 1
        if is_where_q and " went to " in top_fact and " or " not in top_fact and not top_fact.startswith("if "):
            return [top_fact]

        if two_hop and top_k > 1:
            bridge_terms = [w for w in self.docs[chosen[0]] if w not in q_terms]
            best_hop2_idx, best_hop2_score = -1, 0.0

            for idx in range(self.N):
                if idx in chosen:
                    continue
                q_sc = base_scores[idx]
                b_sc = self.score(bridge_terms, self.docs[idx])
                raw_doc = self.facts[idx]
                has_logic_rule = any(w in raw_doc for w in ("things", "if ", "cat"))
                is_spatial_rel = any(w in raw_doc for w in ("north", "south", "bigger", "smaller", "left of", "leads to"))

                # For spatial relations, follow the bridge token even if direct query score is zero!
                if is_spatial_rel and b_sc > 0:
                    weight = 3.0 * b_sc + q_sc
                    if weight > best_hop2_score:
                        best_hop2_score = weight
                        best_hop2_idx = idx
                elif q_sc > 0 or (b_sc > 0 and has_logic_rule):
                    comb = 2.0 * q_sc + b_sc
                    if comb > best_hop2_score:
                        best_hop2_score = comb
                        best_hop2_idx = idx

            if best_hop2_idx != -1:
                chosen.append(best_hop2_idx)

        # Return facts in natural document order so chronological sequences stay intact
        return [self.facts[i] for i in sorted(chosen[:top_k])]

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