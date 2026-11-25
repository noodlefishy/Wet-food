import re

TOKEN_REGEX = r"\[[cqa]\]|<eos>|<unk>|<pad>|'s|\w+|[.,?!:]"

def tokenize(text, word_to_id, unk_id=0):
    s = text.lower()
    for c in (".", "?", "!", ",", ":"):
        s = s.replace(c, f" {c} ")
    s = s.replace("'s", " 's ")
    return [word_to_id.get(w, unk_id) for w in re.findall(TOKEN_REGEX, s)]

def pluralize(word, count=2):
    if count == 1:
        return word
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"

def canonicalize_to_qca(raw):
    p = raw.strip()
    is_logic = (
        any(t in p for t in ("[c]", "[q]", "[a]"))
        or any(p.lower().startswith(w) for w in ("who", "what", "where", "is", "does", "why", "how", "question"))
        or ("?" in p)
    )
    if not is_logic:
        return p, False

    if "[q]" in p and "[c]" in p:
        q_pos = p.find("[q]")
        c_pos = p.find("[c]")
        if q_pos < c_pos:
            if not p.endswith("[a]"):
                p = p.rstrip("?.: ") + " [a]"
            return p, True
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
            if second:
                return f"[q] {first} ? [c] {second} . [a]", True
            return f"[q] {first} ? [a]", True
        return f"[q] {second} ? [c] {first} . [a]", True

    if not p.startswith("[q]"):
        p = f"[q] {p}"
    if not p.endswith("[a]"):
        p = p.rstrip("?.: ") + " ? [a]"
    return p, True