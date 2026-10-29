# chunks.py - Centralized multi-word chunk tokens & tokenizer helpers
import re

GRAMMAR_CHUNKS = [
    ("once upon a time", "once_upon_a_time"),
    ("who went to the",  "who_went_to_the"),
    ("who is in the",    "who_is_in_the"),
    ("went to the",      "went_to_the"),
    ("walked to the",    "walked_to_the"),
    ("arrived at the",   "arrived_at_the"),
    ("is in the",        "is_in_the"),
    ("was in the",       "was_in_the"),
    ("does not have",    "does_not_have"),
    ("did not have",     "did_not_have"),
    ("is north of",      "is_north_of"),
    ("is south of",      "is_south_of"),
    ("bigger than",      "bigger_than"),
    ("smaller than",     "smaller_than"),
    ("where is",         "where_is"),
    ("how many",         "how_many"),
    ("one day",          "one_day")
]

CHUNK_TOKENS = [token for _, token in GRAMMAR_CHUNKS]

TOKEN_REGEX = r"\[[cqa]\]|<eos>|<unk>|<pad>|'s|[a-zA-Z0-9_]+|[.,?!:]"

def preprocess_chunks(text):
    """Greedily replaces multi-word phrases with atomic chunk tokens."""
    t = text.lower()
    for phrase, token in GRAMMAR_CHUNKS:
        t = t.replace(phrase, f" {token} ")
    for char in [".", "?", "!", ",", ":"]:
        t = t.replace(char, f" {char} ")
    t = t.replace("'s", " 's ")
    return " ".join(t.split())

def decode_token(word_str):
    """Converts internal chunk tokens back to natural English for display."""
    return word_str.replace("_", " ")