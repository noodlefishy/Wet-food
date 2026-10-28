import os, sys, json, re

# Add project root to sys.path to import vocab
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from firmware.vocab import VOCAB



# 1. Base Semantic Seeds & Rules
SEEDS = {
    "Mythical & Royalty": {
        "dragon", "castle", "magic", "magical", "treasure", "monster", "giant",
        "fairy", "creature", "tower", "king", "prince", "princess", "queen",
        "wings", "flying", "secret", "sparkly", "rainbow", "dream"
    },
    "Creatures & Wildlife": {
        "cat", "cats", "kitten", "kitty", "dog", "puppy", "bird", "birds",
        "butterfly", "duck", "ducks", "lion", "frog", "squirrel", "bunny",
        "rabbit", "mouse", "spider", "bear", "elephant", "fox", "ant", "bee",
        "owl", "cow", "horse", "snake", "goat", "sheep", "wolf", "swan"
    },
    "Sensory, Food & Comfort": {
        "honey", "cookies", "cake", "candy", "cheese", "apple", "apples",
        "soup", "carrot", "carrots", "jam", "berries", "sugar", "bread", "oven",
        "juice", "tea", "picnic", "snack", "food", "drink", "dinner", "lunch",
        "milk", "fish", "blanket", "pillow", "bath", "bubbles", "mittens", "bed", "yummy"
    },
    "Atmosphere & Settings": {
        "garden", "kitchen", "park", "room", "house", "forest", "woods", "trees",
        "tree", "river", "lake", "ocean", "sea", "pond", "beach", "cave", "yard",
        "jungle", "zoo", "hill", "ground", "path", "field", "town", "village",
        "storm", "clouds", "cloud", "rain", "wind", "snow", "snowman", "sun",
        "sunny", "sunshine", "sky"
    },
    "Mischief & Chaos": {
        "clown", "screamed", "smelly", "naughty", "mud", "messy", "fake", "silly",
        "accidentally", "broke", "bite", "ugly", "rude", "spicy", "danger",
        "dangerous", "fighting", "trouble", "puddle", "dirt", "noise", "loud"
    },
    "Props, Gear & Toys": {
        "ball", "box", "book", "books", "cup", "key", "bag", "toy", "toys",
        "wagon", "kite", "balloon", "boat", "train", "truck", "bike", "umbrella",
        "drum", "doll", "dolls", "pot", "spoon", "bucket", "basket", "bottle",
        "jar", "shoes", "clothes", "dress", "shirt", "rope", "present", "gift",
        "blocks", "paper"
    }
}

VERB_SEEDS = {
    "jump", "climb", "fly", "run", "swim", "dance", "hop", "sing", "shout",
    "chase", "fall", "throw", "grab", "push", "pull", "catch", "bite", "hide",
    "explore", "walk", "pick", "open", "start", "stop", "turn", "roll", "skip",
    "drop", "blow", "scream", "eat", "play", "see", "look", "smile", "laugh"
}

ADJ_SEEDS = {
    "sparkly", "shiny", "bright", "dark", "huge", "giant", "tiny", "small",
    "big", "tall", "soft", "fluffy", "sharp", "spicy", "smelly", "clean", "dirty",
    "hot", "cold", "warm", "wet", "dry", "heavy", "light", "sweet", "delicious",
    "tasty", "yummy", "loud", "quiet", "fast", "wild", "strange", "silly", "funny",
    "brave", "peaceful", "cozy", "empty", "full"
}

# 2. Filter & Classify VOCAB
classified = {cat: [] for cat in SEEDS}
classified["Action Verbs"] = []
classified["Sensory Adjectives"] = []

CONTROL_TOKENS = {"<unk>", "<pad>", "<eos>", ".", "?", ":", ",", "[c]", "[q]", "[a]", "question", "answer", "endoftext"}

for word in VOCAB:
    w = word.lower().strip()
    if w in CONTROL_TOKENS or len(w) <= 1:
        continue

    # Noun categories
    matched = False
    for cat, seed_words in SEEDS.items():
        if w in seed_words:
            classified[cat].append(w)
            matched = True
            break
    if matched:
        continue

    # Action Verbs (morphology + seeds)
    if w in VERB_SEEDS or any(w.startswith(stem) for stem in ["jump", "climb", "run", "swim", "play", "danc", "walk", "hop", "scream", "shout", "chase", "push", "pull", "sing"]):
        classified["Action Verbs"].append(w)
    elif w.endswith("ing") or (w.endswith("ed") and not w in ADJ_SEEDS):
        classified["Action Verbs"].append(w)

    # Sensory Adjectives (morphology + seeds)
    elif w in ADJ_SEEDS or (w.endswith("y") and len(w) > 3 and not w.endswith("ly")) or w.endswith("ful"):
        classified["Sensory Adjectives"].append(w)

# Clean and dedup
for cat in classified:
    classified[cat] = sorted(list(set(classified[cat])))

# Output results
output_path = os.path.join(os.path.dirname(__file__), "..", "firmware", "catalysts.json")
with open(output_path, "w") as f:
    json.dump(classified, f)

print("\n" + "="*60)
print("     CATALYST VOCABULARY GENERATION COMPLETE     ")
print("="*60)
for cat, words in classified.items():
    print(f"{cat:<28} : {len(words):3d} words (e.g. {', '.join(words[:4])})")
print(f"\n[✓] Saved to: {output_path}")