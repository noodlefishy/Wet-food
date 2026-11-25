import os, sys, torch

projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, projectRoot)

from benchmarks.fastEvaluator import model, VOCAB, WORD_TO_ID, BLOCK_SIZE, padId, eosId, device
from benchmarks.ragDemo import MicroBM25, build_rag_prompt, tokenize_words

model.eval()

def ask_rag(question, retriever):
    prompt, picked, unks = build_rag_prompt(question, retriever, WORD_TO_ID, block_size=BLOCK_SIZE)
    toks = [WORD_TO_ID.get(w, 0) for w in tokenize_words(prompt)]
    padded = toks + [padId] * (BLOCK_SIZE - len(toks))
    x = torch.tensor([padded[:BLOCK_SIZE]], dtype=torch.long, device=device)

    out = []
    with torch.no_grad():
        for step in range(18):
            pos = len(toks) + step - 1
            if pos >= BLOCK_SIZE - 1:
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
    return final, raw, picked, unks

def run_section(title, cases, retriever):
    passed = 0
    print("\n" + "=" * 72)
    print(f" {title}")
    print("=" * 72)
    for q, expected in cases:
        ans, raw, picked, unks = ask_rag(q, retriever)
        ok = ans == expected
        passed += int(ok)
        mark = "✓" if ok else "✗"
        unk_note = f" | UNKS={unks}" if unks else ""
        print(f"  [{mark}] Q: {q:<34} | Exp: {expected:<8} | Got: {ans:<8}")
        if not ok:
            print(f"      Retrieved: {picked}")
            print(f"      Raw Out  : {raw}{unk_note}")
    print(f"  -> Section Score: {passed}/{len(cases)}")
    return passed, len(cases)

if __name__ == "__main__":
    with open(os.path.join(projectRoot, "facts.txt"), "r") as f:
        base_facts = [line.strip() for line in f if line.strip()]

    # 1. Deep facts.txt test (untested facts, inverses, multi-hop chains)
    ret_base = MicroBM25(base_facts)
    SECTION_1 = [
        ("where is oscar", "porch"),
        ("where is zoe", "balcony"),
        ("is zoe in the yard", "no"),
        ("where is daisy now", "library"),
        ("where was toby before the castle", "cave"),
        ("where is jack", "school"),
        ("who has the key", "zoe"),
        ("how many items does lily have", "three"),
        ("is the ball dirty", "yes"),
        ("is the box bigger than the blanket", "yes"),
        ("is the blanket bigger than the box", "no"),
        ("can you go from the hallway to the balcony", "yes"),
        ("what is south of the castle", "shed"),
    ]

    # 2. Counterfactual mutation test (proves model obeys RAG, not static priors)
    mutated_facts = list(base_facts)
    replacements = {
        "bob went to the garden": "bob went to the library",
        "if oscar is thirsty then oscar is in the porch": "if oscar is thirsty then oscar is in the attic",
        "alice gave the key to zoe": "alice gave the key to max",
        "felix is not in the park": "felix is not in the forest",
        "if the spoon is in the garage then it is green": "if the spoon is in the garage then it is yellow",
        "simba is a cat": "leo is a cat",
    }
    mutated_facts = [replacements.get(line, line) for line in mutated_facts]
    ret_mut = MicroBM25(mutated_facts)
    SECTION_2 = [
        ("where is bob", "library"),
        ("where is oscar", "attic"),
        ("who has the key", "max"),
        ("where is felix", "park"),
        ("what colour is the spoon", "yellow"),
        ("who likes the toy", "leo"),
    ]

    # 3. Brand-new unseen entities (in 2048 VOCAB via TinyStories, never in world.json)
    UNSEEN_WORLD = [
        "harry went to the farm",
        "claire found the apple in the island",
        "then claire went to the airport",
        "henry is in the office or the bathroom",
        "henry is not in the office",
        "george put the cheese in the bathroom",
        "if the cheese is in the bathroom then it is cold",
        "george is a cat",
        "cats like the milk",
    ]
    ret_unseen = MicroBM25(UNSEEN_WORLD)
    SECTION_3 = [
        ("where is harry", "farm"),
        ("where is the apple now", "airport"),
        ("where is henry", "bathroom"),
        ("is the cheese cold", "yes"),
        ("who likes the milk", "george"),
    ]

    p1, t1 = run_section("1. DEEP facts.txt COVERAGE (13 CASES)", SECTION_1, ret_base)
    p2, t2 = run_section("2. COUNTERFACTUAL MUTATION TEST (PROVES LIVE RAG)", SECTION_2, ret_mut)
    p3, t3 = run_section("3. BRAND-NEW UNSEEN ENTITIES KB (OOD VIA RAG)", SECTION_3, ret_unseen)

    tot_p, tot_t = p1 + p2 + p3, t1 + t2 + t3
    print("\n" + "=" * 72)
    print(f" TOTAL ACTIVE-RAG SCORE: {tot_p}/{tot_t} ({tot_p/tot_t*100:.1f}%)")
    print("=" * 72)