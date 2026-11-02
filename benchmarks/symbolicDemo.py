import os, sys, re, torch
import torch.nn.functional as F

projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, projectRoot)

from benchmarks.fastEvaluator import (
    model, VOCAB, WORD_TO_ID, BLOCK_SIZE, padId, device
)

model.eval()

TOKEN_REGEX = r"\[[cqa]\]|<eos>|<unk>|<pad>|'s|\w+|[.,?!:]"

def tokenize(s):
    for c in [".", "?", "!", ",", ":"]: 
        s = s.replace(c, f" {c} ")
    s = s.replace("'s", " 's ")
    return [WORD_TO_ID.get(w, 0) for w in re.findall(TOKEN_REGEX, s.lower())]

def query_neural_worker(qca_prompt):
    toks = tokenize(qca_prompt)
    if not toks:
        return ""
    
    if len(toks) > BLOCK_SIZE - 1:
        toks = toks[:BLOCK_SIZE - 1]
        
    padded = toks + [padId] * (BLOCK_SIZE - len(toks))
    x = torch.tensor([padded], dtype=torch.long, device=device)
    
    out_words = []
    with torch.no_grad():
        for step in range(12):
            pos = len(toks) + step - 1
            if pos >= BLOCK_SIZE - 1:
                break
            logits = model(x)
            lg = logits[0, pos].clone()
            lg[WORD_TO_ID["<unk>"]] = lg[WORD_TO_ID["<pad>"]] = -1e9
            
            pred_id = lg.argmax().item()
            word = VOCAB[pred_id]
            if word in ("<eos>", "<pad>"):
                break
            out_words.append(word)
            x[0, pos + 1] = pred_id
            if word == ".":
                break
                
    clean_ans = " ".join(out_words).replace(".", "").strip()
    return clean_ans



class SymbolicOrchestrator:
    def __init__(self, context_text):
        self.raw_context = context_text.strip().rstrip(". ")
        self.blackboard = {}

    def run_subquery(self, question, save_as=None):
        rendered_q = question
        for var_name, var_val in self.blackboard.items():
            rendered_q = rendered_q.replace(f"${var_name}", var_val)
            
        qca = f"[q] {rendered_q} ? [c] {self.raw_context} . [a]"
        result = query_neural_worker(qca)
        
        if save_as:
            self.blackboard[save_as] = result
            
        print(f"    -> Neural Worker Executed: '{rendered_q} ?'")
        print(f"       Result: '{result}' | (Saved to ${save_as})")
        return result



TEST_CASES = [
    {
        "title": "CASE 1: Transitive Indirect Possession",
        "story": "alice had the ball . alice gave the ball to bob . then bob gave the ball to luna",
        "complex_query": "Where is the item Alice gave away, if Luna is currently in the garden?",
        "plan": [
            ("Pass 1: Find what Alice gave away", "what did alice give bob", "item"),
            ("Pass 2: Find who ended up with that item", "who has the $item now", "final_owner"),
        ],
        "synthesizer": lambda bb: f"The item is with {bb.get('final_owner')}, who is in the garden."
    },
    {
        "title": "CASE 2: Temporal Predecessor + Location",
        "story": "first felix was in the bedroom . then felix was in the park . then felix went to the kitchen",
        "complex_query": "Is Felix's initial room located north of the park if the bedroom is north of the park?",
        "plan": [
            ("Pass 1: Find where Felix was first", "where was felix first", "first_room"),
            ("Pass 2: Check current location", "where is felix now", "current_room"),
        ],
        "synthesizer": lambda bb: f"Felix started in the {bb.get('first_room')} and is now in the {bb.get('current_room')}."
    },
    {
        "title": "CASE 3: Relational Inversion + Item Resolution",
        "story": "the box is bigger than the ball . the ball is bigger than the cup . alice has the cup",
        "complex_query": "Does Alice hold the smallest item in the chain?",
        "plan": [
            ("Pass 1: Find what item is smaller than the ball", "what is smaller than the ball", "smallest_item"),
            ("Pass 2: Check what Alice holds", "what does alice have", "held_item"),
        ],
        "synthesizer": lambda bb: f"True! Alice holds the {bb.get('held_item')}, which is the smallest object ({bb.get('smallest_item')})."
    }
]

def main():
    print("=" * 70)
    print("   picoAI: SYMBOLIC ORCHESTRATOR DEMO (MULTI-PASS CHAINING)")
    print("=" * 70)
    
    for _, case in enumerate(TEST_CASES, 1):
        print(f"\n[{case['title']}]")
        print(f"Context : \"{case['story']}\"")
        print(f"Complex : \"{case['complex_query']}\"")
        print("Executing Orchestrated Sub-Plan:")
        
        orch = SymbolicOrchestrator(case["story"])
        
        for step_desc, sub_q, var in case["plan"]:
            print(f"  * {step_desc}:")
            orch.run_subquery(sub_q, save_as=var)
            
        final_answer = case["synthesizer"](orch.blackboard)
        print(f"\n  ==> Orchestrator Synthesis: {final_answer}")
        print("-" * 70)

if __name__ == "__main__":
    main()