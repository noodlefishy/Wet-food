# A 2 Node Distributed Autoregressive Nano-Nano-Transformer

```text
       /\_/\                                        /\_/\
      ( o.o )   === [ 384-Byte Latent Vector ] ==> ( >.< )
       > ^ <                                        > ^ <
   MEOW (Master)                            MIAOU (Slave)
 Raspberry Pi Pico WH                     Raspberry Pi Pico W
 Core 0: Layers 0-2                       Core 0: Layers 3-5
 Core 1: HTTP & Telemetry                 Core 0: LM Head (1024 Logits)
```

![video](firmware/Secret%20Demo.mov)

Have you ever wanted to run some amazing (0.87M) autoregressive transformer models across two tiny microcontrollers that beam split layers over wifi? Im so glad you asked!

Meet Meow and Miaou, two horrible electronic kittens made out of custom tensor weights, questionable life choices, love, and standard Microoython. The repo is called **Wet Food** because cats *love* wet food, and this project, you know, feeds raw neural network streams between two cats. So honestly, pretty similar. Anywya....

---

The **Wet Food** model is an experiemtal proof of concept distributed neural interface engine that splits a 6 layer autoregressive transformer across two cheap microcontrollers over a lan inspired by *[AI models may be capable of self-replicating without access to a datacenter](https://www.anthropic.com/research/self-replicating-ai)*.

Utilising non-scary Micropython's standard library (with **ZERO** external C or ML runtime dependancies on the microcontrollers) it executes causual sequence modeling, rotary embeddings and is capable of multi-hop logical deduction (most likely overfitting) directly inside 264kb with an effective 150kb of SRAM in the RP2040.


## Model Card

| Paramater                 | Specification    | Notes                                                 |
| ------------------------- | ---------------- | ----------------------------------------------------- |
| Model Depth               | **6 Layers**     | 3 on Meow, 3 on Miaou                                 |
| Embedding Dimension ($D$) | **96-D**         | It was once 128 (it never worked, RIP)                     |
| Attention Type            | **MHA**          | $d_k = 24$                                            |
| Positional Encoding       | **RoPE**         | Head wise rotation up to 64 Tokens                    |
| Context Window            | **24-Tokens**    | 20 Sliding Window + 4 Sinks                           |
| Vocabulary                | **1 024**        | Whole-word vocabulary with logic tags ([c], [q], [a]) |
| Weight Footprint          | **462kb**        | per board                                             |
| Quantisation              | **INT8 (W8A32)** | Per row INT8 weights paired with FP32 scale factors   |

## Fun details
- **Hardware:** 2x Raspberry Pi Pico W's (264kb SRAM, 2MB external flash)
- **Splitting:** **Meow** does the token embeddings & layer 1-3 and serves the web UI on a seperate core. **Miaou** does layer 4-6 & the 1024 word language modeling head over wifi
- **Network Handoff:** **384** **bytes** per forward pass over TCP sockets
- **Scary Memory Acorbatics**: with only ~150kb effective heap, matrix multiplication lools run with **zero heap allocations**!! Using only static arrays!
- **Lack of C Dependencies**: Everything is pure stdlib Micropython (*struct*, *math*, *array*, *socket*). No external C modules, no micro-TFLite binaries, just raw math and stubborn willpower, and hating _struct_ AAAHHHH.


## Prompting Wet Food : The question first format (QCA)
Because this autoregressive model can only loop backwards (and I think it helped? QCA was not a control variable). Standard CQA (context-question-answer) ordering (`[c] ... [q] ... [a]`) forxed the model to read facts without knowing what to focus on.

Therefor Wet Food uses a **Question First (QCA)** format:
$$
\text{[q] where is alice ? [c] alice went to the kitchen . [a] kitchen . <eos>}
$$
1. `[q]` Question: Sets the querry target for the KV cache.
2. `[c]` Context: Fact clauses attend back to the question while streaming.
3. `[a]` Answer Anchor: The final token that routes attention to the target answer.

*(fun UI Magic!! The web interface excepts and perfers(!!) standard `[c] ... [q] ... [a]` prompt order. And automatically reogranises them internally for the model!! Very magical stuff)*

## What can this little brain do ? (bAbI Tasks 01 until 20)
Trained on synthetic deduction datasets paired with background TinyStories text, Wet Food can do multi-hop logic across all 20  bAbI tasks
- **Movement & State:** Tracking characters & held objects across rooms.
- **Three Argument Transfers:** Tracking an item hand off across multiple actors (`alice` -> `bob` -> `luna`). 
- **Spatial Inversion:** Using compass bearings (garden north of kitchen $\therefore$ kitchen south of garden).
- **Boolean Elimination:** _Disjunctive syllogisms_ (`un garden or kitchen`, `not in garden` $\therefore$ kitchen).
- **Inventory & Sets:** Counting items held by specific actors.
- **Size Transitivity:** Ordering inequalities ($A > B$ and $B > C \therefore A > C$).
- **Open-Ended Stories:** Try typing `once upon a time` and watch it write a random naritive fable.

## Getting started!!

### 1. Test locally on computer
Run this in your shell
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Hosts local server at http://localhost:8080
python3 local.py
```

### 2. Deploy to the Microcontrollers

```sh
# 1. Add your Wi-Fi details in firmware/wifiConfig.py
# 2. Flash Meow (Host)
python3 deploy.py meow

# 3. Flash Miaou (Slave)
python3 deploy.py miaou
```

Once paired over Wi-Fi, open `http://<meow-ip-address>/` in your browser to run queries on the cluster or use `mpremote` to see the ip address outputed by **meow**


## Grrrr
Made with love & love (didn't write in kotlin because Intelij takes too much disk space)
Also be warned, there is still memory issues where the pico's crash. I really dont want to fix it, the ./local.py is the demo anyway