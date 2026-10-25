# A 2-Node Distributed Neural Engine on Raspberry Pi Pico W's

Have you ever wanted to run transformer models on two tiny micorcontrollers that distribute layers over wifi? Im so glad you asked!!! Meet **Meow** and **Miaou**, two eletconic kittens made with custom tensor weights, stealing on google colab, a lot of will power and, love and micropython.

Instead of using one board for everything, this project splits a 4-layer transformer model in half. **Meow** does the first half of the brain and hosts the kawaii telemetry screen, **Miaou** handles the second half of the generation over wifi (way more useless than **Meow**).

The repo is called **Wet Food** because cats _love_ wet food, and this project you know, feeds raw neural network streams between two cats, so pretty similar.


---

## What's New (The Brain Expansion & Logic Update!)

The kittens got a serious brain upgrade:
- **Upgraded to 64-D:** Bumped the embedding dimensions from 48-D to 64-D so the attention heads actually have room to think without overwriting their own thoughts.
- **Doubled Context Window (64 Tokens):** The context went from 32 to 64 tokens so they can read long premises without instant amnesia.
- **Two-Hop Transitive Reasoning:** It doesnt just autocomplete stories anymore, it can figure out indirect logic like "Alice is with Bob, Bob is in the kitchen -> Alice is in the kitchen" without being told directly.
- **Theory of Mind & Functional Needs:** Understands that being hungry means wanting food, and being cold means needing a blanket.
- **Zero-Dependency Local Engine:** Added `evaluator.py` which runs the entire 64-D model locally on a Mac or PC using pure standard python. Zero pytorch, zero numpy, pure standard library math and structs.
- **Live Thought Vector Visualizer:** The web screen previews the actual 64-D float numbers flying through the air between Meow and Miaou along with real-time latency and top-5 word candidate rankings.
- **Soft Reset System:** The boards no longer sever their wifi socket after every run, eliminating the annoying "Miaou is sleeping / offline" bug.


---

## Meet the Crew

- **Meow (The RP pico wh):** The Master Host. Meow initialises its own open access point (named `Meow-Net`) and serves a green phosphor terminal page on port 80. It crunches through layers 0 and 1 of the transformer, handles punctuation tokenization, coordinates the thought vectors, and streams telemetry!
- **Miaou (The RP pico w):** The Slave Worker. It connects to Meow's network, workers(whoosh whoosh) through layers 2 and 3 over wifi, evaluates all 1,024 logits, and performs zero-RAM top-k token sampling.


---

## Specs
Even with extremely humble hardware clocked at 240 MHz, this model has a real transformer architecture.
- **Model Size:** Split into two binary files, ~**670 KB** each (in `/models/`)
- **Embedding Dimension:** 64-D
- **Layers:** 4 (2 per microcontroller)
- **Attention Heads:** 4 heads (16 scalars per head)
- **Vocabulary:** 1 024 words (fairy tales, cat lore, physical states, and logical reasoning syntax)
- **Context Block Size:** 64 tokens
- **Clock Speed:** 240 MHz (tuned down slightly from 250 MHz so the chips stay rock-solid and stop dropping wifi)
- **Weights Execution:** Direct Flash XIP streaming (less than 15 KB RAM used during execution)


---

## Benchmark Scorecard (The 85-Question Battery)

Ran using `evaluator.py` with pure standard python (deterministic argmax on logic benchmarks, creative multinomial on stories):

```text
============================================================================
                           CATEGORY SCORECARD                          
============================================================================
Actor Location               : 09/10 [#########-]  90.0%
Object Location              : 08/08 [##########] 100.0%
Possession (has)             : 06/06 [##########] 100.0%
Possession (found)           : 07/07 [##########] 100.0%
Object Transfer              : 06/06 [##########] 100.0%
Giving (IOI)                 : 06/08 [#######---]  75.0%
Color Binding                : 08/08 [##########] 100.0%
Persona                      : 08/08 [##########] 100.0%
Transitive                   : 04/04 [##########] 100.0%
Functional Needs             : 04/04 [##########] 100.0%
Action Affordance            : 04/04 [##########] 100.0%
Physical States              : 04/04 [##########] 100.0%
Temporal Order               : 02/04 [#####-----]  50.0%
Inverted / Negation          : 01/04 [##--------]  25.0%
----------------------------------------------------------------------------
TOTAL REASONING SCORE        : 77/85 [#########-]  90.6%
============================================================================