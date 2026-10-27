# A 2-Node Distributed Neural Engine on Raspberry Pi Pico W's

Have you ever wanted to run transformer models on two tiny microcontrollers that distribute layers over wifi? I'm so glad you asked!!! Meet **Meow** and **Miaou**, two electronic kittens made with custom tensor weights, a lot of willpower, love, and MicroPython.

Instead of using one board for everything, this project splits a 6-layer transformer model in half. **Meow** does the first half of the brain and hosts the kawaii telemetry screen, and **Miaou** handles the second half of the generation over wifi (way more useless than **Meow**).

The repo is called **Wet Food** because cats _love_ wet food, and this project, you know, feeds raw neural network streams between two cats, so pretty similar.

---

## Meet the Crew

- **Meow (The Master Host):** Initializes its own open access point (named `Meow-Net`) and serves a green phosphor terminal page on port 80. It crunches through Layers 0, 1, and 2 of the transformer, coordinates the thought vectors, and streams telemetry!
- **Miaou (The Slave Worker):** Connects to Meow's network and worker-threads through Layers 3, 4, and 5 over wifi. It evaluates all 1,024 logits and performs zero-RAM top-k token sampling, sending the final thought back to Meow.

---

## The Hardware Specs
Even with extremely humble hardware clocked at 240 MHz, this model runs a legitimate, mathematically exact transformer architecture.

- **Model Size:** Split into two binary files, ~**1.25 MB** each (running direct Flash XIP streaming)
- **Embedding Dimension:** 80-D
- **Layers:** 6 (3 per microcontroller)
- **Attention Heads:** 4 heads (20 scalars per head)
- **Vocabulary:** 1,024 words (fairy tales, cat lore, physical states, and logical reasoning syntax)
- **Context Block Size:** 64 tokens
- **Execution:** Pure standard-library math. Zero dependencies.

---

## Atomic World-State Reasoning

Because this is an 80-D micro-transformer, it doesn't know the whole internet, but it is an absolute genius at **in-context reasoning, entity tracking, and theory of mind** 

Unlike standard benchmarks that use rigid, memorizable templates, this engine was trained using a **dynamic atomic world-state generator**. Actors randomly hold 0, 1, or 2 items, have overlapping emotions and locations, and form compound sentences without fixed slot positions. 

Here is what the kittens can understand:

### 1. Compositional Cross-Tasking & Motivations
The model doesn't just match keywords; it tracks physical locations even inside emotional stories.
* `[c] alice is hungry . alice went to the kitchen . bob is tired . bob went to the room . [q] where is alice ? [a]`
  -> `kitchen .`
* `[c] alice is hungry . alice went to the kitchen . bob is tired . bob went to the room . [q] who is thirsty ? [a]`
  -> `nobody .`

### 2. Variable Cardinality & Possession
Items are dynamically assigned. The model actually has to count and read, preventing positional cheating.
* `[c] felix has the cup . luna has the ball and the book . [q] how many items does luna have ? [a]`
  -> `two .`
* `[c] felix has the cup . luna has the ball and the book . [q] what does felix have ? [a]`
  -> `cup .`
* `[c] alice has the ball . bob has the cup . [q] does alice have the ball ? [a]`
  -> `yes .`

### 3. The Frame Problem (Selective Departure)
Understands that if two people are in a room and one leaves, the other is still there.
* `[c] alice and bob were in the room . then alice went to the garden . [q] where is bob ? [a]`
  -> `room .`

### 4. Transferred Attribute Preservation
Binds adjectives to objects and preserves them across multi-hop ownership transfers.
* `[c] alice had a red ball . alice gave the red ball to bob . [q] who has the red ball ? [a]`
  -> `bob .`
* `[c] alice had a red ball . alice gave the red ball to bob . [q] what colour is the ball ? [a]`
  -> `red .`

### 5. Relational Symmetry & Inversion
Understands reciprocal truths in polar constraints.
* `[c] the box is bigger than the ball . the ball is bigger than the cup . [q] is the cup smaller than the box ? [a]`
  -> `yes .`

### 6. Storytelling Mode (Open-Ended Fiction)
Start a story sentence without logic tags and let them write. (Uses temperature scaling to prevent loops!)
* `once upon a time bob went into the green garden .`
* `one day alice and felix were in the kitchen .`

---

## Benchmark Scorecard (125,000+ Tests)

The model is evaluated against a deterministic, mathematically exhaustive combinatorial stress suite generated dynamically from its atomic logic configurations.

```text
============================================================================
                   PARALLEL EVALUATION SCORECARD                       
============================================================================
Single/Multi Supporting Facts      : 053909/060096 [#########-]  89.7%
Argument/Positional Relations      : 001248/001248 [##########] 100.0%
Yes/No Truth Verification          : 000384/000384 [##########] 100.0%
Counting & Inverse Symmetry        : 010368/010368 [##########] 100.0%
Lists and Sets                     : 010368/010368 [##########] 100.0%
Negation & Indefinite Knowledge    : 000576/000576 [##########] 100.0%
Basic & Compound Coreference       : 003795/004368 [########--]  86.9%
Time Reasoning (Before/After)      : 000420/000480 [########--]  87.5%
Agent Motivations & Needs          : 002201/002208 [#########-]  99.7%
Cross-Task Query Transfer          : 002009/002016 [#########-]  99.7%
Polar Possession (does... have)    : 016390/016416 [#########-]  99.8%
Selective Departure (Frame Prob)   : 004145/004752 [########--]  87.2%
Transferred Attribute (Colour)     : 009812/010368 [#########-]  94.6%
Relational Symmetry Inversion      : 000288/000288 [##########] 100.0%
----------------------------------------------------------------------------
TOTAL REASONING SCORE              : 117262/125312 [#########-]  93.6%
============================================================================
```

## Future Plans & Roadmap

We have pushed pure FP32 float math on these microcontrollers to their absolute limits. To scale the engine further, the following upgrades are planned:

* **INT8 Post-Training Quantization (W8A32):** The RP2040 has no hardware FPU, meaning float math is emulated and slow. By quantizing the weights to 8-bit integers, we can unlock the Pico's 1-cycle hardware `MULS` instruction. This will slash the `.model` file sizes from 1.25 MB to ~315 KB, vastly speeding up SPI Flash reads.
* **Multi-Query Attention (MQA):** To free up SRAM, we plan to shift from standard Multi-Head Attention to MQA. By sharing a single Key and Value head across all 4 Query heads, the KV cache footprint will drop by 4x (from 123 KB down to ~30 KB), completely eliminating memory fragmentation risks.
* **Dual-Core Pipelining:** Currently, Core 0 handles matrix math, web serving, and the WiFi TCP stack. We plan to use the RP2040's hardware FIFOs to offload the `cyw43` WiFi driver and HTTP server to Core 1, allowing Core 0 to compute neural layers at 100% duty cycle without network interrupts.