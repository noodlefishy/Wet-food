# A 2-Node Distributed Neural Engine on Raspberry Pi Pico W's

Have you ever wanted to run transformer models on two tiny microcontrollers that distribute layers over Wi‑Fi? I'm so glad you asked!!! Meet **Meow** and **Miaou**, two electronic kittens made with custom tensor weights, a lot of willpower, love, and MicroPython.

Instead of using one board for everything, this project splits a 6‑layer transformer model in half. **Meow** does the first half of the brain and hosts the kawaii neon telemetry screen, and **Miaou** handles the second half of the generation over Wi‑Fi.

The repo is called **Wet Food** because cats _love_ wet food, and this project, you know, feeds raw neural network streams between two cats, so pretty similar.

---

## Meet the Crew

- **Meow (The Master Host):** Initializes its own network and serves a cyberpunk neon terminal page on port 80. It crunches through Layers 0, 1, and 2, coordinates the thought vectors, and streams token-by-token JSON telemetry using a dual-core async pipeline.
- **Miaou (The Slave Worker):** Connects to Meow's network and crunches Layers 3, 4, and 5 over Wi‑Fi. It evaluates all 1,024 logits, performs in-place Top-K token sampling, and beams the final thought back to Meow.

---

## The Hardware & Architecture Specs

Even with strictly constrained hardware clocked at 240 MHz, this engine runs a mathematically precise, modern LLM architecture heavily optimized for edge inference.

- **Model Size:** 6 Layers (3 per board), ~0.44M Parameters total.
- **Flash Footprint:** Split into two W8A32 INT8 binary files, ~385 KB each (running direct SPI Flash XIP streaming to bypass the 264KB SRAM limit).
- **Embedding Dimension:** 96-D
- **Attention Design:** Multi-Query Attention (MQA) with 4 Query heads ($d_{head}=24$).
- **Position & Context:** RoPE (Rotary Position Embeddings) over a 64-token Sliding Window Attention (SWA) cache with 4 Attention Sinks.
- **Training Paradigm:** Supervised Fine-Tuning (SFT) using `-100` prompt masking and Micro-Chain-of-Thought (Micro-CoT) anchoring.
- **Execution:** Pure standard-library math. _Zero external dependencies_

---

## Architectural Eccentricities & Reasoning Mechanics

While the implementation of spatial logic on dual microcontrollers is inherently _unorthodox_, the Wet Food engine relies on a strict amalgamation of modern training heuristics, memory magic, and algorithmic constraints to prevent high-dimensional superposition collapse 

- **Asymmetric W8A32 Tensor Streaming:** To circumvent the 264 KB SRAM ceiling of the RP2040, the model weights are aggressively compressed via per-row INT8 quantization paired with FP32 scaling vectors. During inference, these weights bypass standard memory allocation entirely, streaming directly from the LittleFS SPI Flash sequentially.

- **Multi-Query Attention (MQA) with Sink-Windowing:** The architecture deploys 4 Query heads mapped to a singular, multiplexed Key/Value head. To maintain context without triggering an Out-Of-Memory kernel panic, the KV cache operates on a strict 32-token sliding window, synthetically stabilized by 4 permanent Attention Sinks to prevent the catastrophic forgetting of initial sequence parameters

- **Anchor-Prefixed Micro-Chain-of-Thought (Micro-CoT):** Because a 6-layer model lacks the depth to resolve multi-hop logic in a single forward pass, it is trained to forcefully echo the subject token prior to the conclusion (eg `[q] where is alice ? [a] alice : kitchen .`). This intermediate temporal step anchors the attention routing and physically bypasses noun recency bias.

- **Adversarial Anti-Recency & Hard-Negative Mining:** The synthetic curriculum generator is shockinly hostile. It intentionally targets the most distant entity in a premise 65 percent of the time, and skews boolean queries toward negative assertions. This mathematically penalizes the network for attempting lazy token distance heuristics or "yes man" co-occurrence shortcuts.

- **Parametric Verb Jitter & Distractor Injection:** Training contexts are subjected to randomized morphological noise (eh swapping "went to" for "arrived at") alongside stochastic distractor insertions (eg injecting `bob saw a little bird` between logical premises). This forces the network to learn robust positional invariance.

- **Bifurcated Loss Annealing & SFT Masking:** The gradient descent isolates reasoning from grammar. Instruction prompts are aggressively masked with `-100` to force 100 percent of the Supervised Fine-Tuning (SFT) gradient onto the micro-cot response. Concurrently, a background TinyStories cross-entropy loss is systematically decayed from 0.35 to 0.15, preserving linguistic fluency without diluting logical convergence.

- **Asynchronous Inter-Core Telemetry:** The host node operates a hardware-decoupled architecture. Core 0 is indefinitely locked in a pure float32 neural loop, pushing activation states through an inter-core FIFO pipeline to Core 1. Core 1 independently manages the TCP/IP stack and non-blocking HTTP daemon to serve real-time json-streamed thought vectors.

- **Dynamic Vocabulary Harvesting:** To prevent Out-Of-Distribution token fragmentation, the compilation pipeline programmatically extracts all entities, container classes, and morphological plurals from the world state configuration, permanently reserving their indices in the 1,024-token latent space before allowing residual narrative words to populate the remaining vocabulary.
---

## Remote Colab GPU Training & Download

The repository ships with a **zero‑click** remote-training pipeline that runs entirely from the terminal, no manual Chrome or Colab UI steps required

### Quick Start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the full remote‑training, live‑streaming output and automatic download
./trainRemote.sh

```

## Hardware Deployment Pipeline

Once the weights and vocabulary are compiled and downloaded to the local machine, the `deploy.py` script does the serial transfer to the RP2040 flash memory utilizing `mpremote`. It manages file system cleanup, payload transfers, and automated soft-reboots.

To flash the master host and slave worker respectively, execute:

```bash
# Flash the HTTP daemon, telemetry logic, and Layer 0-2 tensors to Meow
python3 deploy.py meow

# Flash the worker logic and Layer 3-5 tensors to Miaou
python3 deploy.py miaou

---

## Future Plans & Roadmap

Ive pushed MicroPython and emulated floating point math to their extreme limits on the RP2040. Future scaling will require deep software-level optimizations:

- **Native C-Modules for MicroPython:** Currently, the `stream_linear` inner loops use `@micropython.native`. Writing a dedicated `.c` module to execute matrix multiplication using ARM Thumb-2 Assembly (eg `MULS` and DSP instructions) would slash token generation time from ~6 seconds down to <1 second.

- **INT8 KV Cache Compression:** Storing the SWA KV Cache currently requires ~30 KB of SRAM per board. By quantizing the Key/Value cache elements on the fly from FP32 to INT8, we can instantly double the context window from 64 tokens to 128 tokens without hitting memory fragmentation.

- **Speculative Layer Execution (MoE):** To reduce the massive SPI Flash read overhead, we plan to implement a micro-Mixture-of-Experts (MoE) system where only 2 out of 8 MLP expert blocks are read from Flash per token, heavily reducing the I/O bottleneck.
