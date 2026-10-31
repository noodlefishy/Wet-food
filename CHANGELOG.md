# picoAI (Wet Food) - AI & Model Architecture Changelog

Chronological log of model architecture, attention mechanics, training dynamics, quantization, and evaluation benchmarks.

---

## October 16, 2026

- **Parameter Budgeting & Architecture Selection**
  - Evaluated sub-200 KB autoregressive sequence models for 264 KB SRAM / 2 MB Flash limits.
  - Benchmarked Char-RNN/LSTM vs. tiny Causal Transformer; identified KV-cache expansion as the main RAM limit.
  - Measured software float emulation penalties on dual Cortex-M0+ cores without hardware FPU.
  - Prototyped minimal character-level generator in MicroPython to profile forward-pass overhead.

---

## October 18, 2026

- **Distributed Pipeline Parallelism**
  - Designed 2-stage pipeline split across two microcontrollers.
  - Partitioned 4-layer Transformer: Node A runs Embeddings + Layers 0–1; Node B runs Layers 2–3 + LM Head.
  - Formulated intermediate activation handoff: 48-D float32 latent vector (192 bytes) sent per step via sockets.

---

## October 20, 2026

- **Flash-Streaming (XIP) Execution**
  - Implemented row-by-row weight streaming from Flash into a single reusable 768-byte RAM buffer.
  - Reduced runtime memory from $O(\text{weights})$ to $O(\text{layer width})$, saving SRAM strictly for KV-cache and activations.

---

## October 22, 2026

- **Tokenization & Output Head Scaling ($V=256 \to V=1024$)**
  - Deprecated character-level tokens after diagnosing compounding phonetic errors (e.g. `netindl`).
  - Switched to a 1,024-token whole-word vocabulary with logic tags (`[c]`, `[q]`, `[a]`).
  - Implemented chunked bias accumulation in the LM head (64 floats/step) to prevent 4,097-byte heap allocation crashes.

---

## October 23, 2026

- **Decoding & Sampling Controls**
  - Added zero-heap Top-$K$ ($k=5$) sampling with temperature scaling.
  - Set adaptive temperature: $T=0.05$ (greedy argmax) for logic/math, $T=0.30\text{--}0.70$ for narrative generation.
  - Added leading token suppression filter to strip prompt echoes (`?`, `:`, `answer`).

---

## October 24, 2026

- **64-D Model Scaling & Loss Decoupling**
  - Scaled architecture to 64-D, 4 heads ($d_k=16$), 4 layers, and 64-token context window.
  - Converted KV-cache storage to C-contiguous `array('f')` buffers (80% RAM cut, zero heap fragmentation).
  - Added in-band soft reset (`-999.0` sentinel float) to flush KV-cache between prompts without closing sockets.
  - Diagnosed loss dilution (story tokens dominated 95.6% of gradient updates vs. 4.4% for QA).
  - Implemented target masking (`ignore_index = -100`) on prompt tokens and decoupled loss:
    $$\mathcal{L}_{\text{total}} = 2.5 \cdot \mathcal{L}_{\text{QA}} + 0.3 \cdot \mathcal{L}_{\text{Story}}$$
  - Added single-entity primers and 50/50 clause balance to break binary attention inversion and mode collapse.
  - Hit 90.6% on 85-question reasoning benchmark in pure Python standard library evaluator (`evaluator.py`).

---

## October 25, 2026

- **Baseline Reasoning Verification & Training Data Pipeline**
  - Built offline simulation scripts to verify mathematical parity with microcontroller firmware.
  - Added structured evaluation datasets (`questions.json`, `tests.txt`) for tracking, location, and IOI tasks.
  - Added `stories.txt` integration alongside synthetic logic tasks in `training/local.py`.
  - Automated task generation via rule configs (`config/logic.json`, `config/model.json`).
  - Added dynamic test generation to `evaluator.py` with deterministic scoring.

---

## October 26, 2026

- **Curriculum & Stress Testing**
  - Built `benchmarks/stress_test.py` to test attention under heavy distractor load.
  - Modularized curriculum scheduling in `curriculum.py` for staged task difficulty.
  - Adopted bAbI-style syntactic formatting and clause rules in `config/logic.json`.
  - Added entity collision avoidance in data generation (`config/stress_test.json`).
  - Built vectorized batch evaluator (`benchmarks/fast_evaluator.py`) to speed up testing.

---

## October 27, 2026

- **INT8 Quantization & Multi-Query Attention (MQA)**
  - Migrated from Multi-Head Attention (MHA) to Multi-Query Attention (MQA) (`dd650ff`), reducing KV-cache RAM by 75%.
  - Added asymmetric INT8 weight quantization with per-channel FP32 scaling vectors (`a747614`).
  - Standardized MQA head allocations and sliding window bounds.
  - Expanded data generation to multi-clause deduction chains (`config/logic.json`).
  - Added world-state entity graph sampling (`config/world.json`, `curriculum.py`) for logical consistency.

---

## October 28, 2026

- **Activation Probing & Sparse Autoencoders (SAE)**
  - Built `probe_cortex.py` to log layer-by-layer hidden activation states during inference.
  - Added representation-steering tools (`tools/build_cortex.py`, `tools/build_catalysts.py`) to build catalyst matrices.
  - Trained Sparse Autoencoders (`tools/train_sae.py`) on intermediate activations to identify interpretable latent features.

---

## October 29, 2026

- **6-Layer Scaling & Anti-Recency Tuning**
  - Scaled model from 4 to 6 layers total (3 layers per board) for deeper relational deduction.
  - Added arithmetic and memoization tasks to `config/world.json` and `curriculum.py`.
  - Added adversarial anti-recency rules to prevent heads from using positional distance shortcuts.
  - Added Micro-Chain-of-Thought (Micro-CoT) anchor tokens (`[q] ... [a] ...`) to stabilize attention routing.
  - Built `benchmarks/ood_evaluator.py` for out-of-distribution generalization testing.

---

## October 30, 2026

- **Question-First Formatting & Induction Heads**
  - Expanded Micro-CoT templates for multi-step reasoning traces.
  - Added induction clause copying (`config/world.json`) to bias attention toward premise clauses.
  - Shifted data format to Question-First across training and evaluation loops.
  - Adjusted MQA head dimension ratios and stripped parametric verb jitter to stabilize convergence.

---

## October 31, 2026

- **Architecture Ablations & 99% Milestone**
  - Ran MHA vs. MQA parameter-budget ablations.
  - Balanced TinyStories language modeling loss against masked instruction fine-tuning (`ignore_index = -100`).
  - Applied precision scaling patches (`config/weights.json`, `config/model.json`) to stop superposition collapse.
  - Added Sliding Window Attention (SWA) with attention sinks to preserve initial token representations.
  - Added hard-negative mining and adversarial distractors to curriculum generation.
  - Configured Quantized Clause Attention (QCA) rules aligned with clause boundaries.
  - **Milestone:** Reached ~99% accuracy on distributed 6-layer microcontroller benchmark suite.