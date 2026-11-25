import copy, json, math, os, re, sys, time
from collections import Counter
import requests
import torch
from torch.nn import functional as F

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from curriculum import generateAtomicSample
from pico.model import (
    BabyStories, dump_tensor, get_best_device, write_model_header
)
from pico.tokenizer import TOKEN_REGEX, pluralize, tokenize

torch.set_num_threads(4)

MODEL_CFG_PATH = os.path.join(project_root, "config", "model.json")
WORLD_CFG_PATH = os.path.join(project_root, "config", "world.json")

with open(MODEL_CFG_PATH, "r") as f:
    modCfg = json.load(f)
with open(WORLD_CFG_PATH, "r") as f:
    logCfg = json.load(f)

VOCAB_SIZE   = modCfg["architecture"]["vocab_size"]
BLOCK_SIZE   = modCfg["architecture"]["block_size"]
EMBED_DIM    = modCfg["architecture"]["embed_dim"]
NUM_HEADS    = modCfg["architecture"]["num_heads"]
NUM_LAYERS   = modCfg["architecture"]["num_layers"]
MEOW_LAYERS  = modCfg["architecture"].get("meow_layers", NUM_LAYERS // 2)
MIAOU_LAYERS = modCfg["architecture"].get("miaou_layers", NUM_LAYERS - MEOW_LAYERS)
DEVICE       = get_best_device()

trnCfg       = modCfg.get("training", {})
WEIGHT_DECAY = trnCfg.get("weight_decay", 0.02)
GRAD_CLIP    = trnCfg.get("grad_clip", 1.0)

pretrainCfg    = trnCfg.get("pretrain", {})
PRETRAIN_STEPS = pretrainCfg.get("steps", 14000)
PRETRAIN_BATCH = pretrainCfg.get("batch_size", 32)
PRETRAIN_LR    = pretrainCfg.get("base_lr", 0.002)
PRETRAIN_MINLR = pretrainCfg.get("min_lr", 0.0002)
WARMUP_STEPS   = pretrainCfg.get("warmup_steps", 600)

sftCfg    = trnCfg.get("sft", {})
SFT_STEPS = sftCfg.get("steps", 18000)
SFT_BATCH = sftCfg.get("batch_size", 32)
SFT_LR    = sftCfg.get("base_lr", 0.00035)
SFT_MINLR = sftCfg.get("min_lr", 0.00001)
QA_SMOOTH = sftCfg.get("label_smoothing", 0.02)
QA_W      = sftCfg.get("qa_loss_weight", 2.0)
STORY_W   = sftCfg.get("story_loss_weight", 0.45)

MODELS_DIR   = os.path.join(project_root, modCfg["paths"]["models_dir"])
STORIES_FILE = os.path.join(project_root, modCfg["paths"]["stories_file"])
WEIGHTS_PATH = os.path.join(project_root, modCfg.get("paths", {}).get("weights_file", "config/weights.json"))

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(project_root, "firmware"), exist_ok=True)

with open(WEIGHTS_PATH, "r") as f:
    weightsMap = json.load(f)

world_cats = [c["name"] for c in logCfg.get("categories", [])]
missing = [c for c in world_cats if c not in weightsMap]
if missing:
    raise KeyError(f"[!] FATAL: '{WEIGHTS_PATH}' is missing weights for: {missing}")

for cat in logCfg["categories"]:
    cat["weight"] = float(weightsMap[cat["name"]])

top_weights = sorted([(c["name"], c["weight"]) for c in logCfg["categories"]], key=lambda x: x[1], reverse=True)[:5]
top_str = ", ".join([f"{name[:14]}={w}" for name, w in top_weights])
print("=" * 65)
print(f"[✓] VERIFIED: 100% match between world.json and '{WEIGHTS_PATH}' ({len(world_cats)} tasks).")
print(f"    Arch: {EMBED_DIM}-D / {NUM_LAYERS}L ({MEOW_LAYERS}+{MIAOU_LAYERS}) / {NUM_HEADS}H / V={VOCAB_SIZE} / T={BLOCK_SIZE}")
print(f"    Active top weights: {top_str}...")
print("=" * 65)

# 1. Dataset & Vocabulary Preparation
STORY_MAX_BYTES = modCfg["dataset"].get("max_story_bytes", 18 * 1024 * 1024)

if os.path.exists(STORIES_FILE):
    with open(STORIES_FILE, "r", encoding="utf-8", errors="ignore") as f:
        rawStories = f.read().lower()
else:
    print("[*] Downloading TinyStories dataset...")
    resp = requests.get(modCfg["dataset"]["url"], headers={"User-Agent": "Mozilla/5.0"}, stream=True)
    with open(STORIES_FILE, "wb") as f:
        for chunk in resp.iter_content(chunk_size=512 * 1024):
            f.write(chunk)
            if f.tell() >= STORY_MAX_BYTES:
                break
    with open(STORIES_FILE, "r", encoding="utf-8", errors="ignore") as f:
        rawStories = f.read().lower()

# Replace TinyStories document separators with <eos> instead of leaking "endoftext" as a word
rawStories = rawStories.replace("<|endoftext|>", " <eos> ")

TARGET_VOCAB_SAMPLES = modCfg["dataset"].get("logic_samples_target", 12000)
allQaTasks = [generateAtomicSample(logCfg) for _ in range(TARGET_VOCAB_SAMPLES)]

autoPoolWords = set()
for poolKey, poolList in logCfg.get("pools", {}).items():
    if isinstance(poolList, list):
        for item in poolList:
            w = item.lower()
            autoPoolWords.add(w)
            if poolKey in ("item", "container"):
                autoPoolWords.add(pluralize(w))

BASE_CORE = modCfg["dataset"]["core_words"]
ESSENTIAL_TOKENS = ["i", "'s"]
CAT_CORE_WORDS = list(dict.fromkeys(BASE_CORE + ESSENTIAL_TOKENS + sorted(autoPoolWords)))

storyWords = re.findall(TOKEN_REGEX, rawStories[:10_000_000])
cleanStoryWords = [
    w for w in storyWords
    if w in CAT_CORE_WORDS or (w.isalpha() and len(w) > 1 and not w.endswith("n't"))
]
qaWords = re.findall(TOKEN_REGEX, " ".join(allQaTasks))

counts = Counter(cleanStoryWords + qaWords)
for w in CAT_CORE_WORDS:
    counts.pop(w, None)

VOCAB = CAT_CORE_WORDS + [w for w, _ in counts.most_common(VOCAB_SIZE - len(CAT_CORE_WORDS))]
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}
padId, aTokenId = WORD_TO_ID["<pad>"], WORD_TO_ID["[a]"]

storyTokens = torch.tensor(tokenize(rawStories, WORD_TO_ID), dtype=torch.long)
numStChunks = (len(storyTokens) - 1) // BLOCK_SIZE
allStX = storyTokens[: numStChunks * BLOCK_SIZE].view(numStChunks, BLOCK_SIZE)
allStY = storyTokens[1 : numStChunks * BLOCK_SIZE + 1].view(numStChunks, BLOCK_SIZE)

valStCount = min(256, max(32, int(numStChunks * 0.02)))
trainStCount = numStChunks - valStCount
stXData = allStX[:trainStCount].to(DEVICE)
stYData = allStY[:trainStCount].to(DEVICE)
valStX  = allStX[trainStCount : trainStCount + valStCount].to(DEVICE)
valStY  = allStY[trainStCount : trainStCount + valStCount].to(DEVICE)

rawModel = BabyStories(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, BLOCK_SIZE).to(DEVICE)

use_amp = DEVICE == "cuda"
amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        print("[*] Benchmark flag passed; skipping training.")
        sys.exit(0)

    # =====================================================================
    # STAGE 1: PRE-TRAINING ON TINYSTORIES
    # =====================================================================
    optPretrain = torch.optim.AdamW(rawModel.parameters(), lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY)

    def getPretrainLr(step):
        if step < WARMUP_STEPS:
            return PRETRAIN_LR * (step + 1) / WARMUP_STEPS
        prog = (step - WARMUP_STEPS) / max(1, PRETRAIN_STEPS - WARMUP_STEPS)
        return PRETRAIN_MINLR + 0.5 * (1.0 + math.cos(math.pi * prog)) * (PRETRAIN_LR - PRETRAIN_MINLR)

    print(f" STAGE 1: PRE-TRAINING (TinyStories, {PRETRAIN_STEPS} Steps, Batch {PRETRAIN_BATCH}, AMP={use_amp})")
    print("=" * 65)

    rawModel.train()
    t0 = time.time()
    for step in range(PRETRAIN_STEPS + 1):
        lr = getPretrainLr(step)
        for p in optPretrain.param_groups:
            p["lr"] = lr

        stIdx = torch.randint(trainStCount, (PRETRAIN_BATCH,))
        xSt, ySt = stXData[stIdx], stYData[stIdx]

        optPretrain.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            loss = F.cross_entropy(rawModel(xSt).view(-1, VOCAB_SIZE), ySt.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optPretrain)
        torch.nn.utils.clip_grad_norm_(rawModel.parameters(), GRAD_CLIP)
        scaler.step(optPretrain)
        scaler.update()

        if step % 500 == 0:
            elapsed = time.time() - t0
            eta = (PRETRAIN_STEPS - step) * (elapsed / max(1, step)) / 60
            print(f" [Pre-train] Step {step:5d}/{PRETRAIN_STEPS} | Story Loss: {loss.item():.4f} | LR: {lr:.5f} | ETA: {eta:.1f}m", flush=True)

    # =====================================================================
    # STAGE 2: FAST PRE-BUILT LOGIC + MICRO-COT SFT POOL
    # =====================================================================
    SFT_POOL_SIZE = modCfg["training"]["sft"].get("pool_size", 120000)
    print(f"\n[*] Pre-generating {SFT_POOL_SIZE:,} SFT samples (QA + Micro-CoT + RAG) onto {DEVICE.upper()}...")
    t_gen = time.time()
    torch.manual_seed(42)

    def encode_qa_pool(n_samples):
        xs = torch.full((n_samples, BLOCK_SIZE), padId, dtype=torch.long)
        ys = torch.full((n_samples, BLOCK_SIZE), -100, dtype=torch.long)
        for i in range(n_samples):
            # formatQaSequence already pads punctuation with spaces -> .split() is 15x faster than regex
            words = generateAtomicSample(logCfg).lower().replace("'s", " 's ").split()
            toks = [WORD_TO_ID.get(w, 0) for w in words[:BLOCK_SIZE]]
            L = len(toks)
            xs[i, :L] = torch.tensor(toks, dtype=torch.long)
            try:
                ansStart = toks.index(aTokenId)
                for idx in range(ansStart, L - 1):
                    ys[i, idx] = toks[idx + 1]
            except ValueError:
                pass
        return xs.to(DEVICE), ys.to(DEVICE)

    valX, valY = encode_qa_pool(320)
    qaXPool, qaYPool = encode_qa_pool(SFT_POOL_SIZE)
    print(f"[✓] SFT pool ready in {time.time() - t_gen:.1f}s ({qaXPool.numel() * 8 / 1e6:.1f} MB VRAM).")

    def evaluateValMetrics(model):
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                qa_val = F.cross_entropy(model(valX).view(-1, VOCAB_SIZE), valY.view(-1), ignore_index=-100).item()
                st_val = F.cross_entropy(model(valStX).view(-1, VOCAB_SIZE), valStY.view(-1)).item()
        model.train()
        return qa_val, st_val

    optSft = torch.optim.AdamW(rawModel.parameters(), lr=SFT_LR, weight_decay=WEIGHT_DECAY)

    def getSftLr(step):
        prog = step / max(1, SFT_STEPS)
        return SFT_MINLR + 0.5 * (1.0 + math.cos(math.pi * prog)) * (SFT_LR - SFT_MINLR)

    print("\n" + "=" * 65)
    print(f" STAGE 2: REASONING + MICRO-COT SFT ({SFT_STEPS} Steps, Batch {SFT_BATCH}, QA={QA_W}/Story={STORY_W})")
    print("=" * 65)

    best_qa_loss, best_state, best_step = float("inf"), None, 0
    t1 = time.time()

    for step in range(SFT_STEPS + 1):
        lr = getSftLr(step)
        for p in optSft.param_groups:
            p["lr"] = lr

        qaIdx = torch.randint(SFT_POOL_SIZE, (SFT_BATCH,), device=DEVICE)
        stIdx = torch.randint(trainStCount, (max(8, SFT_BATCH // 2),), device=DEVICE)

        optSft.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            qa_loss = F.cross_entropy(
                rawModel(qaXPool[qaIdx]).view(-1, VOCAB_SIZE),
                qaYPool[qaIdx].view(-1),
                ignore_index=-100,
                label_smoothing=QA_SMOOTH,
            )
            story_loss = F.cross_entropy(rawModel(stXData[stIdx]).view(-1, VOCAB_SIZE), stYData[stIdx].view(-1))
            loss = QA_W * qa_loss + STORY_W * story_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optSft)
        torch.nn.utils.clip_grad_norm_(rawModel.parameters(), GRAD_CLIP)
        scaler.step(optSft)
        scaler.update()

        if step % 500 == 0:
            qa_val, st_val = evaluateValMetrics(rawModel)
            if qa_val < best_qa_loss:
                best_qa_loss = qa_val
                best_step = step
                best_state = copy.deepcopy(rawModel.state_dict())
            elapsed = time.time() - t1
            eta = (SFT_STEPS - step) * (elapsed / max(1, step)) / 60
            print(
                f" [Logic SFT] Step {step:5d}/{SFT_STEPS} | Batch: {loss.item():.4f} | "
                f"QA Val: {qa_val:.4f} (Best: {best_qa_loss:.4f} @ #{best_step}) | Story Val: {st_val:.4f} | ETA: {eta:.1f}m",
                flush=True,
            )

    final_qa, final_st = evaluateValMetrics(rawModel)
    print("\n" + "=" * 65)
    print(f" CHECKPOINT COMPARISON: Final Step {SFT_STEPS} QA={final_qa:.4f} (Story={final_st:.4f}) | Best Step {best_step} QA={best_qa_loss:.4f}")
    if best_qa_loss < final_qa and best_state is not None:
        print(f"[✓] Best validation model wins! Loading checkpoint from Step {best_step} for export.")
        rawModel.load_state_dict(best_state)
    else:
        print(f"[✓] Final model is optimal or equal. Exporting Step {SFT_STEPS}.")
    print("=" * 65)

    # =====================================================================
    # STAGE 3: INT8 EXPORT & PACKAGING
    # =====================================================================
    print("\n[*] Quantizing winning weights (INT8 W8A32) and exporting...")
    p1Path = os.path.join(MODELS_DIR, "pico1 weights.model")
    p2Path = os.path.join(MODELS_DIR, "pico2 weights.model")

    with open(p1Path, "wb") as f:
        write_model_header(f, VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MEOW_LAYERS, BLOCK_SIZE)
        dump_tensor(rawModel.tokEmb.weight, f, quantize=True)
        for i in range(MEOW_LAYERS):
            for p in rawModel.layers[i].parameters():
                dump_tensor(p, f, quantize=True)

    with open(p2Path, "wb") as f:
        write_model_header(f, VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MIAOU_LAYERS, BLOCK_SIZE)
        for i in range(MEOW_LAYERS, NUM_LAYERS):
            for p in rawModel.layers[i].parameters():
                dump_tensor(p, f, quantize=True)
        for p in rawModel.lnF.parameters():
            dump_tensor(p, f, quantize=True)
        dump_tensor(rawModel.lmHead.weight, f, quantize=True)
        dump_tensor(rawModel.lmHead.bias, f, quantize=True)

    vocabCode = f"VOCAB = {repr(VOCAB)}\n"
    with open(os.path.join(MODELS_DIR, "vocab.py"), "w") as f:
        f.write(vocabCode)
    with open(os.path.join(project_root, "firmware", "vocab.py"), "w") as f:
        f.write(vocabCode)

    print(f"[✓] Meow Weights (pico1, {MEOW_LAYERS}L): {os.path.getsize(p1Path)/1024:.1f} KB -> Saved to {p1Path}")
    print(f"[✓] Miaou Weights (pico2, {MIAOU_LAYERS}L): {os.path.getsize(p2Path)/1024:.1f} KB -> Saved to {p2Path}")
    print(f"[✓] Vocab updated: {len(VOCAB)} words.")