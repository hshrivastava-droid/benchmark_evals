# Benchmark Experiments: Mac (MLX) vs DGX Spark (Ollama)

## What Are We Testing

Accuracy and quality of LLM inference across two fundamentally different hardware and runtime stacks:

- **Mac (Apple Silicon)** — MLX runtime, unified memory, Metal GPU, `mlx-community` 4-bit quantized models
- **DGX Spark** — llama.cpp via Ollama, dedicated VRAM, CUDA, GGUF Q4_K_M quantized models

Same model family, same quantization level, same eval harness, same prompts — isolating the hardware and framework as the variable.

---

## Why We Expect Mac to be the Lower Performer

### 1. Quantization Differences
- Ollama uses **Q4_K_M** (k-quant): mixed precision — sensitive layers (attention, first/last) are kept at higher bit depth, less sensitive layers compressed more aggressively. Better accuracy-per-bit than uniform quantization.
- MLX uses **uniform 4-bit**: every layer quantized equally. No layer-wise sensitivity analysis.
- Result: Ollama's GGUF weights are numerically closer to FP16 baseline → higher task accuracy at equivalent "4-bit" label.

### 2. Memory Bandwidth
- Mac unified memory is shared between CPU and GPU. As context length grows, KV cache competes with model weights for the same memory bus.
- DGX Spark has dedicated VRAM with much higher bandwidth → KV cache reads/writes are faster and don't contend with weight loading.
- Result: Mac degrades faster at long context, both in speed and potentially accuracy (cache eviction or approximation).

### 3. Attention Implementation
- llama.cpp uses Flash Attention optimized for CUDA — numerically stable, memory efficient at long sequences.
- MLX uses Metal kernels — solid for short context, less battle-tested at 32K+ tokens.
- Result: At long context, MLX may accumulate more numerical error in attention scores.

### 4. Framework Maturity
- llama.cpp is the most widely used inference engine for GGUF models — highly optimized, years of production tuning.
- MLX is newer, Apple Silicon-native, but less optimized for inference edge cases.

---

## Baseline (Already Running)

**Task:** GSM8K (math word problems, ~200-500 tokens/question)
**Purpose:** Short-context accuracy baseline — KV cache not stressed, pure model quality signal.

| Config | Mac (MLX) | DGX Spark (Ollama) |
|--------|-----------|-------------------|
| Model | `mlx-community/Qwen3-30B-A3B-4bit` | `qwen3:30b` |
| Port | 8080 | 11434 |
| Concurrency | 1 | 1 |
| Few-shot | 4 | 4 |
| Max tokens | 8192 | 8192 |
| Temperature | 0 | 0 |
| Questions | 1319 | 1319 |

```bash
./run.sh --model Qwen/Qwen3-30B --port <PORT> \
  --eval-as <MODEL_NAME> \
  --eval-only --gen-max-tokens 8192 \
  --eval-concurrent 1 --num-fewshot 4
```

**Hypothesis:** Scores will be close (~1-3% gap), since short context doesn't stress either system's weaknesses.

---

## Experiment 1 — Same Setup, Different Models

### Goal
Verify whether the Mac vs DGX accuracy gap is model-specific or systematic across model families and sizes.

### Why
If the gap holds across multiple models, it points to a **framework/quantization systematic issue** rather than a quirk of one model's weight layout.

### Models to Try

| Model | Ollama name | MLX name | Architecture | Why |
|-------|------------|----------|-------------|-----|
| **Kimi K2** | `kimi-k2` | `mlx-community/Kimi-K2-Instruct-4bit` | MoE (1T total / 32B active) | Moonshot AI, same MoE class as Qwen3-30B-A3B — strong coding/reasoning, tests if MoE quantization gap is consistent |
| **MiniMax-M1** | `minimax-m1` | `mlx-community/MiniMax-M1-40k-4bit` | MoE + linear attention hybrid | Extreme long context (1M token claimed) — critical for Exp 2; unique hybrid attention is interesting to stress |
| **Llama-3.1-8B** | `llama3.1:8b` | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | Dense transformer | Reference baseline — widely benchmarked, known GSM8K scores to sanity-check your harness |
| **Gemma-3-9B** | `gemma3:9b` | `mlx-community/gemma-3-9b-it-4bit` | Dense, grouped-query attention | Google architecture, different GQA pattern than MoE models |
| **Mistral-7B** | `mistral:7b` | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | Dense + sliding window attention | Sliding window attention behaves differently at long context — interesting contrast to full attention models |

> **Note:** Verify Ollama availability with `ollama search kimi-k2` and `ollama search minimax` before pulling.
> For MLX, check `https://huggingface.co/mlx-community` — if a 4-bit version isn't there yet, you can convert with `mlx_lm.convert`.

### Commands

```bash
# Template — replace MODEL_HF, PORT, EVAL_AS for each pair

# Mac (MLX)
./run.sh --model <MODEL_HF> \
  --port 8080 \
  --eval-as <MLX_NAME> \
  --eval-only --gen-max-tokens 8192 \
  --eval-concurrent 1 --num-fewshot 4

# DGX Spark (Ollama)
./run.sh --model <MODEL_HF> \
  --port 11434 \
  --eval-as <OLLAMA_NAME> \
  --eval-only --gen-max-tokens 8192 \
  --eval-concurrent 1 --num-fewshot 4
```

### Example — Kimi K2

```bash
# Mac
./run.sh --model moonshotai/Kimi-K2-Instruct \
  --port 8080 \
  --eval-as mlx-community/Kimi-K2-Instruct-4bit \
  --eval-only --gen-max-tokens 8192 \
  --eval-concurrent 1 --num-fewshot 4

# DGX Spark
./run.sh --model moonshotai/Kimi-K2-Instruct \
  --port 11434 \
  --eval-as kimi-k2 \
  --eval-only --gen-max-tokens 8192 \
  --eval-concurrent 1 --num-fewshot 4
```

### What to Look For
- Is the Mac always lower, or does it flip for some models?
- Do MoE models (Kimi K2, MiniMax-M1, Qwen3-30B-A3B) show a larger gap than dense models (Llama, Gemma)?
- MoE quantization is more complex — expert routing + weight quantization — Mac may degrade more on MoE than dense.

---

## Experiment 2 — Long Context Task

### Goal
Measure how accuracy degrades as context length increases, and whether Mac degrades faster than DGX Spark.

### Why This Matters
GSM8K is ~500 tokens — neither system is under memory pressure. Real-world inference (RAG, document QA, long chat) operates at 8K–32K tokens. This is where unified memory vs dedicated VRAM, and Flash Attention vs Metal kernels, actually diverge.

### Recommended Task: RULER - Needle in a Haystack (NIAH)

**Why RULER NIAH:**
- Synthetic, fully controllable context length (1K → 128K)
- Single clear answer buried in noise — binary correct/wrong
- No ambiguity in scoring
- Directly tests KV cache fidelity: can the model retrieve a fact from position X in the context?
- Used in production model evals (GPT-4, Claude, Llama-3 papers)

**What it does:** Hides a key-value pair ("The secret code is 7392") at a specific position in a long document of random text. Asks the model to recall it at the end.

### Context Lengths to Test

| Length | Expected Mac behavior | Expected DGX behavior |
|--------|----------------------|----------------------|
| 4K | Baseline, both fine | Baseline, both fine |
| 8K | Slight degradation | Minimal degradation |
| 16K | Noticeable drop | Small drop |
| 32K | Significant drop | Moderate drop |
| 64K | May fail entirely | Manageable |

### Alternative Tasks

| Task | Context | What it tests | Why useful |
|------|---------|---------------|------------|
| **LongBench-QA** | 4K–16K | Multi-doc QA | Real-world RAG simulation |
| **SCROLLS/QuALITY** | 5K–10K | Reading comprehension | Tests reasoning across long narrative |
| **RULER Multi-key** | 4K–128K | Retrieve multiple needles | Harder than single needle, more realistic |

### Setup

**Step 1 — Generate test data** (one-time, deterministic with seed 42):
```bash
python3 gen_ruler_niah.py                      # all lengths: 4k 8k 16k 32k 64k
python3 gen_ruler_niah.py --lengths 4096 16384 # specific lengths only
python3 gen_ruler_niah.py --samples 200        # more samples per length
```

This writes JSONL to `data/ruler_niah/niah_{4k,8k,16k,32k,64k}.jsonl` (100 samples each by default).
Available task names: `ruler_niah` (=4k), `ruler_niah_4k`, `ruler_niah_8k`, `ruler_niah_16k`, `ruler_niah_32k`, `ruler_niah_64k`.

**Step 2 — Run evals** at increasing context lengths:

```bash
# Mac (MLX) — 4K context
./run.sh --model Qwen/Qwen3-30B --port 8080 \
  --eval-as mlx-community/Qwen3-30B-A3B-4bit \
  --eval-only --task ruler_niah_4k \
  --gen-max-tokens 4096 \
  --eval-concurrent 1 --num-fewshot 0 \
  --limit 100

# Mac (MLX) — 16K context
./run.sh --model Qwen/Qwen3-30B --port 8080 \
  --eval-as mlx-community/Qwen3-30B-A3B-4bit \
  --eval-only --task ruler_niah_16k \
  --gen-max-tokens 16384 \
  --eval-concurrent 1 --num-fewshot 0 \
  --limit 100

# DGX Spark (Ollama) — 4K context
./run.sh --model Qwen/Qwen3-30B --port 11434 \
  --eval-as qwen3:30b-a3b-q4_K_M \
  --eval-only --task ruler_niah_4k \
  --gen-max-tokens 4096 \
  --eval-concurrent 1 --num-fewshot 0 \
  --limit 100

# DGX Spark (Ollama) — 16K context
./run.sh --model Qwen/Qwen3-30B --port 11434 \
  --eval-as qwen3:30b-a3b-q4_K_M \
  --eval-only --task ruler_niah_16k \
  --gen-max-tokens 16384 \
  --eval-concurrent 1 --num-fewshot 0 \
  --limit 100
```

### What to Look For
- At what context length does Mac accuracy start to drop vs DGX?
- Is the drop gradual or a cliff?
- Does it correlate with unified memory pressure (monitor with `sudo powermetrics` on Mac)?

---

## Summary Table

| Experiment | Task | Context | Models | Variable |
|------------|------|---------|--------|----------|
| Baseline | GSM8K | ~500 tok | Qwen3-30B | device/framework |
| Exp 1 | GSM8K | ~500 tok | 5 model pairs | model family + size |
| Exp 2 | RULER NIAH | 4K–64K | Qwen3-30B | context length |

---

## Next Steps

1. Finish baseline GSM8K run (in progress)
2. Pull model pairs on both devices for Experiment 1
3. ~~Write `evals/ruler_niah.yaml` for Experiment 2~~ ✓ Done — `gen_ruler_niah.py` + `evals/ruler_niah*.yaml` (4k/8k/16k/32k/64k)
4. Run RULER NIAH sweeps on both Mac and DGX Spark
5. Build results comparison table once all runs complete
