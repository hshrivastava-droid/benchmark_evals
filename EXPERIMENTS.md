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

## Experiment 2 — RULER Needle in a Haystack (NIAH)

Test long-context retrieval at 4K–64K tokens. A secret key is buried in noise text; the model must recall it exactly. Thinking mode disabled (`/no_think`) on both platforms — NIAH is pure retrieval, not reasoning.

| Config | DGX Spark (Ollama) | Mac M4 (MLX) |
|--------|-------------------|--------------|
| Model | `qwen3:30b-a3b` (GGUF Q4_K_M) | `mlx-community/Qwen3-30B-A3B-4bit` (MLX 4-bit) |
| Port | 11434 | 8081 |
| Samples | 500 per context length | 500 per context length |
| Needle depths | 10%, 25%, 50%, 75%, 90% | 10%, 25%, 50%, 75%, 90% |
| Thinking | Disabled (`/no_think`) | Disabled (`/no_think`) |

### Setup

```bash
python3 gen_ruler_niah.py --samples 500    # generate test data (one-time)
```

### Run Commands

```bash
# DGX Spark — replace TASK and GEN_MAX for each context length
./run.sh --model Qwen/Qwen3-30B --port 11434 \
  --eval-as qwen3:30b-a3b \
  --eval-only --task ruler_niah_32k \
  --gen-max-tokens 32768 --eval-concurrent 1 --num-fewshot 0

# Mac M4 — replace TASK and GEN_MAX for each context length
./run.sh --model mlx-community/Qwen3-30B-A3B-4bit --port 8081 \
  --eval-only --task ruler_niah_32k \
  --gen-max-tokens 32768 --eval-concurrent 1 --num-fewshot 0
```

Available tasks: `ruler_niah_4k` (4096), `ruler_niah_8k` (8192), `ruler_niah_16k` (16384), `ruler_niah_32k` (32768), `ruler_niah_64k` (65536). Match `--gen-max-tokens` to the context length.

### Results

| Context | DGX Spark | Mac M4 | Notes |
|---------|-----------|--------|-------|
| 4K | pending | pending | |
| 8K | pending | pending | |
| 16K | pending | pending | |
| 32K | pending | pending | |
| 64K | pending | pending | |

---

## Summary Table

| Experiment | Task | Context | Models | Variable |
|------------|------|---------|--------|----------|
| Baseline | GSM8K | ~500 tok | Qwen3-30B | device/framework |
| Exp 1 | GSM8K | ~500 tok | 5 model pairs | model family + size |
| Exp 2 | RULER NIAH | 4K–64K | Qwen3-30B-A3B | context length |

---

## Next Steps

1. Finish baseline GSM8K run (in progress)
2. Pull model pairs on both devices for Experiment 1
3. ~~Write `evals/ruler_niah.yaml` for Experiment 2~~ ✓ Done
4. Run RULER NIAH sweeps on both Mac and DGX Spark (in progress)
5. Build results comparison table once all runs complete
