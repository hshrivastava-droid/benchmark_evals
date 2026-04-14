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

## Experiment 1 — GSM8K Baseline (Qwen3-30B)

**Task:** GSM8K (math word problems, ~500 tokens/question). Short-context accuracy baseline — KV cache not stressed, pure model quality signal.

| Config | Mac (MLX) | DGX Spark (Ollama) |
|--------|-----------|-------------------|
| Model | `mlx-community/Qwen3-30B-A3B-4bit` | `qwen3:30b` |
| Port | 8080 | 11434 |
| Few-shot | 4 | 4 |
| Max tokens | 8192 | 8192 |
| Questions | 1319 (full set) | 1319 (full set) |
| Runtime | 14h 23m | 13h 42m |

```bash
# Mac (MLX)
./run.sh --model Qwen/Qwen3-30B --port 8080 \
  --eval-as mlx-community/Qwen3-30B-A3B-4bit \
  --eval-only --gen-max-tokens 8192 \
  --eval-concurrent 1 --num-fewshot 4

# DGX Spark (Ollama)
./run.sh --model Qwen/Qwen3-30B --port 11434 \
  --eval-as qwen3:30b \
  --eval-only --gen-max-tokens 8192 \
  --eval-concurrent 1 --num-fewshot 4
```

### Results

| Metric | Mac (MLX) | DGX Spark (Ollama) | Delta |
|--------|-----------|-------------------|-------|
| **strict-match** | 0.9249 (±0.0073) | **0.9653** (±0.0007) | **+4.0%** |
| **flexible-extract** | 0.9242 (±0.0073) | **0.9653** (±0.0007) | **+4.1%** |

**Conclusion:** DGX Spark (Ollama, Q4_K_M) outperforms Mac (MLX, uniform 4-bit) by ~4% on GSM8K with identical settings (4-shot, 1319 questions). This confirms that Q4_K_M mixed-precision quantization preserves more model quality than MLX's uniform 4-bit — even on a short-context task where memory pressure is minimal.

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

## Experiment 3 — Larger Model: Qwen3-Coder 80B

Scale up to a larger coding-focused model to test whether the DGX advantage grows with model size. Bigger models stress memory bandwidth and KV cache harder — the DGX Spark's dedicated VRAM should pull further ahead.

| Config | DGX Spark (Ollama) | Mac M4 (MLX) |
|--------|-------------------|--------------|
| Model | `qwen3-coder:80b` (GGUF Q4_K_M) | `mlx-community/Qwen3-Coder-80B-4bit` (MLX 4-bit) |
| Port | 11434 | 8081 |
| Tasks | GSM8K + RULER NIAH | GSM8K + RULER NIAH |

### Why This Model
- 80B parameters puts real pressure on both platforms — Mac's 128GB unified memory will be near capacity at 4-bit (~40GB weights + KV cache)
- Coding models are increasingly used for agentic workflows where long-context retrieval matters
- Tests whether the ~5% GSM8K gap from Experiment 1 widens at larger scale

### Run Commands

```bash
# DGX Spark — GSM8K
./run.sh --model Qwen/Qwen3-Coder-80B --port 11434 \
  --eval-as qwen3-coder:80b \
  --eval-only --gen-max-tokens 8192 \
  --eval-concurrent 1 --num-fewshot 4

# Mac M4 — GSM8K
./run.sh --model mlx-community/Qwen3-Coder-80B-4bit --port 8081 \
  --eval-only --gen-max-tokens 8192 \
  --eval-concurrent 1 --num-fewshot 4

# DGX Spark — NIAH 32K
./run.sh --model Qwen/Qwen3-Coder-80B --port 11434 \
  --eval-as qwen3-coder:80b \
  --eval-only --task ruler_niah_32k \
  --gen-max-tokens 32768 --eval-concurrent 1 --num-fewshot 0

# Mac M4 — NIAH 32K
./run.sh --model mlx-community/Qwen3-Coder-80B-4bit --port 8081 \
  --eval-only --task ruler_niah_32k \
  --gen-max-tokens 32768 --eval-concurrent 1 --num-fewshot 0
```

### Results

| Task | DGX Spark | Mac M4 | Notes |
|------|-----------|--------|-------|
| GSM8K | pending | pending | |
| NIAH 32K | pending | pending | |

---

## Summary Table

| Experiment | Task | Model | Variable | Status |
|------------|------|-------|----------|--------|
| Exp 1 | GSM8K | Qwen3-30B-A3B | device/framework | ✓ Done — DGX +4% |
| Exp 2 | RULER NIAH | Qwen3-30B-A3B | context length | In progress |
| Exp 3 | GSM8K + NIAH | Qwen3-Coder-80B | model scale | Planned |

---

## Next Steps

1. ~~GSM8K baseline with Qwen3-30B~~ ✓ Done — DGX 96.5% vs Mac 92.4%
2. Complete RULER NIAH sweeps on both Mac and DGX Spark (in progress)
3. Verify Qwen3-Coder-80B availability on Ollama and MLX
4. Run Experiment 3 on both platforms
5. Build final results comparison
