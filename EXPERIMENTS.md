# Benchmark Experiments: Mac (MLX) vs DGX Spark (Ollama)

## What Are We Testing

Accuracy and quality of LLM inference across two fundamentally different hardware and runtime stacks:

- **Mac (Apple Silicon)** — MLX runtime, unified memory, Metal GPU, `mlx-community` 4-bit quantized models
- **DGX Spark** — llama.cpp via Ollama, dedicated VRAM, CUDA, GGUF Q4_K_M quantized models

Same model family, same quantization level, same eval harness, same prompts — isolating the hardware and framework as the variable.

---

## Experiment 1 — GSM8K Baseline (Qwen3-30B)

Short-context math accuracy baseline (~500 tokens/question). KV cache not stressed, pure model quality signal.

| Config | Mac (MLX) | DGX Spark (Ollama) |
|--------|-----------|-------------------|
| Model | `mlx-community/Qwen3-30B-A3B-4bit` | `qwen3:30b` |
| Port | 8080 | 11434 |
| Few-shot | 4 | 4 |
| Max tokens | 8192 | 8192 |
| Questions | 1319 (full set) | 1319 (full set) |

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

---

## Experiment 2 — RULER Needle in a Haystack (NIAH)

Long-context retrieval at 4K–64K tokens. A secret key is buried in noise text; the model must recall it exactly. Thinking mode disabled (`/no_think`) — NIAH is pure retrieval, not reasoning.

| Config | DGX Spark (Ollama) | Mac M4 (MLX) |
|--------|-------------------|--------------|
| Model | `qwen3:30b-a3b` (GGUF Q4_K_M) | `mlx-community/Qwen3-30B-A3B-4bit` (MLX 4-bit) |
| Port | 11434 | 8081 |
| Samples | 500 per context length | 500 per context length |
| Needle depths | 10%, 25%, 50%, 75%, 90% | 10%, 25%, 50%, 75%, 90% |
| Context lengths | 4K, 8K, 16K, 32K, 64K | 4K, 8K, 16K, 32K, 64K |

```bash
python3 gen_ruler_niah.py --samples 500    # generate test data (one-time)

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

Available tasks: `ruler_niah_4k` (4096), `ruler_niah_8k` (8192), `ruler_niah_16k` (16384), `ruler_niah_32k` (32768), `ruler_niah_64k` (65536).

---

## Experiment 3 — Larger Model: Qwen3-Coder 80B

Scale up to an 80B coding model. Bigger models stress memory bandwidth and KV cache harder — tests whether the DGX advantage widens with model size.

| Config | DGX Spark (Ollama) | Mac M4 (MLX) |
|--------|-------------------|--------------|
| Model | `qwen3-coder:80b` (GGUF Q4_K_M) | `mlx-community/Qwen3-Coder-80B-4bit` (MLX 4-bit) |
| Port | 11434 | 8081 |
| Tasks | GSM8K + RULER NIAH | GSM8K + RULER NIAH |

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

---

## Summary

| Experiment | Task | Model | Variable | Status |
|------------|------|-------|----------|--------|
| Exp 1 | GSM8K | Qwen3-30B-A3B | device/framework | ✓ Done |
| Exp 2 | RULER NIAH | Qwen3-30B-A3B | context length | In progress |
| Exp 3 | GSM8K + NIAH | Qwen3-Coder-80B | model scale | Planned |
