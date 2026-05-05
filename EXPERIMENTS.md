# Benchmark Experiments: Mac (MLX) vs DGX Spark (Ollama)

## What Are We Testing

Accuracy and quality of LLM inference across two fundamentally different hardware and runtime stacks:

- **Mac (Apple Silicon)** — MLX runtime, unified memory, Metal GPU, `mlx-community` 4-bit quantized models
- **DGX Spark** — llama.cpp via Ollama, dedicated VRAM, CUDA, GGUF Q4_K_M quantized models

Same model family, same quantization level, same eval harness, same prompts — isolating the hardware and framework as the variable.

---

## Experiment 1 — GSM8K Baseline (Qwen3-30B)

Short-context math accuracy baseline (~500 tokens/question). KV cache not stressed, pure model quality signal.

| Config | Mac (MLX) | DGX Spark (Ollama) | DGX Spark (vLLM, NVFP4) |
|--------|-----------|--------------------|--------------------------|
| Model | `mlx-community/Qwen3-30B-A3B-4bit` | `qwen3:30b` | `nvidia/Qwen3-30B-A3B-NVFP4` |
| Runtime | MLX 4-bit | Ollama (GGUF Q4_K_M) | vLLM v0.19+ (NVFP4 + FP8 KV) |
| Container | — | — | `nvcr.io/nvidia/vllm:26.04-py3` |
| Port | 8080 | 11434 | 8888 |
| Few-shot | 4 | 4 | 4 |
| Max tokens | 8192 | 8192 | 8192 |
| Questions | 1319 (full set) | 1319 (full set) | 1319 (full set) |

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

# DGX Spark (vLLM, NVFP4)
# 1) Start the container (host shell):
docker run -it --gpus all --ipc=host -p 8888:8888 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  --name hs_vllm \
  nvcr.io/nvidia/vllm:26.04-py3

# 2) Inside the container, launch vLLM in tmux:
tmux new -s vllm
vllm serve nvidia/Qwen3-30B-A3B-NVFP4 \
  --quantization modelopt_fp4 \
  --served-model-name qwen3-30b-a3b-nvfp4 \
  --host 0.0.0.0 --port 8888 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.75 \
  --trust-remote-code \
  2>&1 | tee /workspace/vllm_server_$(date +%Y%m%d_%H%M%S).log
# Detach: Ctrl+b d

# 3) Run the eval (host or container):
./run.sh --model nvidia/Qwen3-30B-A3B-NVFP4 --port 8888 \
  --eval-as qwen3-30b-a3b-nvfp4 \
  --eval-only --gen-max-tokens 8192 \
  --eval-concurrent 1 --num-fewshot 4
```

### DGX Spark vLLM gotchas

- **Unified memory:** GPU and system RAM share the same ~119 GiB pool. vLLM's default `--gpu-memory-utilization 0.9` reserves ~107 GiB and starves the OS, causing swap and 2× pace degradation. **Always cap at `0.75`** on DGX Spark.
- **Run vLLM inside `tmux`** — SSH drops/Ctrl+C from a bare shell can orphan `EngineCore` and pin GPU memory.
- **Memory-bandwidth-bound, not compute-bound:** `nvidia-smi` reports ~96% GPU utilization at only ~9 W power draw — SMs stall on LPDDR5X memory loads (vs HBM3 on H100, ~12× faster). Steady decode rate is ~25 tok/s at concurrency=1.
- **Bump concurrency for real throughput:** at `--eval-concurrent 1`, full 1319 takes ~12 h. KV cache stays at 0.1% used (186× headroom). At `--eval-concurrent 8`, expect ~3 h with same accuracy.
- **vLLM build matters:** `v0.17.1+...nv26.03` crashed mid-eval with `cudaErrorIllegalInstruction` (NVFP4 + FP8-KV + cudagraph kernel bug). `v0.19.0+...nv26.04` runs the same workload to completion stably. Stick to `v0.19+`.

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
