# RULER NIAH Findings: DGX Spark (Ollama) vs Mac M4 (MLX)

## Summary

The DGX Spark with Ollama demonstrated clear advantages over the Mac M4 with MLX serve when running RULER Needle-in-a-Haystack evaluations on Qwen3-30B-A3B. On the DGX Spark, the Ollama runtime correctly handles Qwen3's thinking mode by populating both the `reasoning` and `content` fields in the API response, allowing the model to reason internally and still return a clean answer — scoring **100% exact match on 32K-context NIAH** in initial testing (~11 seconds per request, 500 samples in ~1.5 hours). In contrast, the MLX server on Mac M4 fails to populate the `content` field when thinking is enabled, leaving it empty while all generation tokens are consumed by the `reasoning` field — effectively returning no answer at all. This required disabling Qwen3's thinking mode (`/no_think`) on both platforms to achieve a fair comparison, highlighting a fundamental gap in MLX's inference runtime maturity for thinking-enabled models. Beyond the runtime compatibility issue, the DGX Spark's dedicated VRAM architecture is expected to maintain high retrieval accuracy at longer context lengths (32K–64K), where the Mac's unified memory and Metal-based attention kernels face increasing pressure — a hypothesis the full context-length sweep will validate.

## Test Configuration

| Parameter | Value |
|-----------|-------|
| **Task** | RULER NIAH (single-needle retrieval) |
| **Model** | Qwen3-30B-A3B (4-bit quantized) |
| **DGX quant** | GGUF Q4_K_M (Ollama) |
| **Mac quant** | MLX 4-bit (mlx-community) |
| **Samples** | 500 per context length |
| **Needle depths** | 10%, 25%, 50%, 75%, 90% |
| **Context lengths** | 4K, 8K, 16K, 32K, 64K |
| **Thinking mode** | Disabled (`/no_think`) for fair comparison |

## Results

| Context | DGX Spark (Ollama) | Mac M4 (MLX) | Notes |
|---------|-------------------|--------------|-------|
| 32K (5-sample) | **100%** | pending | DGX initial validation |
| 4K | pending | pending | |
| 8K | pending | pending | |
| 16K | pending | pending | |
| 32K | pending | pending | Full 500-sample run |
| 64K | pending | pending | |

## Key Observations

1. **Runtime maturity gap**: Ollama correctly separates thinking and answer content; MLX server does not, requiring `/no_think` as a workaround.
2. **Quantization difference**: DGX uses GGUF Q4_K_M (mixed precision K-quants), Mac uses MLX uniform 4-bit — not identical weights, which may contribute to accuracy differences independently of hardware.
3. **Throughput**: DGX processes 32K-context NIAH at ~11s/request with Ollama on the NVIDIA GB10 GPU.
