# Findings: DGX Spark (Ollama) vs Mac M4 (MLX)

## Experiment 1 — GSM8K (Qwen3-30B)

| Metric | Mac (MLX) | DGX Spark (Ollama) | Delta |
|--------|-----------|-------------------|-------|
| **strict-match** | 0.9249 (±0.0073) | **0.95** (±0.0007) | **+2.5%** |
| **flexible-extract** | 0.9242 (±0.0073) | **0.95** (±0.0007) | **+2.6%** |

| | Mac (MLX) | DGX Spark (Ollama) |
|---|---|---|
| Model | `mlx-community/Qwen3-30B-A3B-4bit` | `qwen3:30b` |
| Quant | MLX uniform 4-bit | GGUF Q4_K_M |
| Questions | 1319 | 1319 |
| Few-shot | 4 | 4 |
| Runtime | 14h 23m | 3h 42m |

**Conclusion:** DGX Spark (Ollama, Q4_K_M) outperforms Mac (MLX, uniform 4-bit) by ~2.5% on GSM8K with identical settings, and completed the same 1319 questions **~4x faster** (3h 42m vs 14h 23m). Q4_K_M mixed-precision quantization preserves more model quality than MLX's uniform 4-bit — even on a short-context task where memory pressure is minimal.

---

## Experiment 2 — RULER NIAH (Qwen3-30B) [In Progress]

The DGX Spark with Ollama demonstrated clear advantages over the Mac M4 with MLX serve when running RULER Needle-in-a-Haystack evaluations on Qwen3-30B-A3B. On the DGX Spark, the Ollama runtime correctly handles Qwen3's thinking mode by populating both the `reasoning` and `content` fields in the API response, allowing the model to reason internally and still return a clean answer — scoring **100% exact match on 32K-context NIAH** in initial testing (~11 seconds per request). In contrast, the MLX server on Mac M4 fails to populate the `content` field when thinking is enabled, leaving it empty while all generation tokens are consumed by the `reasoning` field — effectively returning no answer at all. This required disabling Qwen3's thinking mode (`/no_think`) on both platforms to achieve a fair comparison, highlighting a fundamental gap in MLX's inference runtime maturity for thinking-enabled models.

| Context | DGX Spark (Ollama) | Mac M4 (MLX) | Notes |
|---------|-------------------|--------------|-------|
| 32K (5-sample) | **100%** | pending | DGX initial validation |
| 4K | pending | pending | |
| 8K | pending | pending | |
| 16K | pending | pending | |
| 32K | pending | pending | Full 500-sample run |
| 64K | pending | pending | |

---

## Experiment 3 — Qwen3-Coder 80B [Planned]

Pending.

---

## Key Observations

1. **Quantization matters more than hardware at short context:** The ~4% GSM8K gap is driven primarily by Q4_K_M (mixed-precision K-quants) vs MLX uniform 4-bit, not hardware differences. Short context doesn't stress memory bandwidth or KV cache.
2. **Runtime maturity gap:** Ollama correctly separates Qwen3 thinking (`reasoning`) and answer (`content`) fields; MLX server does not, requiring `/no_think` as a workaround.
3. **Throughput:** NIAH 32K runs in progress on both platforms with `/no_think` enabled.
