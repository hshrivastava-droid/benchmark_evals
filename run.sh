#!/usr/bin/env bash
# Standalone: throughput benchmark + lm-eval against one already-running OpenAI-compatible server.
#
# Usage (from bench_serving/, after server is up on this node):
#   ./run.sh --model "Qwen/Qwen3.5-397B-A17B-FP8"
#   ./run.sh --model "Qwen/..." --port 8888 --served-model-name "custom-name"
#   ./run.sh --model "..." --eval-only
#   ./run.sh --model "..." --bench-only
#
# Slurm: second container with --container-mounts="$GITHUB_WORKSPACE:/workspace", then:
#   cd /workspace && ./run.sh --model "Qwen/Qwen3.5-397B-A17B-FP8" --port 8888

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
export SCRIPT_DIR

# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib_lm_eval.sh"

MODEL=""
HOST="127.0.0.1"
PORT="8888"
BASE_URL=""
SERVED_MODEL_NAME=""
EVAL_MODEL=""
EVAL_TASK="gsm8k"
NUM_FEWSHOT=""
EVAL_CONCURRENT="64"
GEN_MAX_TOKENS=""
BENCH_RESULT_DIR=""
DO_BENCH=true
DO_EVAL=true

usage() {
  cat <<'EOF'
Usage: ./run.sh --model <hf-model-id> [options]

  Start the inference server first (same node: default client URL is http://127.0.0.1:8888).

Options:
  --model ID              Required. HF id for tokenizer + default API "model" field.
  --host HOST             Server hostname for client (default: 127.0.0.1).
  --port N                Server port (default: 8888).
  --base-url URL          Full URL http://HOST:PORT (overrides --host/--port).
  --served-model-name S   Pass-through to throughput client if API name differs.
  --eval-as NAME          lm-eval "model" in API (default: same as --model).
  --task TASK             evals/<TASK>.yaml (default: gsm8k).
  --num-fewshot N
  --eval-concurrent N
  --gen-max-tokens N      Eval context budget (default: 16384 in lib_lm_eval).
  --results-dir DIR       Throughput JSON output (default: ./results).
  --bench-only            Throughput only.
  --eval-only             lm-eval only.
  -h, --help
EOF
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --served-model-name) SERVED_MODEL_NAME="$2"; shift 2 ;;
    --eval-as) EVAL_MODEL="$2"; shift 2 ;;
    --task) EVAL_TASK="$2"; shift 2 ;;
    --num-fewshot) NUM_FEWSHOT="$2"; shift 2 ;;
    --eval-concurrent) EVAL_CONCURRENT="$2"; shift 2 ;;
    --gen-max-tokens) GEN_MAX_TOKENS="$2"; shift 2 ;;
    --results-dir) BENCH_RESULT_DIR="$2"; shift 2 ;;
    --bench-only) DO_EVAL=false; shift ;;
    --eval-only) DO_BENCH=false; shift ;;
    -h|--help) usage 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage 1
      ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Error: --model is required (Hugging Face id the client uses for tokenizer and default API name)." >&2
  usage 1
fi

if [[ -n "$BASE_URL" ]]; then
  if [[ "$BASE_URL" =~ ^https?://([^:/]+):([0-9]+)(/|$) ]]; then
    HOST="${BASH_REMATCH[1]}"
    PORT="${BASH_REMATCH[2]}"
  else
    echo "Error: --base-url must look like http://HOST:PORT" >&2
    exit 1
  fi
else
  BASE_URL="http://${HOST}:${PORT}"
fi

[[ -n "$BENCH_RESULT_DIR" ]] || BENCH_RESULT_DIR="${SCRIPT_DIR}/results"
mkdir -p "$BENCH_RESULT_DIR"

export MODEL
export MODEL_NAME="${EVAL_MODEL:-$MODEL}"
export EVAL_TASK

concurrency_levels=(4 8 16 32 64 128 256 512)

if [[ "$DO_BENCH" == true ]]; then
  echo "=== Throughput (same server: ${BASE_URL}) ==="
  echo "model=${MODEL} served_model=${SERVED_MODEL_NAME:-"(same as model)"}"

  for conc in "${concurrency_levels[@]}"; do
    num_prompts=$((conc * 10))
    echo "Benchmark concurrency=${conc} num_prompts=${num_prompts}"

    bench_cmd=(
      python3 "$SCRIPT_DIR/benchmark_serving.py"
      --model "$MODEL"
      --backend vllm
      --base-url "$BASE_URL"
      --dataset-name random
      --random-input-len 1024
      --random-output-len 1024
      --random-range-ratio 0.8
      --num-prompts "$num_prompts"
      --max-concurrency "$conc"
      --request-rate inf
      --ignore-eos
      --save-result
      --num-warmups "$((2 * conc))"
      --percentile-metrics 'ttft,tpot,itl,e2el'
      --result-dir "$BENCH_RESULT_DIR"
      --result-filename "throughput_conc${conc}.json"
      --trust-remote-code
    )
    [[ -z "$SERVED_MODEL_NAME" ]] || bench_cmd+=(--served-model-name "$SERVED_MODEL_NAME")
    "${bench_cmd[@]}"

    echo "Completed throughput conc=${conc}"
    echo "-----------------------------------------"
  done
  echo "Throughput JSON: ${BENCH_RESULT_DIR}/throughput_conc*.json"
fi

if [[ "$DO_EVAL" == true ]]; then
  echo "=== lm-eval (${BASE_URL}/v1/chat/completions) model=${MODEL_NAME} task=${EVAL_TASK} ==="
  eval_results="$(mktemp -d /tmp/bench_serving_eval-XXXXXX)"
  eval_cmd=(
    bench_serving_run_lm_eval
    --host "$HOST"
    --port "$PORT"
    --task "$EVAL_TASK"
    --concurrent-requests "$EVAL_CONCURRENT"
    --results-dir "$eval_results"
  )
  [[ -z "$NUM_FEWSHOT" ]] || eval_cmd+=(--num-fewshot "$NUM_FEWSHOT")
  [[ -z "$GEN_MAX_TOKENS" ]] || eval_cmd+=(--gen-max-tokens "$GEN_MAX_TOKENS")
  "${eval_cmd[@]}"
  echo "lm-eval output: ${eval_results}"
fi

echo "Done."
