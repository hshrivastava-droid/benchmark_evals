#!/usr/bin/env bash
# Standalone lm-eval helper for bench_serving/ (no InferenceX benchmark_lib.sh).
# Mirrors the install, patch, and invocation logic from benchmarks/benchmark_lib.sh
# so results are comparable without depending on that file.
# shellcheck shell=bash

_bench_serving_install_lm_eval() {
  python3 -m pip install -q --no-cache-dir "lm-eval[api]" || true

  local lm_eval_ref="b315ef3b05176acc9732bb7fdec116abe1ecc476"
  if command -v git >/dev/null 2>&1; then
    if ! python3 -m pip install -q --no-cache-dir --no-deps \
          "git+https://github.com/EleutherAI/lm-evaluation-harness.git@${lm_eval_ref}" 2>/dev/null; then
      python3 -m pip install -q --no-cache-dir --no-deps \
          "https://github.com/EleutherAI/lm-evaluation-harness/archive/${lm_eval_ref}.tar.gz" || true
    fi
  else
    python3 -m pip install -q --no-cache-dir --no-deps \
        "https://github.com/EleutherAI/lm-evaluation-harness/archive/${lm_eval_ref}.tar.gz" || true
  fi

  # IFEval verifier needs these (not pulled in by lm-eval[api]).
  python3 -m pip install -q --no-cache-dir nltk langdetect immutabledict || true
  python3 - <<'PY' 2>/dev/null || true
import nltk
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass
PY
}

# BFCL-v3 is not a first-party lm-eval task; use gorilla's official runner
# (bfcl-eval on PyPI). Installs the package and downloads test data on first use.
_bench_serving_install_bfcl() {
  python3 -m pip install -q --no-cache-dir bfcl-eval || \
    python3 -m pip install -q --no-cache-dir --break-system-packages bfcl-eval || true
}

# HF datasets + OpenAI client for SWE-bench Lite inference (no full ``swebench`` import; avoids Py version coupling).
_bench_serving_install_swe_bench_lite_inference_deps() {
  python3 -m pip install -q --no-cache-dir datasets openai tqdm numpy || \
    python3 -m pip install -q --no-cache-dir --break-system-packages datasets openai tqdm numpy || true
}

# Run BFCL-v3 against a local OpenAI-compatible chat endpoint.
# Args: task host port model_api results_dir concurrent_requests limit
_bench_serving_run_bfcl_task() {
  local task="$1" host="$2" port="$3" model_api="$4"
  local results_dir="$5" concurrent_requests="$6" limit="$7"

  _bench_serving_install_bfcl

  if ! command -v bfcl >/dev/null 2>&1; then
    echo "Error: 'bfcl' CLI not found after install. Install manually: pip install bfcl-eval" >&2
    return 1
  fi

  # gorilla's CLI reads the upstream URL from these env vars when --backend openai
  # is used with a non-OpenAI model handle.
  export OPENAI_BASE_URL="http://${host}:${port}/v1"
  export OPENAI_API_BASE="$OPENAI_BASE_URL"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

  # BFCL-v3 categories. "all" runs the full v3 suite (~4.4k entries across
  # single/multi-turn/parallel/irrelevance). Override via BFCL_TEST_CATEGORY.
  local test_category="${BFCL_TEST_CATEGORY:-all}"

  local gen_args=(
    generate
    --model "$model_api"
    --test-category "$test_category"
    --num-threads "$concurrent_requests"
    --result-dir "${results_dir}/bfcl_responses"
  )
  # --limit forwards as --num-tests (BFCL's per-category cap; ignored if
  # the installed bfcl-eval version doesn't expose this flag).
  [[ -z "$limit" ]] || gen_args+=(--num-tests "$limit")

  local eval_args=(
    evaluate
    --model "$model_api"
    --test-category "$test_category"
    --result-dir "${results_dir}/bfcl_responses"
    --score-dir "${results_dir}/bfcl_scores"
  )

  echo "=== BFCL ${task} | model=${model_api} | endpoint=${OPENAI_BASE_URL} | category=${test_category} ==="
  set -x
  bfcl "${gen_args[@]}"
  bfcl "${eval_args[@]}"
  local code=$?
  set +x
  return "$code"
}

# SWE-bench Lite: generate predictions JSONL via OpenAI-compatible Chat Completions.
# Evaluation is separate (Docker): python -m swebench.harness.run_evaluation ...
# Args match BFCL-style runner: task host port model_api results_dir concurrent_requests limit
_bench_serving_run_swe_bench_lite_task() {
  local task="$1" host="$2" port="$3" model_api="$4"
  local results_dir="$5" concurrent_requests="$6" limit="$7"

  _bench_serving_install_swe_bench_lite_inference_deps

  if ! python3 -c "import datasets, openai" 2>/dev/null; then
    echo "Error: install datasets and openai for SWE-bench Lite inference." >&2
    return 1
  fi

  local dataset="${SWE_BENCH_LITE_DATASET:-princeton-nlp/SWE-bench_Lite_oracle}"
  local out="${results_dir}/swe_bench_lite_predictions.jsonl"

  local cmd=(
    python3 "${SCRIPT_DIR}/run_swebench_lite_inference.py"
    --host "$host"
    --port "$port"
    --model "$model_api"
    --dataset-name "$dataset"
    --output-jsonl "$out"
    --max-workers "$concurrent_requests"
  )
  [[ -z "$limit" ]] || cmd+=(--limit "$limit")

  echo "=== SWE-bench Lite inference | model=${model_api} | base_url=http://${host}:${port}/v1 | dataset=${dataset} ==="
  echo "Writing: ${out}"
  echo "Evaluate (install swebench + Docker; dataset must match predictions instance_ids):"
  echo "  pip install 'swebench[datasets]>=3.0'"
  echo "  python3 -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Lite --predictions_path ${out} --max_workers 4 --run_id local_run"
  set -x
  "${cmd[@]}"
  local code=$?
  set +x
  return "$code"
}

_bench_serving_patch_lm_eval() {
  local patch_dir
  patch_dir="$(mktemp -d)"
  cat > "$patch_dir/sitecustomize.py" <<'PY'
import re, sys, unicodedata, json
from lm_eval.filters import extraction as ex
from lm_eval.models.openai_completions import LocalChatCompletion as _LCC

def _le_parse_generations(outputs, **kwargs):
      res = []
      if not isinstance(outputs, list):
          outputs = [outputs]
      for out in (outputs or []):
          try:
              choices = out.get("choices", [])
              tmp = ["" for _ in choices]
              for choice in choices:
                  idx = choice.get("index", 0)
                  msg = (choice.get("message") or {})
                  content = msg.get("content")
                  if content in (None, "", []):
                      content = msg.get("reasoning_content") or msg.get("reasoning") or ""
                  tmp[idx] = content
          except Exception:
              tmp = [""]
          res.extend(tmp)
      return res

_LCC.parse_generations = staticmethod(_le_parse_generations)

try:
    from lm_eval.models import api_models as _api_models
    _TemplateAPI = _api_models.TemplateAPI
    _JsonChatStr = _api_models.JsonChatStr
except Exception:
    _TemplateAPI = None
    _JsonChatStr = None

if _TemplateAPI is not None and _JsonChatStr is not None:
    _orig_apply_chat_template = _TemplateAPI.apply_chat_template

    def _patched_apply_chat_template(
        self,
        chat_history,
        add_generation_prompt: bool = True,
    ):
        if self.tokenizer_backend == "huggingface" and self.tokenized_requests:
            return self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )
        elif self.tokenizer_backend == "remote" and self.tokenized_requests:
            return chat_history
        else:
            return _JsonChatStr(
                json.dumps(
                    [{**item} for item in chat_history],
                    ensure_ascii=False,
                )
            )

    _TemplateAPI.apply_chat_template = _patched_apply_chat_template

PY
  export PYTHONPATH="${patch_dir}:${PYTHONPATH:-}"
}

_bench_serving_patch_pretty_print() {
  local tasks_init
  tasks_init="$(python3 -c "import lm_eval.tasks; print(lm_eval.tasks.__file__)")"
  if [[ -f "$tasks_init" ]] && grep -q 'task_manager.task_index\[task_name\]\["yaml_path"\]' "$tasks_init"; then
    sed -i.bak 's|task_manager\.task_index\[task_name\]\["yaml_path"\]|task_manager.task_index.get(task_name, {"yaml_path": "custom"})["yaml_path"]|' "$tasks_init"
    echo "Patched pretty_print_task in ${tasks_init}"
  fi

  local eval_utils
  eval_utils="$(python3 -c "import lm_eval.evaluator_utils; print(lm_eval.evaluator_utils.__file__)")"
  if [[ -f "$eval_utils" ]] && grep -q 'tab_string + alias' "$eval_utils"; then
    sed -i.bak 's|tab_string + alias|tab_string + (alias or name or "task")|' "$eval_utils"
    echo "Patched prepare_print_tasks in ${eval_utils}"
  fi
}

# Run lm-eval against a local OpenAI chat endpoint.
# Args: --port --task --num-fewshot --concurrent-requests --results-dir
#       --gen-max-tokens --temperature --top-p
# Requires: SCRIPT_DIR set to bench_serving directory; task YAML at "${SCRIPT_DIR}/evals/${task}.yaml"
bench_serving_run_lm_eval() {
  local port="${PORT:-8888}"
  local host="${LM_EVAL_HOST:-127.0.0.1}"
  local task="${EVAL_TASK:-gsm8k}"
  local num_fewshot=""
  local results_dir=""
  local eval_context_len=16384
  local temperature=0
  local top_p=1
  local concurrent_requests=64
  local limit=""

  while [[ $# -gt 0 ]]; do
    case $1 in
      --port) port="$2"; shift 2 ;;
      --host) host="$2"; shift 2 ;;
      --task) task="$2"; shift 2 ;;
      --num-fewshot) num_fewshot="$2"; shift 2 ;;
      --results-dir) results_dir="$2"; shift 2 ;;
      --gen-max-tokens) eval_context_len="$2"; shift 2 ;;
      --temperature) temperature="$2"; shift 2 ;;
      --top-p) top_p="$2"; shift 2 ;;
      --concurrent-requests) concurrent_requests="$2"; shift 2 ;;
      --limit) limit="$2"; shift 2 ;;
      *) echo "Unknown parameter: $1" >&2; return 1 ;;
    esac
  done

  if [[ -z "${MODEL_NAME:-}" && -z "${MODEL:-}" ]]; then
    echo "Error: set MODEL or MODEL_NAME for lm-eval." >&2
    return 1
  fi
  local model_api="${MODEL_NAME:-$MODEL}"

  if [[ -z "$results_dir" ]]; then
    results_dir="$(mktemp -d /tmp/bench_serving_lm_eval-XXXXXX)"
  fi
  mkdir -p "$results_dir"

  # BFCL-v3 isn't an lm-eval task — dispatch to gorilla's runner.
  case "$task" in
    bfcl|bfcl_v3)
      _bench_serving_run_bfcl_task "$task" "$host" "$port" "$model_api" \
        "$results_dir" "$concurrent_requests" "$limit"
      return $?
      ;;
    swe_bench_lite|swebench_lite|swe-bench-lite)
      _bench_serving_run_swe_bench_lite_task "$task" "$host" "$port" "$model_api" \
        "$results_dir" "$concurrent_requests" "$limit"
      return $?
      ;;
  esac

  # A local YAML in evals/ takes precedence (registered via --include_path
  # below). When absent, fall through to lm-evaluation-harness's built-in
  # task registry — covers names like mmlu_pro, ifeval, gpqa_diamond_*,
  # aime24, aime25.
  local task_yaml="${SCRIPT_DIR}/evals/${task}.yaml"
  if [[ ! -f "$task_yaml" ]]; then
    echo "No local YAML at ${task_yaml}; expecting built-in lm-eval task '${task}'."
  fi

  # Sensible default num_fewshot per built-in task if caller didn't pass one.
  if [[ -z "$num_fewshot" ]]; then
    case "$task" in
      mmlu_pro)     num_fewshot=5 ;;
      ifeval)       num_fewshot=0 ;;
      bfcl_v3|bfcl) num_fewshot=0 ;;
      aime|aime24|aime25|aime26) num_fewshot=0 ;;
      hle)          num_fewshot=0 ;;
    esac
  fi

  local max_output_tokens=$(( eval_context_len > 4096 ? eval_context_len - 4096 : eval_context_len / 2 ))
  if [ "$max_output_tokens" -gt 16384 ]; then
    max_output_tokens=16384
  fi
  echo "Eval budget: eval_context_len=${eval_context_len}, max_output_tokens=${max_output_tokens}"

  _bench_serving_install_lm_eval
  _bench_serving_patch_lm_eval
  _bench_serving_patch_pretty_print

  local openai_chat_base="http://${host}:${port}/v1/chat/completions"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

  local fewshot_args=()
  if [[ -n "$num_fewshot" ]]; then
    fewshot_args=(--num_fewshot "$num_fewshot")
  fi

  local limit_args=()
  if [[ -n "$limit" ]]; then
    limit_args=(--limit "$limit")
  fi

  set -x
  python3 -m lm_eval --model local-chat-completions --apply_chat_template \
    --include_path "${SCRIPT_DIR}/evals" \
    --tasks "${task}" \
    ${fewshot_args[@]+"${fewshot_args[@]}"} \
    ${limit_args[@]+"${limit_args[@]}"} \
    --output_path "${results_dir}" \
    --log_samples \
    --model_args "model=${model_api},base_url=${openai_chat_base},api_key=${OPENAI_API_KEY},eos_string=</s>,max_retries=5,num_concurrent=${concurrent_requests},timeout=1800,tokenized_requests=False,max_length=${eval_context_len}" \
    --gen_kwargs "max_tokens=${max_output_tokens},temperature=${temperature},top_p=${top_p}"
  local code=$?
  set +x
  return "$code"
}
