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
                      content = msg.get("reasoning_content") or ""
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

  local task_yaml="${SCRIPT_DIR}/evals/${task}.yaml"
  if [[ ! -f "$task_yaml" ]]; then
    echo "Error: task file not found: ${task_yaml}" >&2
    return 1
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
    --tasks "${task_yaml}" \
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
