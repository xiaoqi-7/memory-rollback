#!/usr/bin/env bash

# Shared dataset, model, path, and registry helpers for the experiment scripts.
# This file is sourced by the four runnable scripts; do not execute it directly.

set -euo pipefail

SUPPLEMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SUPPLEMENT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-eval_outputs}"

BASELINES="no_repair full_memory_reset delete_retrieved_memories llm_judge_repair memaudit_style agenttrace_style"
ABLATIONS="skip_support_check skip_rollback_planner skip_selective_replay"
EVAL_METHODS="ours ${BASELINES} ${ABLATIONS}"

die() {
    echo "error: $*" >&2
    exit 2
}

require_python() {
    command -v "$PYTHON_BIN" >/dev/null 2>&1 \
        || die "Python executable not found: $PYTHON_BIN"
}

init_argument_defaults() {
    DATASET_ARG="controlled"
    BACKBONE_ARG="gpt"
    CASE_SELECTOR_ARG="all"
    DATASET_ARG_SEEN=0
    BACKBONE_ARG_SEEN=0
    CASE_SELECTOR_ARG_SEEN=0
}

set_dataset_arg() {
    [[ "$DATASET_ARG_SEEN" -eq 0 ]] || die "dataset specified more than once"
    DATASET_ARG="$1"
    DATASET_ARG_SEEN=1
}

set_backbone_arg() {
    [[ "$BACKBONE_ARG_SEEN" -eq 0 ]] || die "model specified more than once"
    BACKBONE_ARG="$1"
    BACKBONE_ARG_SEEN=1
}

set_case_selector_arg() {
    [[ "$CASE_SELECTOR_ARG_SEEN" -eq 0 ]] || die "case selector specified more than once"
    CASE_SELECTOR_ARG="$1"
    [[ -n "$CASE_SELECTOR_ARG" ]] || die "case selector must not be empty"
    CASE_SELECTOR_ARG_SEEN=1
}

parse_common_argument() {
    local value="$1"
    case "$value" in
        controlled|controlled_customer|controlled_shopping|controlled_travel|lmev2)
            set_dataset_arg "$value"
            return 0
            ;;
        gpt|qwen|gemini)
            set_backbone_arg "$value"
            return 0
            ;;
        one)
            set_case_selector_arg "one"
            return 0
            ;;
        case=*)
            set_case_selector_arg "${value#case=}"
            return 0
            ;;
        */*|*_task_*)
            set_case_selector_arg "$value"
            return 0
            ;;
    esac
    return 1
}

resolve_common_arguments() {
    resolve_dataset "$DATASET_ARG"
    resolve_backbone "$BACKBONE_ARG"
    CASE_SELECTOR="$CASE_SELECTOR_ARG"
}

resolve_dataset() {
    DATASET="$1"
    case "$DATASET" in
        controlled)
            REGISTRY_PATH="data/controlled_benchmark/registry.jsonl"
            DATA_ROOT="data/controlled_benchmark"
            DOMAIN_FILTER="customer_support,shopping,travel"
            DOMAINS="customer_support shopping travel"
            ;;
        controlled_customer)
            REGISTRY_PATH="data/controlled_benchmark/registry.jsonl"
            DATA_ROOT="data/controlled_benchmark"
            DOMAIN_FILTER="customer_support"
            DOMAINS="customer_support"
            ;;
        controlled_shopping)
            REGISTRY_PATH="data/controlled_benchmark/registry.jsonl"
            DATA_ROOT="data/controlled_benchmark"
            DOMAIN_FILTER="shopping"
            DOMAINS="shopping"
            ;;
        controlled_travel)
            REGISTRY_PATH="data/controlled_benchmark/registry.jsonl"
            DATA_ROOT="data/controlled_benchmark"
            DOMAIN_FILTER="travel"
            DOMAINS="travel"
            ;;
        lmev2)
            REGISTRY_PATH="data/adapted_lmev2_subset/registry.jsonl"
            DATA_ROOT="data/adapted_lmev2_subset"
            DOMAIN_FILTER="lme_derived"
            DOMAINS="lme_derived"
            ;;
        *)
            die "unknown dataset '$DATASET' (expected controlled, controlled_customer, controlled_shopping, controlled_travel, or lmev2)"
            ;;
    esac

    [[ -f "$REGISTRY_PATH" ]] || die "registry not found: $REGISTRY_PATH"
}

resolve_backbone() {
    LLM_BACKBONE="$1"
    case "$LLM_BACKBONE" in
        gpt)
            export LLM_MODEL="gpt-4o"
            ;;
        qwen)
            export LLM_MODEL="qwen3.6-27b"
            ;;
        gemini)
            export LLM_MODEL="gemini-3.6-flash"
            ;;
        *)
            die "unknown model '$LLM_BACKBONE' (expected gpt, qwen, or gemini)"
            ;;
    esac

    # The scripts select provider-specific credentials and endpoints. Avoid a
    # generic override silently routing a requested backbone to another API.
    unset LLM_API_KEY LLM_BASE_URL || true
}

require_llm_credentials() {
    local require_embedding="${1:-1}"
    require_python
    "$PYTHON_BIN" - "$require_embedding" <<'PY'
import sys

from client.model_client import _get_llm_client_settings, get_qwen_api_key

settings = _get_llm_client_settings()
if sys.argv[1] == "1":
    get_qwen_api_key()
print(
    "LLM profile: {} (remote model: {})".format(
        settings.profile,
        settings.model,
    )
)
PY
}

domain_config() {
    case "$1" in
        customer_support) echo "configs/customer_support.yaml" ;;
        shopping) echo "configs/shopping.yaml" ;;
        travel) echo "configs/travel.yaml" ;;
        lme_derived) echo "configs/lme_derived.yaml" ;;
        *) die "unknown domain '$1'" ;;
    esac
}

domain_ground_truth() {
    local domain="$1"
    echo "${DATA_ROOT}/${domain}/ground_truth.jsonl"
}

is_full_registry_selection() {
    [[ "$DATASET" == "controlled" || "$DATASET" == "lmev2" ]]
}

registry_case_manifest() {
    require_python
    "$PYTHON_BIN" - "$REGISTRY_PATH" "$DOMAIN_FILTER" "${CASE_SELECTOR:-all}" <<'PY'
import json
import sys
from pathlib import Path

from methods.ours.run_ours import resolve_case_output_dir

registry_path = Path(sys.argv[1])
domains = {value for value in sys.argv[2].split(",") if value}
selector = sys.argv[3]
selected_count = 0

with registry_path.open(encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        domain = row.get("domain")
        if domain not in domains:
            continue

        paths = row.get("paths")
        if not isinstance(paths, dict):
            raise ValueError(f"{registry_path}:{line_number}: missing paths")

        def resolve_data_path(name: str) -> str:
            value = paths.get(name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"{registry_path}:{line_number}: missing paths.{name}"
                )
            path = Path(value)
            return str(path if path.is_absolute() else registry_path.parent / path)

        task_id = row.get("task_id")
        faulty_ids = row.get("faulty_memory_ids")
        fault_types = row.get("fault_types")
        case_id = row.get("case_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"{registry_path}:{line_number}: invalid task_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{registry_path}:{line_number}: invalid case_id")
        if not isinstance(faulty_ids, list) or not faulty_ids:
            raise ValueError(
                f"{registry_path}:{line_number}: invalid faulty_memory_ids"
            )
        if not isinstance(fault_types, list) or not fault_types:
            raise ValueError(f"{registry_path}:{line_number}: invalid fault_types")

        if selector == "one" and selected_count:
            continue
        if selector not in ("all", "one") and selector not in (case_id, task_id):
            continue

        run_ours = row.get("run_ours")
        agent_config = (
            run_ours.get("agent_config")
            if isinstance(run_ours, dict)
            else None
        )
        if not isinstance(agent_config, str) or not agent_config:
            agent_config = f"configs/{domain}.yaml"

        case_suffix = resolve_case_output_dir(
            "{domain}/{fault_types}/{task_id}/{faulty_memory_ids}",
            domain=domain,
            task_id=task_id,
            faulty_memory_ids=faulty_ids,
            fault_types=fault_types,
        )
        fields = (
            case_id,
            domain,
            task_id,
            ",".join(faulty_ids),
            resolve_data_path("faulty_memories"),
            resolve_data_path("faulty_tracesteps"),
            resolve_data_path("session"),
            agent_config,
            case_suffix,
        )
        if any("\t" in value or "\n" in value for value in fields):
            raise ValueError(
                f"{registry_path}:{line_number}: tab/newline in manifest field"
            )
        print("\t".join(fields))
        selected_count += 1

if selector != "all" and selected_count == 0:
    raise ValueError(
        f"case selector {selector!r} did not match the selected dataset/domains"
    )
if selector not in ("all", "one") and selected_count > 1:
    raise ValueError(
        f"case selector {selector!r} matched multiple cases; use the full case_id"
    )
PY
}

selected_case_ids() {
    registry_case_manifest | "$PYTHON_BIN" -c \
        'import sys; [print(line.split("\t", 1)[0]) for line in sys.stdin if line.strip()]'
}

run_ours_cases() {
    local method_dir="$1"
    shift
    local output_template="${OUTPUT_ROOT}/${LLM_BACKBONE}/${method_dir}/{domain}/{fault_types}/{task_id}/{faulty_memory_ids}"
    local count=0
    local skipped=0
    local case_id domain task_id faulty_csv memories steps session agent_config case_suffix

    while IFS=$'\t' read -r case_id domain task_id faulty_csv memories steps session agent_config case_suffix; do
        [[ -n "$case_id" ]] || continue
        count=$((count + 1))
        local result_path="${OUTPUT_ROOT}/${LLM_BACKBONE}/${method_dir}/${case_suffix}/repaired_result.json"
        if [[ -f "$result_path" ]]; then
            echo "skip existing: $result_path"
            skipped=$((skipped + 1))
            continue
        fi

        local -a faulty_ids=()
        IFS=',' read -r -a faulty_ids <<< "$faulty_csv"
        echo "=== ${method_dir}: ${case_id} [${LLM_BACKBONE}] ==="
        local -a command=(
            "$PYTHON_BIN" -m methods.ours.run_ours
            --memories "$memories" \
            --steps "$steps" \
            --session "$session" \
            --task_id "$task_id" \
            --faulty_memory_ids "${faulty_ids[@]}" \
            --config configs/ours_base.yaml \
            --agent_config "$agent_config" \
            --output_dir "$output_template"
        )
        if [[ "$#" -gt 0 ]]; then
            command+=("$@")
        fi
        "${command[@]}"
    done < <(registry_case_manifest)

    [[ "$count" -gt 0 ]] || die "no registry cases selected for $DATASET"
    echo "completed ${method_dir}: ${count} selected, ${skipped} already existed"
}

result_tree_has_cases() {
    local path="$1"
    [[ -d "$path" ]] || return 1
    "$PYTHON_BIN" - "$path" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
raise SystemExit(0 if next(root.rglob("repaired_result.json"), None) else 1)
PY
}

select_result_input() {
    local path="$1"
    local selector="${2:-all}"
    "$PYTHON_BIN" - "$path" "$selector" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
selector = sys.argv[2]
files = sorted(root.rglob("repaired_result.json")) if root.is_dir() else []
if not files:
    raise SystemExit(1)
if selector == "all":
    print(root)
    raise SystemExit(0)
if selector == "one":
    print(files[0])
    raise SystemExit(0)

matches = []
for path in files:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    case = result.get("case")
    if not isinstance(case, dict):
        continue
    if selector in (case.get("case_id"), case.get("task_id")):
        matches.append(path)
if not matches:
    raise SystemExit(1)
if len(matches) > 1:
    raise SystemExit(
        f"case selector {selector!r} matched multiple repaired results under {root}"
    )
print(matches[0])
PY
}

result_case_tag() {
    local result_path="$1"
    "$PYTHON_BIN" - "$result_path" <<'PY'
import json
import re
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
case = result.get("case") if isinstance(result, dict) else None
value = case.get("task_id") if isinstance(case, dict) else None
if not isinstance(value, str) or not value:
    value = case.get("case_id") if isinstance(case, dict) else "case"
print(re.sub(r"[^A-Za-z0-9._+-]+", "_", str(value)).strip("._") or "case")
PY
}
