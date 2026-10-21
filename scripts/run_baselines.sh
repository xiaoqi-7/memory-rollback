#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() {
    echo "Usage: bash scripts/run_baselines.sh [dataset] [model] [baseline] [case selector]"
    echo "Arguments may appear in any order."
    echo "  dataset:  controlled | controlled_customer | controlled_shopping | controlled_travel | lmev2"
    echo "  model:    gpt | qwen | gemini"
    echo "  baseline: no_repair | full_memory_reset | delete_retrieved_memories | llm_judge_repair | memaudit_style | agenttrace_style | all"
    echo "  case:     one | case=<case_id|task_id> | <case_id> | <task_id>"
    echo "Defaults: dataset=controlled, model=gpt, baseline=all, case=all"
}

init_argument_defaults
requested_baseline="all"
baseline_seen=0
for argument in "$@"; do
    case "$argument" in
        -h|--help) usage; exit 0 ;;
        all|no_repair|full_memory_reset|delete_retrieved_memories|llm_judge_repair|memaudit_style|agenttrace_style)
            [[ "$baseline_seen" -eq 0 ]] || die "baseline specified more than once"
            requested_baseline="$argument"
            baseline_seen=1
            ;;
        *)
            parse_common_argument "$argument" \
                || die "unknown argument '$argument'"
            ;;
    esac
done

resolve_common_arguments

case "$requested_baseline" in
    all) selected_baselines="$BASELINES" ;;
    no_repair|full_memory_reset|delete_retrieved_memories|llm_judge_repair|memaudit_style|agenttrace_style)
        selected_baselines="$requested_baseline"
        ;;
    *) die "unknown baseline '$requested_baseline'" ;;
esac

if [[ "$requested_baseline" != "no_repair" ]]; then
    require_llm_credentials 1
fi

case_ids=()
if ! is_full_registry_selection || [[ "$CASE_SELECTOR" != "all" ]]; then
    while IFS= read -r case_id; do
        [[ -n "$case_id" ]] && case_ids+=("$case_id")
    done < <(selected_case_ids)
    [[ "${#case_ids[@]}" -gt 0 ]] || die "no registry cases selected for $DATASET"
fi

for baseline in $selected_baselines; do
    echo "=== baseline ${baseline}: ${DATASET} [${LLM_BACKBONE}] ==="
    command=(
        "$PYTHON_BIN" -m baselines.run_baseline
        --baseline "$baseline"
        --registry "$REGISTRY_PATH"
        --output_dir "${OUTPUT_ROOT}/${LLM_BACKBONE}/{method}/{domain}/{fault_types}/{task_id}/{faulty_memory_ids}"
    )
    if is_full_registry_selection && [[ "$CASE_SELECTOR" == "all" ]]; then
        command+=(--all_cases)
    else
        command+=(--case_id "${case_ids[@]}")
    fi
    "${command[@]}"
done
