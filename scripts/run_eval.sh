#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() {
    echo "Usage:"
    echo "  bash scripts/run_eval.sh"
    echo "  bash scripts/run_eval.sh all"
    echo "  bash scripts/run_eval.sh [dataset] [model] [method] [case selector]"
    echo ""
    echo "Arguments may appear in any order."
    echo "  dataset: controlled | controlled_customer | controlled_shopping | controlled_travel | lmev2"
    echo "  model:   gpt | qwen | gemini"
    echo "  method:  ours | no_repair | full_memory_reset | delete_retrieved_memories | llm_judge_repair | memaudit_style | agenttrace_style | skip_support_check | skip_rollback_planner | skip_selective_replay | all"
    echo "  case:    one | case=<case_id|task_id> | <case_id> | <task_id>"
    echo "Defaults for a partial command: dataset=controlled, model=gpt, method=all, case=all"
    echo "With no arguments, or the sole argument 'all', evaluate every existing result combination."
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

validated_backbones=""
evaluated_count=0

ensure_eval_credentials() {
    case " $validated_backbones " in
        *" $LLM_BACKBONE "*) return ;;
    esac
    require_llm_credentials 1
    validated_backbones="${validated_backbones} ${LLM_BACKBONE}"
}

validate_eval_method() {
    case "$1" in
        ours|no_repair|full_memory_reset|delete_retrieved_memories|llm_judge_repair|memaudit_style|agenttrace_style|skip_support_check|skip_rollback_planner|skip_selective_replay)
            ;;
        *) die "unknown evaluation method '$1'" ;;
    esac
}

eval_combination() {
    local dataset="$1"
    local backbone="$2"
    local method="$3"
    local require_results="$4"
    local selector="${5:-all}"
    local found=0

    resolve_dataset "$dataset"
    resolve_backbone "$backbone"
    validate_eval_method "$method"

    for domain in $DOMAINS; do
        local results_dir="${OUTPUT_ROOT}/${LLM_BACKBONE}/${method}/${domain}"
        local result_input
        if ! result_input="$(select_result_input "$results_dir" "$selector")"; then
            continue
        fi
        found=1
        evaluated_count=$((evaluated_count + 1))
        ensure_eval_credentials

        local config
        local ground_truth
        local eval_dir
        config="$(domain_config "$domain")"
        ground_truth="$(domain_ground_truth "$domain")"
        if [[ "$selector" == "all" ]]; then
            eval_dir="${EVAL_OUTPUT_ROOT}/${LLM_BACKBONE}/${method}/${domain}"
        else
            local case_tag
            case_tag="$(result_case_tag "$result_input")"
            eval_dir="${EVAL_OUTPUT_ROOT}/${LLM_BACKBONE}/${method}/${domain}/single/${case_tag}"
        fi

        echo "=== eval ${method}: ${domain} [${LLM_BACKBONE}] ==="
        "$PYTHON_BIN" -m eval.run_eval_suite \
            --ground-truth "$ground_truth" \
            --results "$result_input" \
            --config "$config" \
            --output-dir "$eval_dir" \
            --use-oracle-judgments \
            --resume-recurrence
    done

    if [[ "$found" -eq 0 && "$require_results" == "1" ]]; then
        die "no repaired results found for dataset=$dataset model=$backbone method=$method"
    fi
}

if [[ "$#" -eq 0 || ( "$#" -eq 1 && "$1" == "all" ) ]]; then
    for dataset in controlled lmev2; do
        for backbone in gpt qwen gemini; do
            for method in $EVAL_METHODS; do
                eval_combination "$dataset" "$backbone" "$method" 0 all
            done
        done
    done
    if [[ "$evaluated_count" -eq 0 ]]; then
        echo "no existing repaired-result combinations found under ${OUTPUT_ROOT}/"
    else
        echo "evaluated ${evaluated_count} existing model/method/domain combinations"
    fi
    exit 0
fi

init_argument_defaults
requested_method="all"
method_seen=0
for argument in "$@"; do
    case "$argument" in
        -h|--help) usage; exit 0 ;;
        all|ours|no_repair|full_memory_reset|delete_retrieved_memories|llm_judge_repair|memaudit_style|agenttrace_style|skip_support_check|skip_rollback_planner|skip_selective_replay)
            [[ "$method_seen" -eq 0 ]] || die "evaluation method specified more than once"
            requested_method="$argument"
            method_seen=1
            ;;
        *)
            parse_common_argument "$argument" \
                || die "unknown argument '$argument'"
            ;;
    esac
done

resolve_common_arguments

case "$DATASET" in
    controlled|controlled_customer|controlled_shopping|controlled_travel|lmev2) ;;
    *) die "unknown evaluation dataset '$DATASET'" ;;
esac

start_count="$evaluated_count"
if [[ "$requested_method" == "all" ]]; then
    for method in $EVAL_METHODS; do
        eval_combination "$DATASET" "$LLM_BACKBONE" "$method" 0 "$CASE_SELECTOR"
    done
    if [[ "$evaluated_count" -eq "$start_count" ]]; then
        die "no repaired results found for dataset=$DATASET model=$LLM_BACKBONE method=all case=$CASE_SELECTOR"
    fi
else
    eval_combination "$DATASET" "$LLM_BACKBONE" "$requested_method" 1 "$CASE_SELECTOR"
fi
echo "evaluated ${evaluated_count} model/method/domain combinations"
