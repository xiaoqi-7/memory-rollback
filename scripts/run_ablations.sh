#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() {
    echo "Usage: bash scripts/run_ablations.sh [dataset] [model] [ablation] [case selector]"
    echo "Arguments may appear in any order."
    echo "  dataset:  controlled | controlled_customer | controlled_shopping | controlled_travel | lmev2"
    echo "  model:    gpt | qwen | gemini"
    echo "  ablation: skip_support_check | skip_rollback_planner | skip_selective_replay | all"
    echo "  case:     one | case=<case_id|task_id> | <case_id> | <task_id>"
    echo "Defaults: dataset=controlled, model=gpt, ablation=all, case=all"
}

init_argument_defaults
requested_ablation="all"
ablation_seen=0
for argument in "$@"; do
    case "$argument" in
        -h|--help) usage; exit 0 ;;
        all|skip_support_check|skip_rollback_planner|skip_selective_replay)
            [[ "$ablation_seen" -eq 0 ]] || die "ablation specified more than once"
            requested_ablation="$argument"
            ablation_seen=1
            ;;
        *)
            parse_common_argument "$argument" \
                || die "unknown argument '$argument'"
            ;;
    esac
done

resolve_common_arguments

case "$requested_ablation" in
    all) selected_ablations="$ABLATIONS" ;;
    skip_support_check|skip_rollback_planner|skip_selective_replay)
        selected_ablations="$requested_ablation"
        ;;
    *) die "unknown ablation '$requested_ablation'" ;;
esac

require_llm_credentials 1

for ablation in $selected_ablations; do
    case "$ablation" in
        skip_support_check)
            override="ablation.skip_support_check=true"
            ;;
        skip_rollback_planner)
            override="ablation.skip_rollback_optimizer=true"
            ;;
        skip_selective_replay)
            override="ablation.skip_selective_replay=true"
            ;;
    esac
    run_ours_cases "$ablation" --set "$override"
done
