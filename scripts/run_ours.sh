#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() {
    echo "Usage: bash scripts/run_ours.sh [dataset] [model] [case selector]"
    echo "Arguments may appear in any order."
    echo "  dataset: controlled | controlled_customer | controlled_shopping | controlled_travel | lmev2"
    echo "  model:   gpt | qwen | gemini"
    echo "  case:    one | case=<case_id|task_id> | <case_id> | <task_id> | all"
    echo "Defaults: dataset=controlled, model=gpt, case=all"
}

init_argument_defaults
for argument in "$@"; do
    case "$argument" in
        -h|--help) usage; exit 0 ;;
        all) set_case_selector_arg "all" ;;
        *)
            parse_common_argument "$argument" \
                || die "unknown argument '$argument'"
            ;;
    esac
done

resolve_common_arguments
require_llm_credentials 1
run_ours_cases "ours"
