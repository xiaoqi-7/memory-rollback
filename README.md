### Overview

This repository contains the code and data supplement for *“From Faulty Memories to Corrected Actions: Dependency-Guided Rollback Repair for Memory-Augmented Agents.”*

The main directories are organized as follows:

```text
memory_rollback_code_data_supplement/
├── agent/          # Agent runtime and memory/trace implementation
├── methods/ours/   # Our Dependency-Guided Rollback Repair method
├── baselines/      # Baseline implementations
├── eval/           # Oracle checks and evaluation metrics
├── simulator/      # Domain tools
├── configs/        # Domain, method, and baseline configurations
├── client/         # LLM and embedding API clients
├── scripts/        # Scripts for running experiments and evaluations
└── data/
    ├── controlled_benchmark/
    └── adapted_lmev2_subset/
```

The implemented baselines include:

- No Repair
- Full Memory Reset
- Delete Retrieved Memories
- LLM-Judge Repair
- MemAudit-Style
- AgentTrace-Style

### Setup

The code was tested with Python 3.12.9. Install the required dependencies using:

```bash
python -m pip install -r requirements.txt
```

### API Configuration

Use `.env.example` as the template for `.env`:

```bash
cp .env.example .env
```

Fill in the required API keys. `OPENAI_API_KEY` is used for GPT-4o, and `GEMINI_API_KEY` is used for Gemini-3.6-Flash. `QWEN_API_KEY` is used for both Qwen3.6-27B and `text-embedding-v4`, which supports agent memory retrieval. Therefore, `QWEN_API_KEY` is required for all agent-based experiments, while `OPENAI_API_KEY` or `GEMINI_API_KEY` is required when the corresponding backbone is selected.

The default temperature is `0`, with seed `42` for GPT-4o and Qwen3.6-27B. Gemini-3.6-Flash does not expose a seed parameter.

The code automatically loads `.env` from the project root. To test the configured LLMs, run:

```bash
python -m client.model_client --model gpt-4o --prompt "Hi"
python -m client.model_client --model gemini --prompt "Hi"
python -m client.model_client --model qwen --prompt "Hi"
```

### Running Experiments

#### Quick Start

Run the complete experiment with the default settings: all 150 controlled benchmark cases with GPT-4o, including our method, all baselines, all ablations, and evaluation.

```bash
bash scripts/run_ours.sh
bash scripts/run_baselines.sh
bash scripts/run_ablations.sh
bash scripts/run_eval.sh
```

#### Our Method

Available datasets: `controlled` for all 150 cases in the controlled benchmark (default); `controlled_customer` for its 53 customer support cases; `controlled_shopping` for its 48 shopping cases; `controlled_travel` for its 49 travel cases; and `lmev2` for 50 cases from the adapted LongMemEval-V2 subset.

Available LLM backbones: `gpt` for GPT-4o (default); `qwen` for Qwen3.6-27B; and `gemini` for Gemini-3.6-Flash.

Template:

```bash
bash scripts/run_ours.sh [dataset] [backbone]
```

Example:

```bash
bash scripts/run_ours.sh controlled gpt
```

Results are written under:

```text
outputs/<backbone>/ours/<domain>/
```

Existing completed cases are skipped automatically.

#### Baselines

Available baselines: `no_repair`, `full_memory_reset`, `delete_retrieved_memories`, `llm_judge_repair`, `memaudit_style`, and `agenttrace_style`. Use `all` to run all baselines (default).

Template:

```bash
bash scripts/run_baselines.sh [dataset] [backbone] [baseline]
```

Example:

```bash
bash scripts/run_baselines.sh lmev2 qwen no_repair
```

Results are written under:

```text
outputs/<backbone>/<baseline>/<domain>/
```

#### Ablations

Available ablations: `skip_support_check`, `skip_rollback_planner`, and `skip_selective_replay`. Use `all` to run all ablation settings (default).

Template:

```bash
bash scripts/run_ablations.sh [dataset] [backbone] [ablation]
```

Example:

```bash
bash scripts/run_ablations.sh lmev2 qwen all
```

Results are written under:

```text
outputs/<backbone>/<ablation>/<domain>/
```

#### Evaluation

Available methods: `ours`, `no_repair`, `full_memory_reset`, `delete_retrieved_memories`, `llm_judge_repair`, `memaudit_style`, `agenttrace_style`, `skip_support_check`, `skip_rollback_planner`, and `skip_selective_replay`. Use `all` to evaluate all methods with available results (default).

Template:

```bash
bash scripts/run_eval.sh [dataset] [backbone] [method]
```

Example:

```bash
bash scripts/run_eval.sh controlled gpt ours
```

Evaluation outputs are written under:

```text
eval_outputs/<backbone>/<method>/<domain>/
```