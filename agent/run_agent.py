'''
Run the REACT agent over user inputs.

    # clean mode: one isolated session (multiple turns share memory)
    python -m agent.run_agent --config configs/shopping.yaml -p "hi" -p "buy milk"

    # clean mode: a single user input
    python -m agent.run_agent --config configs/shopping.yaml --single -p "hi"

    # faulty mode: replay every faulted task from the injected fault inputs
    python -m agent.run_agent --config configs/shopping.yaml --mode faulty

`clean` mode runs with an empty memory store (no injected faults); memories
accumulate only from the agent's own writes across the session.

`faulty` mode (memory-injection scenario) replays the injected-fault benchmark.
For each faulted task it loads the seeded memory store and the cut trace steps
from `<in_dir>` (default `<data_source>/faulty_mid`), restores them into a fresh
agent, then runs that task's remaining user inputs (the `session_turns_to_run`)
against the shared store so the agent reasons over the faulty memories. The
complete memories and trace steps produced are concatenated across tasks and
written to `<out_dir>` (default `<data_source>/faulty`) as
`faulty_memories.jsonl` and `faulty_tracesteps.jsonl`.

`respond_single` answers one user input; `respond_session` answers an isolated
session's user inputs (a `list[str]`) in order, sharing memory across turns.
Optionally restores prior-turn memories first; neither writes artifacts to disk.
'''


import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import importlib.util

import yaml

from .react_agent import ReactAgent


_SUPPLEMENT_ROOT = Path(__file__).resolve().parents[1]


class AgentRunner:
    """High-level runner that loads config, initializes components, and
    iterates over task sessions.
    """

    def __init__(self, config_path: str, mode: str = "clean") -> None:
        """Create a runner for `config_path` in the chosen `mode`."""
        if mode not in ("clean", "faulty"):
            raise ValueError(f"Unknown mode: {mode!r} (expected 'clean' or 'faulty')")

        self.mode = mode
        self.config_path = str(self._resolve_config_path(config_path))
        self.config = self._load_config(self.config_path)
        self.data_source = self._resolve_data_source()
        self.config["data_source"] = self.data_source
        self.tasks_source = self._resolve_tasks_source()
        self.tools = self._load_tools()
        self.agent = ReactAgent(self.tools, self.config)

    @staticmethod
    def _resolve_config_path(config_path: str) -> Path:
        """Resolve relative CLI paths from the supplement root."""
        path = Path(config_path).expanduser()
        if path.is_absolute():
            return path.resolve()
        supplement_path = _SUPPLEMENT_ROOT / path
        if supplement_path.exists():
            return supplement_path.resolve()
        return path.resolve()

    @staticmethod
    def _resolve_supplement_path(value: str) -> str:
        """Resolve paths stored in supplement configs from the supplement root."""
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = _SUPPLEMENT_ROOT / path
        return str(path.resolve())

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError(
                f"Config file must contain a mapping, got {type(config).__name__}"
            )

        return config

    def _resolve_tasks_source(self) -> str:
        # tasks_full_session.jsonl lives at the root of data_source, not in a
        # mode-specific subdirectory.
        return os.path.join(self.data_source, "tasks_full_session.jsonl")

    def _resolve_data_source(self) -> str:
        """Return the absolute data_source directory from the config."""
        data_source = self.config.get("data_source")
        if not isinstance(data_source, str) or not data_source:
            raise KeyError("Missing 'data_source' in config file")
        return self._resolve_supplement_path(data_source)

    def _load_tools(self) -> Dict[str, Any]:
        """Load the domain's tool callables from the TOOLS registry."""
        tool_source = self.config.get("available_tools_source")
        if tool_source is None:
            raise KeyError("Missing 'available_tools_source' in config file")

        tool_source = self._resolve_supplement_path(tool_source)
        self.config["available_tools_source"] = tool_source

        if not os.path.exists(tool_source):
            raise FileNotFoundError(f"Tool source not found: {tool_source}")

        spec = importlib.util.spec_from_file_location("tools_module", tool_source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "TOOLS"):
            raise ValueError(f"TOOLS dictionary not found in {tool_source}")

        domain = self.config.get("domain", "")
        if hasattr(module, "configure_data_source"):
            module.configure_data_source(domain, self.data_source)

        # TOOLS expected format: {"shopping": {"search_product": func, ...}, ...}
        tools_registry = module.TOOLS
        if domain not in tools_registry:
            raise ValueError(
                f"Domain '{domain}' not found in TOOLS. "
                f"Available: {list(tools_registry.keys())}"
            )
        return tools_registry[domain]

    def load_tasks(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.tasks_source):
            raise FileNotFoundError(f"Task source not found: {self.tasks_source}")

        tasks: List[Dict[str, Any]] = []
        with open(self.tasks_source, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    task = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_no} of {self.tasks_source}: {exc}"
                    ) from exc
                tasks.append(task)

        return tasks

    def clear(self) -> None:
        """Reset the agent's memory store and trace logger for a new task."""
        self.agent.clear()

    def run_single(self, task_id: str, prompt: str) -> Dict[str, Any]:
        """Process a single prompt within a task session."""
        return self.agent.run(task_id, prompt)

    def respond(
        self,
        task_id: str,
        prompt: str,
        session_memories: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Respond to one user input given the current session's memories.

        Restores `session_memories` (serialized dicts, e.g. from a prior
        turn's `memory_store.get_memories()`) into a fresh store, then runs the
        REACT loop for `prompt`. Memories written during this turn are appended
        to the store, so the returned `memories` reflect the updated session
        state and can be passed back in on the next call.
        """
        self.clear()
        if session_memories:
            self.agent.memory_store.load_memories(session_memories)

        result = self.run_single(task_id, prompt)
        result["memories"] = self.agent.memory_store.get_memories()
        result["trace_steps"] = self.agent.trace_logger.get_trace_steps()
        return result

    def run_full_session(self, task_id: str, session: List[str]) -> List[Dict[str, Any]]:
        """Process a full session of prompts in order and return the results.

        Memories written during earlier turns are available to later turns
        because the store is shared across the session (cleared per task).
        """
        results = []
        for prompt in session:
            results.append(self.run_single(task_id, prompt))
        return results

    def respond_full_session(
        self,
        task_id: str,
        session: List[str],
        session_memories: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Respond to one isolated session's user inputs given in order.

        Restores `session_memories` into a fresh store, then runs every prompt
        in `session` turn by turn against the shared store, so memories written
        in earlier turns are visible to later turns. Returns the per-turn
        results plus the final response and the updated session memories and
        traces.
        """
        self.clear()
        if session_memories:
            self.agent.memory_store.load_memories(session_memories)

        session_results = self.run_full_session(task_id, session)
        return {
            "session_results": session_results,
            "response": session_results[-1]["response"] if session_results else None,
            "tool_calls": session_results[-1].get("tool_calls") if session_results else None,
            "memories": self.agent.memory_store.get_memories(),
            "trace_steps": self.agent.trace_logger.get_trace_steps(),
        }


    def respond_faulty_session(
        self,
        task_id: str,
        session_turns: List[str],
        faulty_memories: List[Dict[str, Any]],
        cut_steps: Optional[List[Dict[str, Any]]] = None,
        replay_start_turn: int = 1,
    ) -> Dict[str, Any]:
        """Replay one faulted task from its injected-fault inputs.

        Restores the seeded ``faulty_memories`` (the reconstructed store at the
        injection point, including the faulty record) and the ``cut_steps`` (the
        trace steps that ran before the replay boundary) into a fresh agent, then
        runs every prompt in ``session_turns`` — the task's ``session_turns_to_run``
        — turn by turn against the shared store, so the agent reasons over the
        faulty memories. ``replay_start_turn`` is the 1-based turn the replay
        resumes from; the agent's turn counter is offset so a memory written on
        the first replayed turn is authored by ``user_input_{replay_start_turn}``,
        matching the clean trace's turn numbering. Returns the complete memories
        and trace steps (loaded + newly produced).
        """
        self.clear()
        if faulty_memories:
            self.agent.memory_store.load_memories(faulty_memories)
        if cut_steps:
            self.agent.trace_logger.load(cut_steps)
            # Rebuild the session-scoped read-only call guard from the cut trace
            # so a replayed turn that re-plans a read-only call already made
            # before the boundary is dropped (forcing reuse of the memory
            # written from it) rather than re-fetched.
            self.agent.restore_executed_from_trace()

        # Continue the session's turn numbering from the replay boundary so the
        # user-input ids minted for written memories line up with the clean trace.
        self.agent.task_id = task_id
        self.agent.turn_index = max(0, replay_start_turn - 1)

        session_results = self.run_full_session(task_id, session_turns)
        return {
            "session_results": session_results,
            "response": session_results[-1]["response"] if session_results else None,
            "tool_calls": session_results[-1].get("tool_calls") if session_results else None,
            "memories": self.agent.memory_store.get_memories(),
            "trace_steps": self.agent.trace_logger.get_trace_steps(),
        }


    def run_faulty(
        self,
        in_dir: str,
        out_dir: str,
    ) -> Dict[str, Any]:
        """Replay every faulted task from the injected-fault inputs in `in_dir`.

        Reads the three per-task input files written by ``data/inject_faults.py``
        — ``cut_tracesteps.jsonl`` (the steps before the replay boundary),
        ``faulty_memories.jsonl`` (the seeded store with the faulty record), and
        ``faulty_sessions.jsonl`` (each task's ``session_turns_to_run`` plus its
        ``replay_start_turn``) — and joins them by ``task_id``. For each task it
        restores the memories and cut steps, replays the remaining user inputs
        against the shared store, and collects the complete memories and trace
        steps. All tasks' outputs are concatenated and written to
        ``<out_dir>/faulty_memories.jsonl`` and
        ``<out_dir>/faulty_tracesteps.jsonl``.
        """
        cut_by_task = _index_by_task(_read_jsonl(os.path.join(in_dir, "cut_tracesteps.jsonl")))
        mem_by_task = _index_by_task(_read_jsonl(os.path.join(in_dir, "faulty_memories.jsonl")))
        sess_by_task = _index_by_task(_read_jsonl(os.path.join(in_dir, "faulty_sessions.jsonl")))

        if not sess_by_task:
            raise FileNotFoundError(
                f"No faulty sessions found in {os.path.join(in_dir, 'faulty_sessions.jsonl')}"
            )

        all_memories: List[Dict[str, Any]] = []
        all_traces: List[Dict[str, Any]] = []

        for task_id, sess in sess_by_task.items():
            session_turns = sess.get("session_turns_to_run") or []
            if not session_turns:
                print(f"skipping {task_id}: no session_turns_to_run")
                continue
            replay_start_turn = sess.get("replay_start_turn", 1)

            mem_row = mem_by_task.get(task_id, {})
            faulty_memories = mem_row.get("memories") or []
            cut_row = cut_by_task.get(task_id, {})
            cut_steps = cut_row.get("cut_steps") or []

            result = self.respond_faulty_session(
                task_id,
                session_turns,
                faulty_memories,
                cut_steps=cut_steps,
                replay_start_turn=replay_start_turn,
            )
            memories = result.get("memories", [])
            traces = result.get("trace_steps", [])
            all_memories.extend(memories)
            all_traces.extend(traces)
            print(f"{task_id}: {len(memories)} memories, {len(traces)} trace steps")

        memories_path = os.path.join(out_dir, "faulty_memories.jsonl")
        traces_path = os.path.join(out_dir, "faulty_tracesteps.jsonl")
        _write_jsonl(memories_path, all_memories)
        _write_jsonl(traces_path, all_traces)
        print(
            f"\n=== wrote {len(all_memories)} memories to {memories_path} ==="
            f"\n=== wrote {len(all_traces)} trace steps to {traces_path} ==="
        )
        return {
            "tasks": len(sess_by_task),
            "memories_path": memories_path,
            "traces_path": traces_path,
            "memory_count": len(all_memories),
            "trace_count": len(all_traces),
        }


def respond_session(
    config_path: str,
    task_id: str,
    session: List[str],
    session_memories: List[Dict[str, Any]] | None = None,
    mode: str = "clean",
) -> Dict[str, Any]:
    """Respond to one isolated session's user inputs given as a list.

    Builds a runner from `config_path`, restores `session_memories` (the
    serialized memory dicts carried over from a prior session), and runs the
    REACT loop for each prompt in `session` in order against a shared store, so
    memories written in earlier turns are visible to later turns. Returns the
    per-turn results plus the final response and the updated `memories` and
    `trace_steps`.

    Like `respond_single`, this writes no artifacts to disk; managing session
    state is left to the caller.
    """
    runner = AgentRunner(config_path, mode)
    return runner.respond_full_session(task_id, session, session_memories)


def respond_single(
    config_path: str,
    task_id: str,
    prompt: str,
    session_memories: List[Dict[str, Any]] | None = None,
    mode: str = "clean",
) -> Dict[str, Any]:
    """Respond to a single user input using the current session's memories.

    Builds a runner from `config_path`, restores `session_memories` (the
    serialized memory dicts carried over from earlier turns), and runs the
    REACT loop once for `prompt`. Returns the turn result augmented with the
    updated `memories` and `trace_steps`, so the caller can persist them and
    feed `memories` back on the next call to continue the session.

    Unlike `run_from_config`, this writes no artifacts to disk; managing
    session state is left to the caller.
    """
    runner = AgentRunner(config_path, mode)
    return runner.respond(task_id, prompt, session_memories)


def run_faulty(
    config_path: str,
    in_dir: Optional[str] = None,
    out_dir: Optional[str] = None,
    mode: str = "faulty",
) -> Dict[str, Any]:
    """Replay the injected-fault benchmark for a config and write the outputs.

    Builds a runner from `config_path`, resolves the default `in_dir`
    (`<data_source>/faulty_mid`) and `out_dir` (`<data_source>/faulty`) when not
    given, then replays every faulted task and writes the complete memories and
    trace steps. Returns a summary with the output paths and per-task counts.
    """
    runner = AgentRunner(config_path, mode)
    data_source = runner._resolve_data_source()
    in_dir = in_dir or os.path.join(data_source, "faulty_mid")
    out_dir = out_dir or os.path.join(data_source, "faulty")
    return runner.run_faulty(in_dir, out_dir)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts. Returns [] if the file is absent."""
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return rows


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write `rows` to `path` as JSONL, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _index_by_task(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index per-task rows by ``task_id`` (last row wins on duplicates)."""
    return {row.get("task_id"): row for row in rows if row.get("task_id") is not None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config file.")
    parser.add_argument("--mode", "-m", default="clean", help="Execution mode: 'clean' or 'faulty' (default: clean).")
    parser.add_argument(
        "--single", action="store_true",
        help="Clean mode only: test respond_single (one user input). Without "
             "it, test respond_session (an isolated session's user inputs in order).",
    )
    parser.add_argument(
        "--prompt", "-p", action="append", default=[],
        help="Clean mode: a user input. Repeat for multiple turns of a session; "
             "in --single mode only the first is used.",
    )
    parser.add_argument(
        "--task-id", "-t", default="task_adhoc",
        help="Clean mode: task id to attribute the response to (default: task_adhoc).",
    )
    parser.add_argument(
        "--in-dir",
        help="Faulty mode: directory holding cut_tracesteps.jsonl, "
             "faulty_memories.jsonl and faulty_sessions.jsonl. Defaults to "
             "<data_source>/faulty_mid.",
    )
    parser.add_argument(
        "--out-dir",
        help="Faulty mode: directory to write faulty_memories.jsonl and "
             "faulty_tracesteps.jsonl. Defaults to <data_source>/faulty.",
    )
    args = parser.parse_args()

    if args.mode == "faulty":
        run_faulty(args.config, args.in_dir, args.out_dir, args.mode)
        return

    if not args.prompt:
        parser.error("at least one --prompt is required in clean mode")

    if args.single:
        result = respond_single(
            args.config, args.task_id, args.prompt[0], None, args.mode
        )
    else:
        result = respond_session(
            args.config, args.task_id, args.prompt, None, args.mode
        )

    _print_result(result)


def _print_result(result: Dict[str, Any]) -> None:
    """Pretty-print a response result for the CLI.

    Memories are printed to the console; trace steps are written to
    ``agent/temp_tracesteps.jsonl`` (one JSON object per line) instead of being
    printed, so the full trace can be inspected as an artifact.
    """
    memories = result.get("memories", [])
    memory_path = os.path.join(os.path.dirname(__file__), "temp_memories.jsonl")
    with open(memory_path, "w", encoding="utf-8") as f:
        for memory in memories:
            f.write(json.dumps(memory, ensure_ascii=False) + "\n")
    print(f"\n=== wrote {len(memories)} memories to {memory_path} ===")

    trace_steps = result.get("trace_steps", [])
    trace_path = os.path.join(os.path.dirname(__file__), "temp_tracesteps.jsonl")
    with open(trace_path, "w", encoding="utf-8") as f:
        for trace in trace_steps:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
    print(f"\n=== wrote {len(trace_steps)} trace steps to {trace_path} ===")


if __name__ == "__main__":
    main()


__all__ = ["AgentRunner", "respond_single", "respond_session", "run_faulty"]
