"""Run the benchmark against the real agent and write a reproducible report.

    python -m eval.run_eval                 # all tasks
    python -m eval.run_eval --only fastapi-vs-flask
    python -m eval.run_eval --limit 3 --pause 30

Uses the same orchestrator wiring as the API (real LLM + search providers from .env), so it
spends free-tier quota: about 12-15 LLM calls and 8-12 searches per task. Tasks run one at
a time with a pause between them to stay inside per-minute rate limits. Results are written
to eval/results/ as JSON (full scores) and Markdown (readable summary).
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from agentops.contracts import RunState
from agentops.orchestrator import EventBus

from .metrics import EvalTask, RunScore, score_run, summarize

TASKS_PATH = Path(__file__).parent / "tasks.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_tasks(path: Path = TASKS_PATH) -> list[EvalTask]:
    return [EvalTask.model_validate(item) for item in json.loads(path.read_text("utf-8"))]


def run_task(task: EvalTask) -> tuple[RunState, float]:
    from api.dependencies import build_orchestrator  # imported lazily: needs .env keys

    state = RunState(objective=task.objective)
    orchestrator = build_orchestrator(EventBus())
    started = time.monotonic()
    orchestrator.run(state)
    return state, time.monotonic() - started


_COLUMNS = (
    "Task",
    "Status",
    "Subtasks",
    "Evid. cov.",
    "Cit. resolved",
    "Supported",
    "Topics",
    "Tools ok",
    "Retries",
    "Calls (LLM/tool)",
    "Tokens in/out",
    "Time (s)",
)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render_markdown(scores: list[RunScore], summary: dict, stamp: str) -> str:
    lines = [
        f"# AgentOps evaluation: {stamp}",
        "",
        f"{summary['n_tasks']} task(s). Grounding metrics are structural: a cited id exists in",
        "the collected evidence. They are not a semantic entailment check.",
        "",
        "| " + " | ".join(_COLUMNS) + " |",
        "|" + "---|" * len(_COLUMNS),
    ]
    for s in scores:
        lines.append(
            f"| {s.task_id} | {s.status} | {_fmt(s.subtask_completion)} "
            f"| {_fmt(s.evidence_coverage)} | {_fmt(s.citation_resolution)} "
            f"| {_fmt(s.supported_rate)} | {_fmt(s.topic_coverage)} "
            f"| {_fmt(s.tool_success_rate)} | {s.retries} | {s.llm_calls}/{s.tool_calls} "
            f"| {s.tokens_in}/{s.tokens_out} | {s.wall_s} |"
        )
    lines += ["", "## Summary", ""]
    lines += [f"- **{key}**: {value}" for key, value in summary.items() if key != "n_tasks"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the AgentOps benchmark.")
    parser.add_argument("--only", action="append", default=[], help="task id (repeatable)")
    parser.add_argument("--limit", type=int, default=0, help="run only the first N tasks")
    parser.add_argument("--pause", type=float, default=20.0, help="seconds between tasks")
    args = parser.parse_args(argv)

    tasks = load_tasks()
    if args.only:
        tasks = [t for t in tasks if t.id in args.only]
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        print("no tasks selected", file=sys.stderr)
        return 2

    scores: list[RunScore] = []
    for index, task in enumerate(tasks):
        print(f"[{index + 1}/{len(tasks)}] {task.id} ...", flush=True)
        try:
            state, wall_s = run_task(task)
        except Exception as exc:  # a crashed task is a result, not a reason to lose the others
            print(f"  crashed: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        score = score_run(state, task, wall_s)
        scores.append(score)
        print(f"  {score.status} in {score.wall_s}s, {score.llm_calls} LLM calls", flush=True)
        if index < len(tasks) - 1:
            time.sleep(args.pause)

    if not scores:
        return 1
    summary = summarize(scores)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {"stamp": stamp, "summary": summary, "scores": [s.model_dump() for s in scores]}
    (RESULTS_DIR / f"{stamp}.json").write_text(json.dumps(payload, indent=2), "utf-8")
    markdown = render_markdown(scores, summary, stamp)
    (RESULTS_DIR / "latest.md").write_text(markdown, "utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
