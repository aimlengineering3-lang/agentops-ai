# AgentOps AI

Autonomous research, analysis and decision-intelligence agent. Give it a complex objective;
it plans, researches the web, analyses the evidence, criticises its own work, retries within
hard budgets, and returns a structured, source-backed report with a full execution trace.

This is not "prompt in, answer out". A deterministic orchestrator drives five agents through a
plan, research, analyse, critique, retry/finalise loop, and every claim in the final report
is tied to a collected evidence record.

## How it works


flowchart LR
    O[Objective] --> P[Planner]
    P --> R[Researcher<br/>search tool]
    R --> A[Analyst]
    A --> C{Critic}
    C -- "weak / missing evidence<br/>(bounded retries)" --> R
    C -- pass --> F[Finalizer]
    F --> X[Report + sources + trace]

| Agent | Job | Output contract |
|---|---|---|
| Planner | Decompose the objective into subtasks with acceptance criteria and dependencies | `Plan` |
| Researcher | Propose queries, call search, extract one grounded claim per useful result | `Evidence` records |
| Analyst | Turn evidence into findings, labelled evidence / analysis / conclusion | `Finding` (must cite evidence ids) |
| Critic | Deterministic checks (coverage, missing subtasks, thin sources) plus an LLM pass for contradictions and unsupported claims | `Critique` with a concrete action per issue |
| Finalizer | Build the report; citations are verified against collected evidence | `Report` |

The **orchestrator is plain Python, not an LLM**. Only code decides whether the loop continues.

## Design decisions worth knowing

- **Bounded execution.** Hard limits on subtasks, critic cycles, retries per subtask, tool calls,
  LLM calls and wall-clock time, with a time reserve so a slow run still gets a final report.
  Running out of budget ends the work phase gracefully; it never crashes the run.
- **Graceful degradation.** A provider failure fails one subtask, not the run. The run finishes
  as `completed_with_limitations` and says what is missing. Only "no plan at all" is `failed`.
- **Evidence model.** Evidence (what a source says) is separate from findings (derived) and
  conclusions (generated). A finding that cites an unknown evidence id is rejected.
- **Provider abstraction with quota-aware failover.** Gemini models then Groq for LLMs; Tavily
  then Serper for search. Per-model cooldowns, retry-delay parsing and bounded waits.
- **Structured outputs with one repair attempt.** Malformed JSON gets one repair retry, then the
  step fails. The repair's tokens are counted (`usage.llm_repairs`).
- **No hidden chain-of-thought.** The trace shows events, tool calls, decisions and validation
  results only.

## Security and guardrails

- Search results are **untrusted data**. The extraction prompt says so, and a deterministic
  filter (`agentops/guardrails.py`) drops results containing instruction-like text ("ignore
  previous instructions", role tags, requests for secrets) before the model sees them. The drop
  is recorded in the trace and in the run's limitations. This is a first line of defence, not a
  complete one: the model can only cite results by index, and every claim must resolve to a
  collected evidence record.
- Secrets are `SecretStr` and never logged.
- The public API clamps caller-supplied budgets to fixed ceilings and enforces `DAILY_RUN_CAP`
  (HTTP 429), so one visitor cannot spend the free-tier quota.
- On a deployed UI the backend URL is fixed by `AGENTOPS_API_URL` and the field is locked.
- **Not implemented:** code-execution and file-analysis tools. The `ENABLE_CODE_MODE` flag exists
  but currently does nothing. No arbitrary code execution exists in this system.

## Run locally

python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                                     # then fill in keys and model names
uvicorn api.main:app --port 8000                         # backend
streamlit run ui/app.py                                  # UI, in a second terminal
pytest -q                                                # hermetic: no keys, no network


Set `RUNS_DB_PATH=runs.db` to keep runs across restarts (SQLite). Leave it empty for in-memory.

## Evaluation

`eval/tasks.json` holds a small benchmark: architecture, comparison, decision, risk, numeric and
one **hallucination probe** (an invented ISO standard; the right outcome is an honest, limited
report). Run it against the real agent:

python -m eval.run_eval                       # all tasks, ~13 LLM calls each
python -m eval.run_eval --only fastapi-vs-flask

Output goes to `eval/results/` (JSON plus a readable `latest.md`). Metrics are computed from the
run record, never from the model's own claims: completion, subtask completion, evidence coverage,
citation resolution, supported-finding rate, topic coverage, tool success rate, duplicate
searches, retries, critic recovery, schema repairs, injection drops, latency and tokens.

What the grounding numbers do **not** prove: they check that each citation points at collected
evidence, not that the evidence semantically supports the claim. That needs an LLM judge or human
review and is listed under limitations.

### Results (2026-10-02 run)

7/7 tasks completed (1 `completed_with_limitations`). All grounding metrics at 1.00:
evidence coverage, citation resolution, supported-finding rate, tool success rate.
Topic coverage averaged 0.92. Mean 13.6 LLM calls / 10.9 tool calls per run; p50 latency 39.8s
(max 56.1s on the most complex task). The hallucination probe (`unanswerable-premise`,
an invented ISO standard) correctly produced a limited, honest report instead of
fabricating compliance details — the intended outcome for that task.

| Task | Status | Evid. cov. | Cit. resolved | Supported | Topics | Calls (LLM/tool) | Time (s) |
|---|---|---|---|---|---|---|---|
| enterprise-ai-platform | completed | 1.00 | 1.00 | 1.00 | 0.75 | 12/9 | 35.8 |
| fastapi-vs-flask | completed | 1.00 | 1.00 | 1.00 | 0.75 | 15/13 | 45.2 |
| build-vs-buy-chatbot | completed | 1.00 | 1.00 | 1.00 | 1.00 | 12/9 | 39.8 |
| vector-db-selection | completed_with_limitations | 1.00 | 1.00 | 1.00 | 1.00 | 17/14 | 53.9 |
| agent-security-risks | completed | 1.00 | 1.00 | 1.00 | 1.00 | 9/7 | 26.5 |
| travel-cost-estimate | completed | 1.00 | 1.00 | 1.00 | 1.00 | 18/15 | 56.1 |
| unanswerable-premise | completed | 1.00 | 1.00 | 1.00 | n/a | 12/9 | 33.5 |

Full metrics: `eval/results/latest.md`.

## Deploy for free

1. **Backend on Render** (free web service). Use `render.yaml`; set the API keys as secrets.
2. **UI on Streamlit Community Cloud.** App file `ui/app.py`, `requirements.txt` as provided, and
   set the secret `AGENTOPS_API_URL` to the Render service URL.

## Free-tier limits that affect the demo

- **Render free** spins down after 15 minutes without traffic and can take up to about a minute
  to wake; the UI shows "backend unreachable, retrying". Local files are lost on every restart,
  so SQLite does not persist there: past runs vanish on spin-down. Use `RUNS_DB_PATH` locally or
  on a host with a persistent disk, or add a Postgres repository (`DATABASE_URL` is reserved).
- **Gemini and Groq free tiers** are limited per model per minute and per day; see
  `scripts/quota_probe.py`. A run uses about 12 to 15 LLM calls, so the daily cap protects the
  quota. Check the provider's rate-limit page for current numbers.
- **Search free tiers** (Tavily, Serper) have monthly credit limits; results are cached per run.

## Known limitations

- Grounding is structural, not semantic (see Evaluation).
- The planner sometimes drifts off-topic in search queries; weaker "lite" models make this worse.
- Re-planning is not supported yet: a critic `replan` is recorded as a limitation.
- No Python or file-analysis tool yet.
- Runs are in memory unless `RUNS_DB_PATH` is set.

## Repository layout

agentops/        core: contracts, orchestrator, agents, llm and search routers, runs, guardrails
api/             FastAPI app (thin shell around RunManager)
ui/              Streamlit app
eval/            benchmark tasks, metrics, runner
tests/           hermetic tests using fake LLM and search providers
scripts/         quota probes (manual, never in CI)