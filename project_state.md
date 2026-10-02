# PROJECT_STATE.md (AgentOps AI)

Last updated: 2026-10-02 (end of session 1, later update)

## Current stage
Core system written AND applied locally. Verified on the owner's machine: `pytest -q` gives
221 passed, 1 warning (update zip applied; eval/ exists and runs).
First eval attempt (--limit 2) ran end to end but its numbers are INVALID: Gemini and Groq were
unreachable from the owner's network (DNS getaddrinfo failures, timeouts, 100% ping loss to
google.com), so runs degraded to completed_with_limitations (subtasks 0.33, 7-8 LLM calls,
278s and 76s). Do not use or commit those numbers.
Positive signal: with every LLM provider down, the system did not crash; budgets held (278s of
a 300s limit), failures were traced, runs finished with limitations.
scripts/quota_probe.py result: Tavily OK, Serper OK, Gemini FAILED (timeout), Groq FAILED (timeout).
Cause is network or ISP, not code or keys (not fully confirmed). Owner currently has no network.
Not yet done: valid eval run, deployment, `ruff check` result not confirmed.

## Completed
- Pydantic contracts, deterministic Orchestrator (state machine, BudgetGuard, EventBus)
- 5 agents: LLMPlanner, LLMResearcher, LLMAnalyst, HybridCritic, LLMFinalizer
- LLMRouter (Gemini models -> Groq), SearchRouter (Tavily -> Serper) + search cache
- FastAPI app + RunManager; InMemoryRunRepository and SQLiteRunRepository (set RUNS_DB_PATH)
- Streamlit UI: live trace, report, evidence tabs; numbered citations [n] with a per-finding
  evidence expander; compact token display (17.5K); backend URL locked when AGENTOPS_API_URL is set
- Token accounting: schema-repair attempts are summed into usage and counted in
  usage.llm_repairs; a failed repair still records tokens (MalformedOutputError carries
  tokens_in / tokens_out / attempts). llm_calls stays a count of LOGICAL calls on purpose.
- Guardrail: agentops/guardrails.py drops injection-like search results before the extraction
  LLM sees them (recorded as a VALIDATION event plus a run limitation)
- API: DAILY_RUN_CAP enforced (HTTP 429), budget ceilings via clamp_limits, objective length
  validated (422)
- Startup recovery: orphaned PENDING/RUNNING runs become INTERRUPTED (SQLite repo)
- Evaluation: eval/tasks.json (7 tasks incl. one hallucination probe), eval/metrics.py,
  eval/run_eval.py (writes eval/results/*.json and latest.md). Metrics are computed from the
  RunState, not from model claims.
- Deploy files: render.yaml (backend), requirements.txt (`-e .`, for Streamlit Cloud),
  .github/workflows/ci.yml, README.md

## Not working / known issues
- Planner sometimes drifts off-topic in search queries; lite models are weaker
- Groq free tier: 8000 tokens/min is the fallback bottleneck
- Re-planning unsupported: critic REPLAN is recorded as a limitation
- No Python/data-analysis tool and no file-analysis tool (both were in the original spec).
  ENABLE_CODE_MODE exists in config but is a no-op. No arbitrary code execution exists.
- Grounding metrics are structural (cited id exists in collected evidence), not semantic
  entailment
- Render free tier: sleeps after 15 min idle, ephemeral disk, so SQLite does not persist
  there. Leave RUNS_DB_PATH empty on Render.
- No Postgres repository (DATABASE_URL is reserved only)
- `requirements.txt` with `-e .` is unverified on Streamlit Community Cloud

## Setup notes
- Eval idea not built: flag/exclude runs hit by provider outages from the eval summary
  (currently a network-degraded run is scored like any other)
- Windows PowerShell here does not support `&&`; run commands on separate lines
- Free-tier quotas are per model: check aistudio.google.com/rate-limit
- Search: Tavily and Serper (2,500/month). See .env.example for all settings.

## Important decisions
- Orchestrator is plain code, not an LLM; only code decides loop/stop
- Raw REST providers (httpx), no vendor SDKs
- BudgetExceededError stops the whole run; ProviderError / MalformedOutputError fails one subtask
- Provider failure -> subtask FAILED -> run completed_with_limitations; only "no plan" is FAILED
- Tests must never read the real .env or launch real runs (hermetic, fake LLM and search)
- Never commit or zip .env / .venv

## Next tasks (in order)
0. Restore access to Gemini/Groq: try phone hotspot, DNS 8.8.8.8/1.1.1.1, disable VPN/proxy,
   check firewall/antivirus. Verify with `python scripts/quota_probe.py` (all four must say OK).
   Diagnostics: Test-NetConnection generativelanguage.googleapis.com -Port 443 (same for
   api.groq.com). If the home network stays blocked, run eval from a deployed host or CI.
0b. Then `python -m eval.run_eval --only fastapi-vs-flask` (expect ~12-15 LLM calls, status
   completed), then the full set; only then write real numbers into README.
1. `ruff check --fix .` and `ruff format .`, confirm clean, then commit
2. `python -m eval.run_eval --limit 2`, then the full set; commit eval/results/latest.md and
   put the real numbers in README (none are written there yet)
3. Deploy backend on Render, UI on Streamlit Cloud (set AGENTOPS_API_URL), smoke-test one run
4. Open question to the owner: build a safe calculator tool (arithmetic only, no exec/eval) so
   numeric tasks like the travel budget are computed, or leave as roadmap. Full Python sandbox
   is NOT recommended on free hosting. Owner has not answered yet.
5. Optional: Postgres repository (Supabase) if past runs must survive Render restarts