# AgentOps evaluation: 2026-10-02_104206

2 task(s). Grounding metrics are structural: a cited id exists in
the collected evidence. They are not a semantic entailment check.

| Task | Status | Subtasks | Evid. cov. | Cit. resolved | Supported | Topics | Tools ok | Retries | Calls (LLM/tool) | Tokens in/out | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| enterprise-ai-platform | completed_with_limitations | 0.33 | 0.33 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 7/4 | 4475/3051 | 278.2 |
| fastapi-vs-flask | completed_with_limitations | 0.33 | 0.33 | n/a | n/a | 0.00 | 0.33 | 1 | 8/3 | 652/342 | 76.1 |

## Summary

- **completion_rate**: 1.0
- **clean_completion_rate**: 0.0
- **validation_pass_rate**: 1.0
- **mean_subtask_completion**: 0.333
- **mean_evidence_coverage**: 0.333
- **mean_citation_resolution**: 1.0
- **mean_supported_rate**: 1.0
- **mean_evidence_backed_rate**: 1.0
- **mean_topic_coverage**: 0.5
- **mean_tool_success_rate**: 0.667
- **total_duplicate_searches**: 0
- **runs_with_retries**: 1
- **critic_recovery**: 0/1
- **total_llm_repairs**: 0
- **total_injection_drops**: 0
- **honest_degradation**: n/a
- **mean_llm_calls**: 7.5
- **mean_tool_calls**: 3.5
- **mean_tokens_in**: 2563.5
- **mean_tokens_out**: 1696.5
- **latency_p50_s**: 177.1
- **latency_max_s**: 278.2
