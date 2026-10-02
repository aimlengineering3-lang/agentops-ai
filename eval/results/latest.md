# AgentOps evaluation: 2026-10-02_124437

7 task(s). Grounding metrics are structural: a cited id exists in
the collected evidence. They are not a semantic entailment check.

| Task | Status | Subtasks | Evid. cov. | Cit. resolved | Supported | Topics | Tools ok | Retries | Calls (LLM/tool) | Tokens in/out | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| enterprise-ai-platform | completed | 1.00 | 1.00 | 1.00 | 1.00 | 0.75 | 1.00 | 0 | 12/9 | 11702/4248 | 35.8 |
| fastapi-vs-flask | completed | 1.00 | 1.00 | 1.00 | 1.00 | 0.75 | 1.00 | 0 | 15/13 | 15614/4689 | 45.2 |
| build-vs-buy-chatbot | completed | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 12/9 | 13723/4721 | 39.8 |
| vector-db-selection | completed_with_limitations | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1 | 17/14 | 23533/5935 | 53.9 |
| agent-security-risks | completed | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 9/7 | 8278/2741 | 26.5 |
| travel-cost-estimate | completed | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 18/15 | 19389/5565 | 56.1 |
| unanswerable-premise | completed | 1.00 | 1.00 | 1.00 | 1.00 | n/a | 1.00 | 0 | 12/9 | 11625/3681 | 33.5 |

## Summary

- **completion_rate**: 1.0
- **clean_completion_rate**: 0.857
- **validation_pass_rate**: 1.0
- **mean_subtask_completion**: 1.0
- **mean_evidence_coverage**: 1.0
- **mean_citation_resolution**: 1.0
- **mean_supported_rate**: 1.0
- **mean_evidence_backed_rate**: 1.0
- **mean_topic_coverage**: 0.917
- **mean_tool_success_rate**: 1.0
- **total_duplicate_searches**: 0
- **runs_with_retries**: 1
- **critic_recovery**: 0/1
- **total_llm_repairs**: 3
- **total_injection_drops**: 0
- **honest_degradation**: 0/1
- **mean_llm_calls**: 13.571
- **mean_tool_calls**: 10.857
- **mean_tokens_in**: 14837.714
- **mean_tokens_out**: 4511.429
- **latency_p50_s**: 39.8
- **latency_max_s**: 56.1
