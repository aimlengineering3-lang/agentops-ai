import html as _html

import streamlit as st

from agentops.contracts import RunState
from ui.formatting import (
    badge_html,
    citation_numbers,
    event_html,
    format_duration,
    format_tokens,
    status_kind,
)

_ST_STATUS = {"running": "running", "success": "complete", "warning": "complete", "danger": "error"}


def stat_cards(pairs: list[tuple[str, str]]) -> None:
    cards = "".join(
        f'<div class="ao-stat"><div class="ao-stat-label">{label}</div>'
        f'<div class="ao-stat-value">{value}</div></div>'
        for label, value in pairs
    )
    st.markdown(f'<div class="ao-stats">{cards}</div>', unsafe_allow_html=True)


def render_run(state: RunState) -> None:
    _render_header(state)
    trace_tab, report_tab, evidence_tab = st.tabs(["Live trace", "Report", "Evidence"])
    with trace_tab:
        _render_trace(state)
    with report_tab:
        _render_report(state)
    with evidence_tab:
        _render_evidence(state)


def _render_header(state: RunState) -> None:
    st.markdown(
        f"""
        <div class="ao-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:16px;">
                <div style="font-size:17px; font-weight:600; color:var(--ao-text);">
                    {_html.escape(state.objective)}
                </div>
                {badge_html(state.status)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    done = sum(1 for s in state.plan.subtasks if s.status.value == "done") if state.plan else 0
    total = len(state.plan.subtasks) if state.plan else 0
    stat_cards(
        [
            ("Subtasks", f"{done}/{total}" if total else "—"),
            ("LLM calls", str(state.usage.llm_calls)),
            ("Tool calls", str(state.usage.tool_calls)),
            ("Elapsed", format_duration(state.usage.elapsed_s)),
        ]
    )
    if state.limitations:
        st.warning("**Limitations**\n" + "\n".join(f"- {item}" for item in state.limitations))


def _render_trace(state: RunState) -> None:
    kind = status_kind(state.status)
    st_state = _ST_STATUS[kind]
    label = "Agent is working..." if st_state == "running" else "Run finished"
    with st.status(label, state=st_state, expanded=True):
        if state.plan:
            st.caption("Plan")
            plan_html = "".join(
                f'<div style="margin-bottom:4px; font-size:13px; color:var(--ao-text-muted);">'
                f"<code>{_html.escape(s.id)}</code> "
                f'<span style="color:var(--ao-text);">[{s.status.value}]</span> '
                f"{_html.escape(s.description)}</div>"
                for s in state.plan.subtasks
            )
            st.markdown(plan_html, unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if state.events:
            timeline = (
                '<div class="ao-timeline">'
                + "".join(event_html(e) for e in state.events)
                + "</div>"
            )
            st.markdown(timeline, unsafe_allow_html=True)
        else:
            st.caption("No events yet...")


def _render_report(state: RunState) -> None:
    report = state.report
    if report is None:
        if state.is_terminal:
            st.error("This run finished without producing a report.")
        else:
            st.info("Report not available yet — it's produced once the run finishes.")
        return

    st.markdown("#### Executive summary")
    st.write(report.executive_summary)

    if report.comparison:
        st.markdown("#### Comparison")
        st.write(report.comparison)

    st.markdown("#### Key findings")
    st.caption(
        "[n] marks the source behind each claim (see Sources below). Open "
        "'Evidence' under a finding to read the exact extracted claim and link. "
        "Confidence is the model's own estimate, not a measured value."
    )
    numbers = citation_numbers(state.evidence, report.key_findings)
    for finding in report.key_findings:
        mark = "✅" if finding.supported else "⚠️ unsupported"
        cited = [e for e in finding.evidence_ids if e in state.evidence]
        unresolved = [e for e in finding.evidence_ids if e not in state.evidence]
        refs = "".join(
            f'<span style="color:var(--ao-text-muted); font-size:12px;"> [{n}]</span>'
            for n in sorted({numbers[e] for e in cited})
        )
        st.markdown(
            f'<div class="ao-finding">{mark} <strong>{finding.confidence:.0%} confidence</strong> '
            f"— {_html.escape(finding.statement)}{refs}</div>",
            unsafe_allow_html=True,
        )
        if cited:
            with st.expander(f"Evidence ({len(cited)})"):
                for ev_id in cited:
                    ev = state.evidence[ev_id]
                    label = _html.escape(ev.title)
                    title = (
                        f'<a class="ao-source-link" href="{_html.escape(ev.url)}" '
                        f'target="_blank">{label}</a>'
                        if ev.url
                        else label
                    )
                    st.markdown(
                        f"<strong>[{numbers[ev_id]}] {_html.escape(ev.domain or ev.source_type.value)}"
                        f"</strong> — {title}",
                        unsafe_allow_html=True,
                    )
                    st.caption(ev.extracted_claim)
        if unresolved:
            st.caption(f"⚠️ Could not resolve evidence: {', '.join(unresolved)}")

    for heading, items in (
        ("Trade-offs", report.trade_offs),
        ("Risks", report.risks),
        ("Implementation considerations", report.implementation_considerations),
        ("Limitations", report.limitations),
    ):
        if items:
            st.markdown(f"#### {heading}")
            for item in items:
                st.write(f"- {item}")

    if report.sources:
        st.markdown("#### Sources")
        url_numbers = {
            state.evidence[e].url: n for e, n in numbers.items() if state.evidence[e].url
        }
        for source in report.sources:
            n = url_numbers.get(source.url)
            prefix = f"[{n}] " if n else ""
            st.markdown(
                f"{prefix}"
                f'<a class="ao-source-link" href="{_html.escape(source.url)}" target="_blank">'
                f"{_html.escape(source.title)}</a> "
                f'<span style="color:var(--ao-text-muted); font-size:12px;">— '
                f"{_html.escape(source.domain)}</span>",
                unsafe_allow_html=True,
            )

    st.markdown("#### Run metadata")
    meta = report.metadata
    stat_cards(
        [
            ("Duration", format_duration(meta.duration_s)),
            ("LLM calls", str(meta.llm_calls)),
            (
                "Tokens in/out",
                f"{format_tokens(meta.tokens_in)} / {format_tokens(meta.tokens_out)}",
            ),
            ("Critic cycles", str(meta.critic_cycles)),
        ]
    )
    if meta.llm_calls:
        st.caption(
            f"Tokens are summed over {meta.llm_calls} LLM calls "
            f"(about {meta.tokens_in // meta.llm_calls} in / "
            f"{meta.tokens_out // meta.llm_calls} out per call)"
            + (
                f", plus {state.usage.llm_repairs} schema-repair retries."
                if state.usage.llm_repairs
                else "."
            )
        )
    st.caption(f"Providers used: {', '.join(meta.providers_used) or '—'}")
    st.caption(f"Validation: {'✅ passed' if report.validation_passed else '⚠️ not fully passed'}")


def _render_evidence(state: RunState) -> None:
    if not state.evidence:
        st.info("No evidence collected yet.")
        return

    by_subtask: dict[str, list] = {}
    for item in state.evidence.values():
        by_subtask.setdefault(item.subtask_id, []).append(item)

    for subtask_id, items in by_subtask.items():
        st.markdown(f"**Subtask `{subtask_id}`**")
        for item in items:
            url_line = (
                f'<a class="ao-source-link" href="{_html.escape(item.url)}" target="_blank">'
                f"{_html.escape(item.url)}</a>"
                if item.url
                else ""
            )
            snippet_line = (
                f'<div style="font-size:12px; color:var(--ao-text-muted); margin-top:4px;">'
                f"{_html.escape(item.snippet)}</div>"
                if item.snippet
                else ""
            )
            st.markdown(
                f'<div class="ao-card" style="padding:14px 16px;">'
                f'<div style="font-weight:600; font-size:13.5px;">{_html.escape(item.title)}</div>'
                f'<div style="font-size:12px; margin:2px 0 6px;">{url_line}</div>'
                f'<div style="font-size:13px; color:var(--ao-text);">'
                f"{_html.escape(item.extracted_claim)}</div>{snippet_line}</div>",
                unsafe_allow_html=True,
            )
