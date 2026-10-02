import os

import httpx
import streamlit as st

from agentops.contracts import BudgetLimits
from ui.api_client import AgentOpsClient
from ui.components import render_run, stat_cards
from ui.formatting import badge_html
from ui.theme import inject_theme

st.set_page_config(page_title="AgentOps AI", page_icon="🧠", layout="wide")
inject_theme()

# On a public deployment the backend URL is fixed by the AGENTOPS_API_URL environment
# variable and the sidebar field is locked: a visitor must not be able to point the
# Streamlit server at an arbitrary address.
CONFIGURED_BASE_URL = os.environ.get("AGENTOPS_API_URL", "").strip()
DEFAULT_BASE_URL = CONFIGURED_BASE_URL or "http://localhost:8000"

EXAMPLE_TASKS = [
    "Evaluate three approaches for building an enterprise AI platform. Research current "
    "technology options, compare architecture, trade-offs, security considerations and "
    "deployment complexity, then produce a structured implementation plan.",
    "Compare FastAPI, Django and Flask for a small team building a new SaaS product. "
    "Weigh learning curve, ecosystem maturity, performance and hiring pool.",
    "Research whether a 10-person startup should build or buy a customer support "
    "chatbot in 2026, covering cost, data privacy and time-to-value.",
]


def _init_state() -> None:
    st.session_state.setdefault("base_url", DEFAULT_BASE_URL)
    st.session_state.setdefault("run_id", None)
    st.session_state.setdefault("objective_draft", "")


def _error_detail(exc: httpx.HTTPStatusError) -> str:
    """Prefer the API's own message (e.g. the daily-limit 429) over httpx's generic one."""
    try:
        return str(exc.response.json().get("detail", exc))
    except Exception:
        return str(exc)


def _client() -> AgentOpsClient:
    return AgentOpsClient(st.session_state.base_url)


def _sidebar(client: AgentOpsClient, healthy: bool) -> list:
    with st.sidebar:
        st.markdown(
            '<div class="ao-brand"><div class="ao-brand-mark">🧠</div>'
            '<div class="ao-brand-name">AgentOps AI</div></div>'
            '<div class="ao-brand-tag">Autonomous research &amp; analysis agent</div>',
            unsafe_allow_html=True,
        )

        st.session_state.base_url = st.text_input(
            "Backend URL",
            value=st.session_state.base_url,
            disabled=bool(CONFIGURED_BASE_URL),
        )
        if healthy:
            st.markdown(
                '<span class="ao-badge success"><span class="ao-dot"></span>'
                "Backend connected</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="ao-badge warning"><span class="ao-dot"></span>'
                "Backend unreachable — retrying...</span>",
                unsafe_allow_html=True,
            )

        try:
            runs = client.list_runs(limit=50)
        except Exception:
            runs = []

        st.divider()
        st.markdown("**Session**")
        completed = sum(
            1 for r in runs if r.status.value in ("completed", "completed_with_limitations")
        )
        failed = sum(1 for r in runs if r.status.value in ("failed", "interrupted", "cancelled"))
        stat_cards([("Runs", str(len(runs))), ("Done", str(completed)), ("Failed", str(failed))])

        st.divider()
        if st.button("➕  New run", use_container_width=True):
            st.session_state.run_id = None
            st.rerun()

        st.markdown("**Past runs**")
        if not runs:
            st.caption("No runs yet.")
        for run in runs[:15]:
            st.markdown(badge_html(run.status), unsafe_allow_html=True)
            label = run.objective[:60] + ("…" if len(run.objective) > 60 else "")
            if st.button(label, key=f"run-{run.run_id}", use_container_width=True):
                st.session_state.run_id = run.run_id
                st.rerun()

        return runs


def _render_landing(client: AgentOpsClient, healthy: bool) -> None:
    st.markdown(
        '<div style="max-width:720px; margin:48px auto 20px; text-align:center;">'
        '<div style="font-size:28px; font-weight:700; color:var(--ao-text);">'
        "What should I research?</div>"
        '<div style="color:var(--ao-text-muted); font-size:14px; margin-top:6px;">'
        "Give AgentOps an objective — it plans, researches, critiques itself and "
        "hands back a source-backed report.</div></div>",
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1, 3, 1])
    with center:
        objective = st.text_area(
            "Objective",
            value=st.session_state.objective_draft,
            height=120,
            placeholder="Describe what you want researched and analyzed...",
            label_visibility="collapsed",
        )

        example_cols = st.columns(len(EXAMPLE_TASKS))
        for col, example in zip(example_cols, EXAMPLE_TASKS, strict=True):
            with col:
                if st.button(
                    example[:48] + "…", key=f"ex-{hash(example)}", use_container_width=True
                ):
                    st.session_state.objective_draft = example
                    st.rerun()

        with st.expander("Advanced budget limits"):
            max_subtasks = st.slider("Max subtasks", 2, 8, 6)
            max_critic_cycles = st.slider("Max critic cycles", 1, 3, 2)
            max_wall_clock_s = st.slider("Max wall-clock (s)", 60, 600, 300, step=30)

        start_disabled = not healthy or len(objective.strip()) < 10
        if st.button(
            "Start run", type="primary", disabled=start_disabled, use_container_width=True
        ):
            limits = BudgetLimits(
                max_subtasks=max_subtasks,
                max_critic_cycles=max_critic_cycles,
                max_wall_clock_s=max_wall_clock_s,
            )
            try:
                run = client.create_run(objective.strip(), limits)
                st.session_state.run_id = run.run_id
                st.session_state.objective_draft = ""
                st.rerun()
            except httpx.HTTPStatusError as exc:
                st.error(f"Could not start run: {_error_detail(exc)}")
            except Exception as exc:
                st.error(f"Could not start run: {exc}")


def main() -> None:
    _init_state()
    client = _client()
    healthy = client.health()
    run_id = st.session_state.run_id

    preview = client.get_run(run_id) if run_id else None
    _sidebar(client, healthy)

    if not run_id:
        _render_landing(client, healthy)
        return
    if preview is None:
        st.error("This run no longer exists.")
        return

    # None stops auto-ticking once the run is terminal; the fragment itself forces
    # one last full rerun below so this gets recomputed as None right after finishing.
    run_every = None if preview.is_terminal else "1s"

    @st.fragment(run_every=run_every)
    def _live_view() -> None:
        state = client.get_run(run_id)
        if state is None:
            st.error("This run no longer exists.")
            return
        render_run(state)
        if state.is_terminal and run_every is not None:
            st.rerun()

    _live_view()


if __name__ == "__main__":
    main()
