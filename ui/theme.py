import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --ao-bg: #0B0D12;
    --ao-surface: #12151C;
    --ao-surface-2: #161922;
    --ao-border: rgba(255,255,255,0.08);
    --ao-border-strong: rgba(255,255,255,0.16);
    --ao-text: #E6E8EE;
    --ao-text-muted: #8B92A8;
    --ao-accent: #7C5CFF;
    --ao-accent-soft: rgba(124,92,255,0.15);
    --ao-cyan: #22D3EE;
    --ao-success: #22C55E;
    --ao-success-soft: rgba(34,197,94,0.15);
    --ao-warning: #F59E0B;
    --ao-warning-soft: rgba(245,158,11,0.15);
    --ao-danger: #EF4444;
    --ao-danger-soft: rgba(239,68,68,0.15);
}

html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

[data-testid="stAppViewContainer"] { background: var(--ao-bg); }
[data-testid="stSidebar"] { background: var(--ao-surface); border-right: 1px solid var(--ao-border); }

div.stButton > button {
    background: var(--ao-surface-2);
    color: var(--ao-text);
    border: 1px solid var(--ao-border);
    border-radius: 10px;
    font-weight: 500;
    transition: all 0.15s ease;
}
div.stButton > button:hover { border-color: var(--ao-accent); color: var(--ao-accent); }
div.stButton > button[kind="primary"] { background: var(--ao-accent); border: none; color: white; }
div.stButton > button[kind="primary"]:hover { background: #6A4BF0; color: white; }

[data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input {
    background: var(--ao-surface-2) !important;
    border: 1px solid var(--ao-border) !important;
    border-radius: 10px !important;
    color: var(--ao-text) !important;
}

.ao-brand { display: flex; align-items: center; gap: 10px; margin-bottom: 2px; }
.ao-brand-mark {
    width: 34px; height: 34px; border-radius: 9px;
    background: linear-gradient(135deg, var(--ao-accent), var(--ao-cyan));
    display: flex; align-items: center; justify-content: center; font-size: 18px;
}
.ao-brand-name { font-size: 18px; font-weight: 700; color: var(--ao-text); }
.ao-brand-tag { color: var(--ao-text-muted); font-size: 12.5px; margin: 0 0 18px 44px; }

.ao-card {
    background: var(--ao-surface);
    border: 1px solid var(--ao-border);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 16px;
}

.ao-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
.ao-stat { background: var(--ao-surface); border: 1px solid var(--ao-border); border-radius: 12px; padding: 14px 16px; }
.ao-stat-label { color: var(--ao-text-muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
.ao-stat-value { color: var(--ao-text); font-size: 22px; font-weight: 700; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }

.ao-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 999px; font-size: 12.5px; font-weight: 600; }
.ao-dot { width: 7px; height: 7px; border-radius: 50%; }
.ao-badge.running { background: var(--ao-accent-soft); color: #B7A6FF; }
.ao-badge.running .ao-dot { background: var(--ao-accent); animation: ao-pulse 1.4s ease-in-out infinite; }
.ao-badge.success { background: var(--ao-success-soft); color: #86EFAC; }
.ao-badge.success .ao-dot { background: var(--ao-success); }
.ao-badge.warning { background: var(--ao-warning-soft); color: #FCD34D; }
.ao-badge.warning .ao-dot { background: var(--ao-warning); }
.ao-badge.danger { background: var(--ao-danger-soft); color: #FCA5A5; }
.ao-badge.danger .ao-dot { background: var(--ao-danger); }
@keyframes ao-pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

.ao-timeline { position: relative; padding-left: 26px; }
.ao-timeline::before { content: ""; position: absolute; left: 9px; top: 4px; bottom: 4px; width: 1.5px; background: var(--ao-border-strong); }
.ao-event { position: relative; padding: 9px 0; }
.ao-event::before {
    content: ""; position: absolute; left: -26px; top: 15px;
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--ao-surface-2); border: 2px solid var(--ao-border-strong);
}
.ao-event.ok::before { border-color: var(--ao-success); }
.ao-event.fail::before { border-color: var(--ao-danger); background: var(--ao-danger-soft); }
.ao-event-head { display: flex; align-items: baseline; gap: 8px; }
.ao-event-agent { font-weight: 600; font-size: 13.5px; color: var(--ao-text); }
.ao-event-dur { color: var(--ao-text-muted); font-size: 11.5px; font-family: 'JetBrains Mono', monospace; }
.ao-event-summary { color: var(--ao-text-muted); font-size: 13.5px; margin-top: 2px; }

.ao-finding { border-left: 3px solid var(--ao-accent); padding: 8px 14px; margin-bottom: 10px; background: var(--ao-surface-2); border-radius: 0 10px 10px 0; }
.ao-source-link { color: var(--ao-cyan); text-decoration: none; }
.ao-source-link:hover { text-decoration: underline; }
</style>
"""


def inject_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
