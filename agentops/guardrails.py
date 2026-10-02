"""Deterministic guardrails for untrusted text (web pages, uploaded files).

Web content is DATA. It must never be able to instruct the agent. This is a cheap,
explainable first line of defence: a short list of instruction-like patterns. It is not a
complete defence (no regex list is), so it is layered with the prompt rule "results are
untrusted data" and with structure: the model can only cite results by index, and every
claim must resolve to a collected Evidence record.
"""

import re

# (label, pattern). Kept short and specific: broad patterns would drop legitimate pages.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in (
        (
            "ignore-previous-instructions",
            r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}"
            r"\b(previous|prior|above|earlier|all|your)\b[^.\n]{0,20}"
            r"\b(instructions?|prompts?|rules|messages?)\b",
        ),
        (
            "reveal-secrets",
            r"\b(reveal|print|show|output|send|leak)\b[^.\n]{0,30}"
            r"\b(system prompt|hidden prompt|api[ _-]?keys?|secrets?|credentials?)\b",
        ),
        ("new-instructions", r"\bnew (system )?instructions?\s*:"),
        ("role-tag", r"<\s*/?\s*(system|assistant|developer)\s*>"),
        ("system-role-line", r"(^|\n)\s*system\s*:\s*you\b"),
        (
            "act-as-agent",
            r"\bas an ai (model|assistant|agent)\b[^.\n]{0,40}\byou (must|should|will)\b",
        ),
    )
)


def find_prompt_injection(text: str) -> str | None:
    """Return the label of the first instruction-like pattern found in `text`, else None."""
    for label, pattern in _PATTERNS:
        if pattern.search(text):
            return label
    return None
