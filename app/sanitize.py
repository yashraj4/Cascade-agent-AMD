"""
Post-processing sanitizer.

A second, independent safety layer on top of prompting. No system prompt can
GUARANTEE format compliance - models sometimes add preamble ("Here is the JSON:"),
wrap code in fences despite instructions, or vary label capitalization.

Hard rule: IF IN DOUBT, RETURN THE ORIGINAL TEXT UNCHANGED.
A sanitizer that "fixes" something it misunderstood risks turning a correct
answer into a wrong one. Every function only acts when it recognises a clear,
specific, fixable pattern.
"""
from __future__ import annotations

import json
import re

from .classifier import Category

_VALID_NER_LABELS = {"PERSON", "ORGANIZATION", "LOCATION", "DATE"}

# Maps common label variants/typos to the canonical uppercase spelling.
_NER_LABEL_ALIASES: dict[str, str] = {
    "PERSON": "PERSON", "PER": "PERSON",
    "ORGANIZATION": "ORGANIZATION", "ORG": "ORGANIZATION", "ORGANISATION": "ORGANIZATION",
    "LOCATION": "LOCATION", "LOC": "LOCATION", "GPE": "LOCATION", "PLACE": "LOCATION",
    "DATE": "DATE",
}

_SENTIMENT_LABEL_CANON: dict[str, str] = {
    "positive": "Positive", "negative": "Negative",
    "neutral": "Neutral", "mixed": "Mixed",
}


def sanitize_answer(category: Category, raw: str) -> str:
    """Dispatches to the right sanitizer. Always safe — returns raw unchanged
    for categories with no sanitizer, or on any failure."""
    if not raw:
        return raw
    try:
        if category in (Category.CODE_DEBUG, Category.CODE_GEN):
            return _strip_code_fences(raw)
        if category == Category.NER:
            return _sanitize_ner(raw)
        if category == Category.SENTIMENT:
            return _sanitize_sentiment(raw)
        if category == Category.SUMMARIZATION:
            return _sanitize_bullets(raw)
    except Exception:
        # Sanitizing must never be the reason a task fails.
        pass
    return raw


# ---------------------------------------------------------------------------
# Code: strip markdown fences
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    # Match ```optional-lang\n...\n``` or ```optional-lang\n...```
    for pattern in (
        r"^```[a-zA-Z0-9_+\-]*\n(.*)\n```$",
        r"^```[a-zA-Z0-9_+\-]*\n(.*)```$",
        r"^```\n(.*)\n```$",
    ):
        m = re.match(pattern, stripped, re.DOTALL)
        if m:
            return m.group(1).strip()
    return text


# ---------------------------------------------------------------------------
# NER: extract + normalize JSON array
# ---------------------------------------------------------------------------

def _sanitize_ner(text: str) -> str:
    # Try parsing as-is first.
    parsed = _try_parse_json_array(text.strip())

    if parsed is None:
        # Model added preamble/postamble — extract [ ... ] span.
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            parsed = _try_parse_json_array(text[start: end + 1])

    if parsed is None:
        return text  # can't confidently extract valid JSON

    normalized = []
    for entity in parsed:
        if not isinstance(entity, dict) or "text" not in entity or "label" not in entity:
            return text  # unexpected structure — don't guess
        raw_label = str(entity["label"]).strip().upper()
        canonical = _NER_LABEL_ALIASES.get(raw_label, raw_label)
        normalized.append({"text": entity["text"], "label": canonical})

    return json.dumps(normalized)


def _try_parse_json_array(text: str):
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sentiment: normalise label casing and strip preamble
# ---------------------------------------------------------------------------

def _sanitize_sentiment(text: str) -> str:
    # Use re.search (not match) so preamble before "Sentiment:" doesn't block.
    m = re.search(
        r"Sentiment:\s*([A-Za-z]+)\.\s*Justification:\s*(.+?)\.?\s*$",
        text.strip(), re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return text  # pattern not found — leave untouched

    raw_label, justification = m.group(1), m.group(2).strip()
    canonical = _SENTIMENT_LABEL_CANON.get(raw_label.lower())
    if canonical is None:
        return text  # unrecognised label word — don't guess

    # Ensure the justification ends with a period.
    if justification and not justification.endswith("."):
        justification += "."
    return f"Sentiment: {canonical}. Justification: {justification}"


# ---------------------------------------------------------------------------
# Summarization: normalise bullet markers to '- '
# ---------------------------------------------------------------------------

def _sanitize_bullets(text: str) -> str:
    lines = text.strip().split("\n")
    bullet_pat = re.compile(r"^\s*(?:[-*•]|\d+[.)]) +")
    non_empty = [l for l in lines if l.strip()]
    bullet_lines = [l for l in non_empty if bullet_pat.match(l)]

    # Only normalise if ALL non-empty lines look like bullet markers —
    # otherwise this might be a sentence-based summary.
    if len(bullet_lines) < 2 or len(bullet_lines) != len(non_empty):
        return text

    return "\n".join(bullet_pat.sub("- ", line) for line in lines)
