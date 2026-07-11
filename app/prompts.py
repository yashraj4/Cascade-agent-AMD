"""
Category-specific prompt templates.

Each template adds a short system instruction pushing the model toward a
concise, directly-scoreable answer — the goal is fewer output tokens without
losing the information the LLM-Judge needs to grade intent correctly.
"""
from __future__ import annotations

from .classifier import Category

_SYSTEM_PROMPTS: dict[Category, str] = {
    Category.FACTUAL: (
        "Answer factually and concisely. No preamble, no filler phrases like "
        "'Great question' — just the answer. If the question asks to distinguish "
        "between two things, make the distinction explicit rather than describing "
        "them separately."
    ),
    Category.MATH: (
        "Solve step by step internally, but output ONLY the final numeric answer(s) "
        "and a brief justification. If the prompt asks multiple sub-questions "
        "(e.g. both a quantity and a cost), answer ALL of them explicitly — do not "
        "only answer the first part."
    ),
    Category.SENTIMENT: (
        "Classify the sentiment as Positive, Negative, Neutral, or Mixed — use "
        "whichever label best fits, including Mixed or Neutral when the text "
        "contains both positive and negative aspects. Then give a one-sentence "
        "justification that explicitly references the SPECIFIC positive and "
        "negative details from the text if both are present — a generic tone "
        "statement is not sufficient when the content is mixed. "
        "Format: 'Sentiment: <label>. Justification: <reason>.'"
    ),
    Category.SUMMARIZATION: (
        "Summarise to EXACTLY the length/format constraint requested (e.g. if asked "
        "for exactly two sentences, output exactly two sentences; if asked for "
        "exactly three bullet points each under a word limit, follow that precisely). "
        "The summary must cover every major side/aspect mentioned in the source "
        "(e.g. both benefits and drawbacks, if both are present) — do not omit one "
        "side. Do not add commentary about the summary itself."
    ),
    Category.NER: (
        "Extract named entities as a JSON list of objects with 'text' and 'label' "
        "fields. Labels MUST be exactly one of these uppercase strings: PERSON, "
        "ORGANIZATION, LOCATION, DATE. Output ONLY valid JSON, no other text, no "
        "markdown code fences."
    ),
    Category.CODE_DEBUG: (
        "Identify the bug and provide the corrected code only, with a one-line "
        "comment explaining the fix. Do not restate the original buggy code. "
        "Output raw code only — no markdown code fences (no ```), no surrounding text."
    ),
    Category.LOGICAL: (
        "Solve the constraint puzzle. State the final answer clearly, with a brief "
        "justification that all constraints are satisfied. Skip exhaustive case-by-case narration."
    ),
    Category.CODE_GEN: (
        "Write a correct, well-structured function per the spec. Output ONLY the code, "
        "no explanation unless explicitly requested. Output raw code only — no markdown "
        "code fences (no ```), no surrounding text."
    ),
}

# Conservative per-category output caps to bound token spend even when the
# model wants to ramble. Tune these once real accuracy/token tradeoffs are visible.
_MAX_TOKENS: dict[Category, int] = {
    Category.FACTUAL: 200,
    Category.MATH: 150,
    Category.SENTIMENT: 120,
    Category.SUMMARIZATION: 200,
    Category.NER: 250,
    Category.CODE_DEBUG: 350,
    Category.LOGICAL: 300,
    Category.CODE_GEN: 400,
}


def build_messages(category: Category, prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPTS[category]},
        {"role": "user", "content": prompt},
    ]


def max_tokens_for(category: Category) -> int:
    return _MAX_TOKENS[category]