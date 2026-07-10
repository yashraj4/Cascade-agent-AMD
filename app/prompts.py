"""
Prompt templates for task categories.
"""
from __future__ import annotations

from .classifier import Category

_SYSTEM_PROMPTS: dict[Category, str] = {
    Category.FACTUAL: (
        "Answer factually and concisely. No preamble, no filler phrases like "
        "'Great question' — just the answer."
    ),
    Category.MATH: (
        "Solve step by step internally, but output ONLY the final numeric answer "
        "and a one-line justification. Do not show full derivations."
    ),
    Category.SENTIMENT: (
        "Classify the sentiment as positive, negative, or neutral, and give a "
        "one-sentence justification. Format: 'Sentiment: <label>. Justification: <reason>.'"
    ),
    Category.SUMMARIZATION: (
        "Summarise to exactly the length/format constraint requested. Do not add "
        "commentary about the summary itself."
    ),
    Category.NER: (
        "Extract named entities as a JSON list of objects with 'text' and 'label' "
        "fields (label one of: person, organization, location, date). Output ONLY the JSON."
    ),
    Category.CODE_DEBUG: (
        "Identify the bug and provide the corrected code only, with a one-line "
        "comment explaining the fix. Do not restate the original buggy code."
    ),
    Category.LOGICAL: (
        "Solve the constraint puzzle. State the final answer clearly, with a brief "
        "justification that all constraints are satisfied. Skip exhaustive case-by-case narration."
    ),
    Category.CODE_GEN: (
        "Write a correct, well-structured function per the spec. Output ONLY the code, "
        "no explanation unless explicitly requested."
    ),
}

# Output token caps per category
_MAX_TOKENS: dict[Category, int] = {
    Category.FACTUAL: 200,
    Category.MATH: 120,
    Category.SENTIMENT: 80,
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
