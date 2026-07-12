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
        "them separately. Do not use markdown formatting."
    ),
    Category.MATH: (
        "Solve the problem step by step, then state the final answer clearly. "
        "Show your reasoning briefly so the answer is verifiable. "
        "If the problem has multiple parts or sub-questions, answer ALL of them explicitly. "
        "Do not use markdown formatting or code fences."
    ),
    Category.SENTIMENT: (
        "You are a sentiment classifier. Classify the sentiment of the given text as "
        "EXACTLY one of: Positive, Negative, Neutral, or Mixed.\n"
        "Rules:\n"
        "- Use 'Mixed' when the text contains BOTH clearly positive AND clearly negative aspects.\n"
        "- Use 'Neutral' when the text is purely factual/descriptive with no strong emotion.\n"
        "- Use 'Positive' when the overall tone is clearly positive.\n"
        "- Use 'Negative' when the overall tone is clearly negative.\n"
        "Your response MUST follow this EXACT format (no deviations):\n"
        "Sentiment: <label>. Justification: <one sentence referencing specific details from the text>.\n"
        "For Mixed sentiment, the justification MUST explicitly mention both the positive aspect "
        "AND the negative aspect found in the text. Do not write a generic tone statement."
    ),
    Category.SUMMARIZATION: (
        "Summarise to EXACTLY the length/format constraint specified in the request. "
        "If asked for exactly N sentences, output exactly N sentences — no more, no less. "
        "If asked for exactly N bullet points, output exactly N bullet points starting with '- '. "
        "The summary must cover every major topic/aspect in the source text. "
        "Do not add any preamble, commentary, or explanation about the summary itself. "
        "Do not use markdown headers or code fences."
    ),
    Category.NER: (
        "Extract named entities from the text. "
        "Output ONLY a valid JSON array of objects. Each object must have exactly two fields: "
        "'text' (the entity as it appears in the text) and 'label' (one of: PERSON, ORGANIZATION, LOCATION, DATE). "
        "Labels MUST be UPPERCASE. "
        "Output ONLY the JSON array — no explanation, no markdown, no code fences, no trailing text. "
        "Example: [{\"text\": \"Barack Obama\", \"label\": \"PERSON\"}, {\"text\": \"Paris\", \"label\": \"LOCATION\"}]"
    ),
    Category.CODE_DEBUG: (
        "You are a code debugger. Identify the bug in the provided code and output the corrected version. "
        "Include a brief comment on the line that was fixed explaining what the bug was. "
        "Output ONLY the corrected code — no markdown code fences (no ```), no surrounding explanation text, "
        "no restating of the original buggy code."
    ),
    Category.LOGICAL: (
        "Solve the logic puzzle step by step. "
        "Work through the constraints systematically, eliminating possibilities. "
        "State your final answer clearly at the end. "
        "Be concise — do not narrate every case exhaustively, just show the key deductive steps."
    ),
    Category.CODE_GEN: (
        "Write a correct, complete Python function that satisfies the given specification. "
        "Output ONLY the code — no markdown code fences (no ```), no explanation, no surrounding text. "
        "The code must be directly runnable."
    ),
}

# Token limits — generous enough to not cut off reasoning, but bounded.
_MAX_TOKENS: dict[Category, int] = {
    Category.FACTUAL: 300,
    Category.MATH: 400,
    Category.SENTIMENT: 150,
    Category.SUMMARIZATION: 300,
    Category.NER: 400,
    Category.CODE_DEBUG: 500,
    Category.LOGICAL: 500,
    Category.CODE_GEN: 600,
}


def build_messages(category: Category, prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPTS[category]},
        {"role": "user", "content": prompt},
    ]


def max_tokens_for(category: Category) -> int:
    return _MAX_TOKENS[category]