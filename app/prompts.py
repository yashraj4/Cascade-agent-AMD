"""
Category-specific prompt templates.

Prompting alone cannot GUARANTEE format compliance — models sometimes add
preamble text, wrap code in fences, or vary label capitalization. app/sanitize.py
provides a second, independent post-processing layer that catches these cases
without ever risking corrupting an already-correct answer.
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
        "Your response MUST follow this EXACT format, with nothing before or after it:\n"
        "Sentiment: <label>. Justification: <one sentence>.\n"
        "The justification MUST quote at least one short exact phrase (in double quotes) "
        "directly from the input text — never write a generic statement like 'the overall "
        "tone is positive' with no reference to the actual content. "
        "For Mixed sentiment, you MUST quote one phrase supporting the positive aspect "
        "AND one phrase supporting the negative aspect."
    ),
    Category.SUMMARIZATION: (
        "Summarise to EXACTLY the length/format constraint specified in the request. "
        "If asked for exactly N sentences, output exactly N sentences — no more, no less. "
        "If asked for exactly N bullet points, output exactly N bullet points, each starting "
        "with '- ' (a hyphen and a space). "
        "The summary must cover every major topic/aspect in the source text. "
        "Do not add any preamble, commentary, or explanation about the summary itself. "
        "Do not use markdown headers or code fences."
    ),
    Category.NER: (
        "Extract named entities from the text. "
        "Output ONLY a valid JSON array of objects, and nothing else. Each object must have "
        "exactly two fields: 'text' (the entity as it appears in the text) and 'label' "
        "(one of: PERSON, ORGANIZATION, LOCATION, DATE). Labels MUST be UPPERCASE. "
        "Your entire response must start with '[' and end with ']' — do not write 'Here is "
        "the JSON:' or any other text before or after the array, and do not wrap it in "
        "markdown code fences. "
        "Example of the ENTIRE expected response: "
        "[{\"text\": \"Barack Obama\", \"label\": \"PERSON\"}, {\"text\": \"Paris\", \"label\": \"LOCATION\"}]"
    ),
    Category.CODE_DEBUG: (
        "You are a code debugger. Identify the bug in the provided code and output the "
        "corrected version. Include a brief comment on the fixed line explaining the bug. "
        "Output ONLY the corrected code, and nothing else — no markdown code fences (no ```), "
        "no surrounding explanation text, no restating of the original buggy code. "
        "Your entire response must be valid, directly-runnable code."
    ),
    Category.LOGICAL: (
        "Solve the logic puzzle step by step. "
        "Work through the constraints systematically, eliminating possibilities. "
        "State your final answer clearly at the end. "
        "Be concise — do not narrate every case exhaustively, just show the key deductive steps."
    ),
    Category.CODE_GEN: (
        "Write a correct, complete Python function that satisfies the given specification. "
        "Output ONLY the code, and nothing else — no markdown code fences (no ```), "
        "no explanation, no surrounding text. "
        "Your entire response must be valid, directly-runnable code."
    ),
}

# Token budgets — generous enough not to cut off reasoning.
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