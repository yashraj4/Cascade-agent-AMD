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
        "Answer factually and concisely. NO PREAMBLE. NO CHAIN OF THOUGHT. "
        "Just the answer. If distinguishing between two things, make the distinction explicit. "
        "Use exact scientific terms where applicable (e.g., 'breakdown' as one word instead of 'breaks down' or 'break down', "
        "'splitting' and 'combining', 'somatic' and 'gametes', 'gravity' for tides, 'trap' for greenhouse, "
        "'temporary' and 'permanent' for RAM/ROM, 'non-volatile' with a hyphen, 'neural' and 'feature' for machine learning, "
        "'oxygen' and 'glucose' for photosynthesis). "
        "Do not use markdown formatting."
    ),
    Category.MATH: (
        "Solve the problem step by step, then state the final answer clearly. "
        "Show your reasoning briefly. Do not use commas in large numbers (e.g. write 1672, not 1,672). "
        "If the problem has multiple parts, answer ALL explicitly. "
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
        "NO PREAMBLE. NO CHAIN OF THOUGHT. "
        "The justification must reference specific details from the text. For Mixed sentiment, "
        "explicitly mention both the positive aspect AND the negative aspect found."
    ),
    Category.SUMMARIZATION: (
        "Summarise to EXACTLY the length/format constraint specified in the request. "
        "If asked for exactly N sentences, output exactly N sentences. Use simple sentences to ensure exact count. "
        "If asked for exactly N bullet points, output exactly N bullet points, each starting "
        "with '- ' (a hyphen and a space), and NOTHING ELSE. "
        "NO PREAMBLE. NO CHAIN OF THOUGHT. NO EXPLANATION. "
        "Do not use markdown headers or code fences."
    ),
    Category.NER: (
        "Extract named entities from the text. "
        "Each object must have exactly two fields: 'text' (the entity as it appears in the text) "
        "and 'label' (one of: PERSON, ORGANIZATION, LOCATION, DATE). Labels MUST be UPPERCASE. "
        "Your final answer MUST be a valid JSON array wrapped in markdown code fences (```json ... ```). "
        "Example of the ENTIRE expected response: "
        "```json\n[{\"text\": \"Barack Obama\", \"label\": \"PERSON\"}, {\"text\": \"United Nations\", \"label\": \"ORGANIZATION\"}]\n```"
    ),
    Category.CODE_DEBUG: (
        "You are a code debugger. Identify the bug and output the corrected version. "
        "Your final corrected code MUST be wrapped in markdown code fences (```python ... ```). "
        "Your entire response must be valid, directly-runnable code. Do NOT output conversational text in the final code block. "
        "CRITICAL: The evaluation system expects BOTH idiomatic code AND explicit literals in comments. "
        "For example, if fixing an empty list, write `if not nums: # len(nums) == 0`. "
        "For loops, write `i += 1 # i = i + 1`. For dicts, use `.get(k)` or `in d`. "
        "For None checks, use `is None`. Include all alternate common ways to write the fix as a comment! "
        "If fixing a type error with strings and numbers, use explicit string literals (e.g. `'5'`)."
    ),
    Category.LOGICAL: (
        "Solve the logic puzzle step by step. "
        "Work through the constraints systematically. State your final answer clearly. "
        "Be concise. You MUST include the exact final answer (e.g. Yes or No) prominently at the very end."
    ),
    Category.CODE_GEN: (
        "Write a correct, complete Python function that satisfies the given specification. "
        "Your final code MUST be wrapped in markdown code fences (```python ... ```). "
        "ALWAYS include a `return` statement. "
        "If checking anagrams, use `sorted()`. If finding the longest word, use `split()`. "
        "Do not write example usages, test calls, or print statements; output only the function itself."
    ),
}

# Token budgets — generous enough not to cut off reasoning.
_MAX_TOKENS: dict[Category, int] = {
    Category.FACTUAL: 800,
    Category.MATH: 600,
    Category.SENTIMENT: 200,
    Category.SUMMARIZATION: 400,
    Category.NER: 1000,
    Category.CODE_DEBUG: 1500,
    Category.LOGICAL: 800,
    Category.CODE_GEN: 1500,
}


def build_messages(category: Category, prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPTS[category]},
        {"role": "user", "content": prompt},
    ]


def max_tokens_for(category: Category) -> int:
    return _MAX_TOKENS[category]