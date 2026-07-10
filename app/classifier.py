"""
Task category classifier.
"""
from __future__ import annotations

import re
from enum import Enum


class Category(str, Enum):
    FACTUAL = "factual_knowledge"
    MATH = "mathematical_reasoning"
    SENTIMENT = "sentiment_classification"
    SUMMARIZATION = "text_summarisation"
    NER = "named_entity_recognition"
    CODE_DEBUG = "code_debugging"
    LOGICAL = "logical_deductive_reasoning"
    CODE_GEN = "code_generation"


_PATTERNS: list[tuple[Category, re.Pattern]] = [
    (Category.CODE_DEBUG, re.compile(
        r"\b(bug|fix|error|traceback|debug|doesn'?t work|not working|why does this fail)\b",
        re.IGNORECASE)),
    (Category.CODE_GEN, re.compile(
        r"\b(write a function|implement|write code|write a program|create a class|"
        r"def \w+\(|write.*script)\b", re.IGNORECASE)),
    (Category.SUMMARIZATION, re.compile(
        r"\b(summar(y|ise|ize)|condense|tl;?dr|in one sentence|in \d+ words|shorten)\b",
        re.IGNORECASE)),
    (Category.SENTIMENT, re.compile(
        r"\b(sentiment|positive or negative|how does .* feel|classify the (tone|emotion))\b",
        re.IGNORECASE)),
    (Category.NER, re.compile(
        r"\b(extract (the )?(entities|names|people|organi[sz]ations|locations|dates)|"
        r"named entit(y|ies)|identify (all )?(people|organizations|locations|dates))\b",
        re.IGNORECASE)),
    (Category.MATH, re.compile(
        r"\b(\d+\s*[\+\-\*/%]\s*\d+|percent|percentage|how many|calculate|"
        r"if .* costs?|what is \d)\b", re.IGNORECASE)),
    (Category.LOGICAL, re.compile(
        r"\b(puzzle|each of|exactly one|must (be|satisfy)|constraints?|who (is|owns|lives)|"
        r"if and only if|either .* or)\b", re.IGNORECASE)),
]


def classify(prompt: str) -> Category:
    for category, pattern in _PATTERNS:
        if pattern.search(prompt):
            return category
    # Default fallback
    return Category.FACTUAL
