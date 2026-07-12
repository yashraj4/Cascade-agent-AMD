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
        r"\b(bug|fix|error|traceback|debug|doesn'?t\s+work|not\s+working|why\s+does\s+this\s+fail|buggy)\b",
        re.IGNORECASE)),
    (Category.CODE_GEN, re.compile(
        r"(\b(write|implement|create|generate|coding|program)\b.*\b(function|code|program|script|class|decorator|generator)\b|def\s+\w+\s*\(|write\s+a\s+python\b)",
        re.IGNORECASE)),
    (Category.SUMMARIZATION, re.compile(
        r"\b(summar(y|ise|ize|ised|ized|ing)|condense|tl;?dr|tldr|shorten|synopsis|in\s+(one|two|three|\d+)\s+(sentence|sentences|bullet|bullets|point|points))\b",
        re.IGNORECASE)),
    (Category.SENTIMENT, re.compile(
        r"\b(sentiment|positive\s+or\s+negative|how\s+does\s+.*feel|classify\s+the\s+(tone|emotion|sentiment)|tone\s+is\s+(positive|negative|neutral|mixed))\b",
        re.IGNORECASE)),
    (Category.NER, re.compile(
        r"(\b(extract|identify|find|label)\b.*\b(entities|names|people|persons|organizations|locations|dates|person|organization|location|date)\b|\bnamed\s+entit(y|ies)\b)",
        re.IGNORECASE)),
    (Category.LOGICAL, re.compile(
        r"(\b(puzzle|riddle|logic|deduce|deduction|truth\s+value|wason|handshakes|shakes\s+hands|vowel|consonant|syllogism|yes\s+or\s+no|yes/no|circular\s+table|opposite|older\s+than|younger\s+than|taller\s+than|shorter\s+than|older|youngest|oldest|tallest|shortest|pills|take\s+one\s+every)\b|"
        r"\b(all|some|no)\b.*\bare\b.*\b(not\b)?|"
        r"\b(only\s+one|exactly\s+one|statements?)\b.*\btrue\b|"
        r"\b(which|who)\b.*\b(box|cup|card|house|lives|owns|finished|first|last|youngest|oldest|opposite)\b|"
        r"\b(brother|sister|father|son|mother|daughter)\b.*\bmy\s+father\b)",
        re.IGNORECASE)),
    (Category.MATH, re.compile(
        r"(\b(percent|percentage|how\s+many|how\s+much|how\s+long|calculate|compute|evaluate|solve\s+for|probability|interest|area|radius|sequence|ratio|fraction|sum|addition|subtraction|multiplication|divided|multiplied|plus|minus|math|arithmetic)\b|%|\b(q1|q2|q3|q4)\b.*\b(sell|restock|remain)\b|\d+\s*[\+\-\*/=]\s*\d+)",
        re.IGNORECASE)),
]


def classify(prompt: str) -> Category:
    for category, pattern in _PATTERNS:
        if pattern.search(prompt):
            return category
    # Default fallback
    return Category.FACTUAL
