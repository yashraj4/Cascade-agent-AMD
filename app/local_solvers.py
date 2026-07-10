"""
Local heuristic-based solvers.
"""
from __future__ import annotations

import ast
import operator
import re

# Sentiment Analysis (VADER)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
except ImportError:
    _vader = None


def solve_sentiment(prompt: str) -> str | None:
    if _vader is None:
        return None
    # Extract target text
    match = re.search(r'[:"](.+)["]?$', prompt.strip())
    text = match.group(1).strip(' "') if match else prompt
    scores = _vader.polarity_scores(text)
    compound = scores["compound"]

    # Escalate ambiguous cases
    if -0.15 < compound < 0.15:
        return None

    label = "positive" if compound > 0 else "negative"
    return (
        f"Sentiment: {label}. Justification: VADER compound score {compound:.2f} "
        f"(pos={scores['pos']:.2f}, neu={scores['neu']:.2f}, neg={scores['neg']:.2f})."
    )


# Named Entity Recognition (spaCy)
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None

_LABEL_MAP = {
    "PERSON": "person",
    "ORG": "organization",
    "GPE": "location",
    "LOC": "location",
    "DATE": "date",
}


def solve_ner(prompt: str) -> str | None:
    if _nlp is None:
        return None
    doc = _nlp(prompt)
    entities = [
        {"text": ent.text, "label": _LABEL_MAP[ent.label_]}
        for ent in doc.ents
        if ent.label_ in _LABEL_MAP
    ]
    if not entities:
        return None
    return str(entities)


# Simple Arithmetic Solver
_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


def solve_simple_arithmetic(prompt: str) -> str | None:
    # Handle only simple, raw arithmetic expressions
    match = re.search(r"([\d\.\s\+\-\*/\(\)%]+)\s*=?\s*\??\s*$", prompt.strip())
    if not match:
        return None
    expr = match.group(1).strip().rstrip("=").strip()
    if not re.fullmatch(r"[\d\.\s\+\-\*/\(\)]+", expr) or not any(c.isdigit() for c in expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception:
        return None
