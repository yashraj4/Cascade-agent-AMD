"""
Model tiering / fallback ordering.

ALLOWED_MODELS is published at runtime and must never be hardcoded - but as
of the Track 2 announcement, the actual candidate model IDs are known:

    minimax-m3              - large general reasoning model, always deployed
    kimi-k2p7-code          - large model specialized for code, always deployed
    gemma-4-31b-it          - Gemma, on-demand deployment required
    gemma-4-26b-a4b-it      - Gemma MoE variant (~4B active params), on-demand
    gemma-4-31b-it-nvfp4    - Gemma, quantized (nvfp4), on-demand, likely cheapest Gemma option

IMPORTANT COST NOTE: Gemma models must be manually deployed at
https://app.fireworks.ai/models before they're callable, and billing runs
~$7/hour even while idle. A model can appear in ALLOWED_MODELS and still
404 if your team hasn't deployed it (or undeployed it to save budget). This
router therefore never assumes a specific model is actually reachable - it
builds an ORDERED CANDIDATE LIST per category, and app/fireworks_client.py's
chat_completion_with_fallback() tries them in order until one works.

Preference logic:
  - Code categories try the code-specialized model first (kimi-k2p7-code),
    since a model literally trained for code should out-perform a generic
    one at the same or lower token cost.
  - Logical/deductive reasoning tries the general large reasoning model
    first (minimax-m3).
  - Simple/mid categories try the cheapest Gemma variants first (since
    they're meant to be lighter-weight than minimax/kimi) - but this is
    genuinely optional, not required to pass the accuracy gate, and the
    fallback chain means it's never a hard dependency.
  - Every list ends with whatever ALLOWED_MODELS entries weren't already
    matched, so an unrecognized/new model ID is never silently ignored -
    worst case it's tried last.

This is the piece to sanity-check again if the model list changes.
"""
from __future__ import annotations

from .classifier import Category

# Ordered by preference, most-preferred first. Matched case-insensitively as
# substrings against whatever's actually in ALLOWED_MODELS at runtime.
_CATEGORY_PREFERENCE: dict[Category, list[str]] = {
    # CODE tasks: Kimi (code-specialist) is the strongest here — keep it first.
    # Gemma is tried as a last resort only.
    Category.CODE_DEBUG: ["kimi", "minimax", "gemma-4-31b-it-nvfp4", "gemma-4-31b-it", "gemma-4-26b-a4b-it"],
    Category.CODE_GEN:   ["kimi", "minimax", "gemma-4-31b-it-nvfp4", "gemma-4-31b-it", "gemma-4-26b-a4b-it"],
    # NER: MiniMax is strongest for structured JSON compliance; Gemma as fallback.
    Category.NER:        ["minimax", "kimi", "gemma-4-31b-it-nvfp4", "gemma-4-26b-a4b-it", "gemma-4-31b-it"],
    # LOGICAL: Gemma 31B has a native thinking mode — try it first at 0 tokens.
    # MiniMax/Kimi are premium fallbacks only if Gemma fails.
    Category.LOGICAL:    ["gemma-4-31b-it", "gemma-4-31b-it-nvfp4", "gemma-4-26b-a4b-it", "minimax", "kimi"],
    # Simple text tasks: always try Gemma first (0-token cost), escalate only on failure.
    Category.FACTUAL:      ["gemma-4-31b-it-nvfp4", "gemma-4-26b-a4b-it", "gemma-4-31b-it", "minimax", "kimi"],
    Category.MATH:         ["gemma-4-26b-a4b-it", "gemma-4-31b-it-nvfp4", "gemma-4-31b-it", "minimax", "kimi"],
    Category.SUMMARIZATION:["gemma-4-31b-it-nvfp4", "gemma-4-26b-a4b-it", "gemma-4-31b-it", "minimax", "kimi"],
    Category.SENTIMENT:    ["gemma-4-26b-a4b-it", "gemma-4-31b-it-nvfp4", "gemma-4-31b-it", "minimax", "kimi"],
}


class ModelTiers:
    def __init__(self, allowed_models: list[str]):
        if not allowed_models:
            raise ValueError("ALLOWED_MODELS is empty - cannot route any task.")
        self.allowed_models = allowed_models

    def candidates_for(self, category: Category) -> list[str]:
        """
        Returns an ordered list of models to try for this category, filtered
        to only what's actually present in ALLOWED_MODELS, with any
        unmatched/unknown models appended at the end as a last resort.
        """
        preference = _CATEGORY_PREFERENCE.get(category, [])
        ordered: list[str] = []

        for term in preference:
            for model in self.allowed_models:
                if term.lower() in model.lower() and model not in ordered:
                    ordered.append(model)

        # Anything in ALLOWED_MODELS not matched by any preference term - try
        # it last rather than never trying it at all.
        for model in self.allowed_models:
            if model not in ordered:
                ordered.append(model)

        return ordered
