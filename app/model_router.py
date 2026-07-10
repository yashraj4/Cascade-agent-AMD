"""
Model selection and routing logic.
"""
from __future__ import annotations

from .classifier import Category

# Family model tiers
_FAMILY_TIER = {
    "gemma": 0,    # lightweight, prefer for simple/cheap escalations
    "minimax": 2,  # large reasoning model, prefer for hard categories
    "kimi": 2,     # large reasoning model, prefer for hard categories
}

_HIGH_TIER_CATEGORIES = {Category.LOGICAL, Category.CODE_GEN, Category.CODE_DEBUG}


def _tier_of(model_id: str) -> int:
    lowered = model_id.lower()
    for family, tier in _FAMILY_TIER.items():
        if family in lowered:
            return tier
    return 1


class ModelTiers:
    def __init__(self, allowed_models: list[str]):
        if not allowed_models:
            raise ValueError("ALLOWED_MODELS is empty - cannot route any task.")
        self.ranked = sorted(allowed_models, key=_tier_of)

    @property
    def smallest(self) -> str:
        return self.ranked[0]

    @property
    def largest(self) -> str:
        return self.ranked[-1]

    @property
    def mid(self) -> str:
        return self.ranked[len(self.ranked) // 2]

    def select(self, category: Category) -> str:
        if category in _HIGH_TIER_CATEGORIES:
            return self.largest
        if category in (Category.FACTUAL, Category.SUMMARIZATION):
            return self.mid
        return self.smallest
