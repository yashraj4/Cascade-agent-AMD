"""
Fireworks client.

Reads FIREWORKS_API_KEY / FIREWORKS_BASE_URL / ALLOWED_MODELS purely from the
environment at call time - never hardcoded, never bundled in a .env in the
image, per the harness contract. All calls MUST go through FIREWORKS_BASE_URL
or they won't be recorded (and the submission scores zero tokens).

IMPORTANT: Gemma models on this track are on-demand deployments. A model can
be listed in ALLOWED_MODELS and still return 404 if it hasn't actually been
deployed (or was undeployed to save cost - Gemma 4 E4B bills ~$7/hour even
idle). chat_completion_with_fallback tries an ordered list of candidates and
moves to the next one on failure, so a single undeployed model never fails
the whole task.
"""
from __future__ import annotations

import os

import requests


class AllCandidatesFailedError(Exception):
    """Raised when every candidate model in the fallback chain failed."""


class FireworksClient:
    def __init__(self):
        self.api_key = os.environ["FIREWORKS_API_KEY"]
        self.base_url = os.environ["FIREWORKS_BASE_URL"].rstrip("/")

    def chat_completion(self, model: str, messages: list[dict], max_tokens: int) -> tuple[str, int]:
        """Single-model call. Raises on any failure (network, 404, etc)."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if not resp.ok:
            raise requests.exceptions.HTTPError(
                f"{resp.status_code} error calling model '{model}': {resp.text}"
            )
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        tokens_used = data.get("usage", {}).get("total_tokens", 0)
        return answer, tokens_used

    def chat_completion_with_fallback(
        self, candidates: list[str], messages: list[dict], max_tokens: int
    ) -> tuple[str, int, str]:
        """
        Tries each candidate model in order. Returns (answer, tokens_used, model_used)
        from the first one that succeeds.
        """
        last_error: Exception | None = None
        for model in candidates:
            try:
                answer, tokens = self.chat_completion(model, messages, max_tokens)
                return answer, tokens, model
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        raise AllCandidatesFailedError(
            f"All candidate models failed: {candidates}. Last error: {last_error}"
        )
