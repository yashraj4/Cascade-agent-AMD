"""
API client for Fireworks inference.
"""
from __future__ import annotations

import os

import requests


class FireworksClient:
    def __init__(self):
        self.api_key = os.environ["FIREWORKS_API_KEY"]
        self.base_url = os.environ["FIREWORKS_BASE_URL"].rstrip("/")

    def chat_completion(self, model: str, messages: list[dict], max_tokens: int) -> tuple[str, int]:
        """Returns (answer_text, tokens_used)."""
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
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        tokens_used = data.get("usage", {}).get("total_tokens", 0)
        return answer, tokens_used
