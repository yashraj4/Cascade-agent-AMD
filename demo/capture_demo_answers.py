"""
Run this ONCE to capture real Fireworks answers for the demo site.

This deliberately makes only ~8-10 real API calls total (one per category,
plus one extra forced to Gemma specifically) so it costs a small fraction of
your $50 credit, not the full budget.

USAGE:
    1. Deploy Gemma at https://app.fireworks.ai/models (pick the cheapest
       variant, e.g. gemma-4-26b-a4b-it) if you want a genuine Gemma example.
    2. export FIREWORKS_API_KEY="your-real-key"
       export FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1"
       export ALLOWED_MODELS="minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4"
    3. python3 capture_demo_answers.py
    4. IMMEDIATELY undeploy Gemma at https://app.fireworks.ai/models to stop
       the ~$7/hour idle billing.
    5. This writes demo_answers.json - copy it next to index.html.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "..")  # so we can import the real app package

from app.classifier import classify
from app.fireworks_client import FireworksClient
from app.local_solvers import solve_ner, solve_sentiment, solve_simple_arithmetic
from app.model_router import ModelTiers
from app.prompts import build_messages, max_tokens_for

# One representative prompt per category. Kept short deliberately - this is
# a demo, not the real eval, so there's no reason to spend extra tokens on
# larger prompts.
DEMO_PROMPTS = [
    {"category_hint": "factual_knowledge", "prompt": "What causes the seasons on Earth?"},
    {"category_hint": "mathematical_reasoning", "prompt": "If a shirt costs $40 and is discounted by 25%, what is the final price?"},
    {"category_hint": "sentiment_classification", "prompt": "Classify the sentiment: 'The service was slow and the food was cold.'"},
    {"category_hint": "text_summarisation", "prompt": "Summarise in one sentence: Solar panels convert sunlight into electricity using photovoltaic cells, and their efficiency has improved significantly over the past decade while costs have fallen."},
    {"category_hint": "named_entity_recognition", "prompt": "Extract the named entities from: Marie Curie won the Nobel Prize in Stockholm in 1911."},
    {"category_hint": "code_debugging", "prompt": "This code has a bug, fix it: def is_even(n): return n % 2 == 1"},
    {"category_hint": "logical_deductive_reasoning", "prompt": "Three boxes are labeled Apples, Oranges, and Mixed, but all labels are wrong. You pick one fruit from the box labeled Mixed and get an apple. What is in each box?"},
    {"category_hint": "code_generation", "prompt": "Write a Python function that checks if a string is a palindrome."},
]


def main():
    allowed_models = os.environ["ALLOWED_MODELS"].split(",")
    tiers = ModelTiers([m.strip() for m in allowed_models if m.strip()])
    client = FireworksClient()

    results = []
    for item in DEMO_PROMPTS:
        prompt = item["prompt"]
        category = classify(prompt)

        # Try the zero-cost local solver first, same as the real app -
        # some of these will resolve locally and cost nothing at all.
        local_solvers = {
            "sentiment_classification": solve_sentiment,
            "named_entity_recognition": solve_ner,
            "mathematical_reasoning": solve_simple_arithmetic,
        }
        solver = local_solvers.get(category.value)
        if solver:
            local_answer = solver(prompt)
            if local_answer is not None:
                results.append({
                    "prompt": prompt,
                    "category": category.value,
                    "path": "zero_cost_local",
                    "model_used": None,
                    "answer": local_answer,
                    "tokens_used": 0,
                })
                print(f"[local]     {category.value}: resolved with zero tokens")
                continue

        candidates = tiers.candidates_for(category)
        messages = build_messages(category, prompt)
        max_tokens = max_tokens_for(category)
        answer, tokens_used, model_used = client.chat_completion_with_fallback(
            candidates, messages, max_tokens
        )
        results.append({
            "prompt": prompt,
            "category": category.value,
            "path": "fireworks_escalation",
            "model_used": model_used,
            "answer": answer,
            "tokens_used": tokens_used,
        })
        print(f"[fireworks] {category.value}: {model_used} ({tokens_used} tokens)")

    with open("demo_answers.json", "w") as f:
        json.dump(results, f, indent=2)

    total_tokens = sum(r["tokens_used"] for r in results)
    print(f"\nDone. Total tokens spent: {total_tokens}")
    print("REMINDER: undeploy Gemma now at https://app.fireworks.ai/models if you deployed it.")
    print("demo_answers.json written - copy it next to demo/index.html")


if __name__ == "__main__":
    main()
