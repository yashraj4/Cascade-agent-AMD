"""
Main entry point.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from .classifier import Category, classify
from .fireworks_client import FireworksClient
from .local_solvers import solve_ner, solve_sentiment, solve_simple_arithmetic
from .model_router import ModelTiers
from .prompts import build_messages, max_tokens_for

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")

# Fallback for local development if paths do not exist
if not os.path.exists(INPUT_PATH) and os.path.exists("input/tasks.json"):
    INPUT_PATH = "input/tasks.json"

if not os.path.exists(os.path.dirname(OUTPUT_PATH)) and os.path.exists("output"):
    OUTPUT_PATH = "output/results.json"
MAX_WORKERS = 8

# Local solvers mapped by category

_LOCAL_SOLVERS = {
    Category.SENTIMENT: solve_sentiment,
    Category.NER: solve_ner,
    Category.MATH: solve_simple_arithmetic,
}


def solve_task(task: dict, tiers: ModelTiers, client: FireworksClient) -> dict:
    task_id = task["task_id"]
    prompt = task["prompt"]

    try:
        category = classify(prompt)

        # Try local solver first

        local_solver = _LOCAL_SOLVERS.get(category)
        if local_solver is not None:
            local_answer = local_solver(prompt)
            if local_answer is not None:
                return {"task_id": task_id, "answer": local_answer}

        # Fallback to model inference

        model = tiers.select(category)
        messages = build_messages(category, prompt)
        max_tokens = max_tokens_for(category)
        answer, _tokens_used = client.chat_completion(model, messages, max_tokens)
        return {"task_id": task_id, "answer": answer}

    except Exception as exc:  # noqa: BLE001

        print(f"[warn] task {task_id} failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {"task_id": task_id, "answer": ""}


def main() -> int:
    try:
        with open(INPUT_PATH, "r") as f:
            tasks = json.load(f)
    except Exception as exc:
        print(f"[fatal] could not read {INPUT_PATH}: {exc}", file=sys.stderr)
        return 1

    try:
        allowed_models = os.environ["ALLOWED_MODELS"].split(",")
        tiers = ModelTiers([m.strip() for m in allowed_models if m.strip()])
        client = FireworksClient()
    except Exception as exc:
        print(f"[fatal] setup failed: {exc}", file=sys.stderr)
        return 1

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(solve_task, task, tiers, client): task["task_id"] for task in tasks}
        for future in as_completed(futures):
            results.append(future.result())

    # Preserve original task order

    order = {task["task_id"]: i for i, task in enumerate(tasks)}
    results.sort(key=lambda r: order.get(r["task_id"], 0))

    try:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
    except Exception as exc:
        print(f"[fatal] could not write {OUTPUT_PATH}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
