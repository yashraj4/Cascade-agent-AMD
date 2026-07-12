"""
Main entry point.

Contract (per challenge spec):
  - Read tasks from /input/tasks.json on startup
  - Write results to /output/results.json before exiting
  - Exit code 0 on success, non-zero on failure
  - Max runtime: 10 minutes
  - All model inference must go through FIREWORKS_BASE_URL using an
    ALLOWED_MODELS model - local code-only solving is fine and scores zero
    tokens, but local *LLM* inference would not be recorded at all.

DEFENSIVE DESIGN NOTE: task field names (task_id / prompt) are read with
.get() and common fallbacks, and every task is processed independently with
its own try/except - a single malformed task (wrong field name, missing
key, whatever) must never crash the whole batch and produce zero valid
results. A debug log is also written to /output/debug.log, since container
stdout/stderr may not be visible from the judging harness, but /output is
always yours to inspect afterward.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from .classifier import Category, classify
from .fireworks_client import FireworksClient
from .local_solvers import solve_ner, solve_simple_arithmetic
from .model_router import ModelTiers
from .prompts import build_messages, max_tokens_for
from .sanitize import sanitize_answer

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
DEBUG_LOG_PATH = os.environ.get("DEBUG_LOG_PATH", "/output/debug.log")
MAX_WORKERS = 8

_LOCAL_SOLVERS = {
    # NOTE: sentiment deliberately excluded. Real judged examples require a
    # justification that concretely references BOTH sides of mixed-content
    # reviews (e.g. "late delivery" AND "fast resolution") - VADER can only
    # produce a generic tone-based justification ("the overall tone is
    # positive"), which fails that requirement regardless of whether the
    # label itself is correct. Always escalating sentiment to Fireworks
    # costs a few tokens but protects the whole category from failing on
    # format/content grounds. See app/local_solvers.py for the full story.
    Category.NER: solve_ner,
    Category.MATH: solve_simple_arithmetic,
}

_debug_lines: list[str] = []


def log(msg: str):
    print(msg, file=sys.stderr)
    _debug_lines.append(msg)


def get_task_id(task: dict, fallback_index: int) -> str:
    # Accept a few common field name variants defensively - if the real
    # harness uses "id" instead of "task_id", this must not crash.
    for key in ("task_id", "id", "taskId", "ID"):
        if key in task:
            return str(task[key])
    return f"unknown_task_{fallback_index}"


def get_prompt(task: dict) -> str | None:
    for key in ("prompt", "text", "question", "input", "query"):
        if key in task and isinstance(task[key], str):
            return task[key]
    return None


def solve_task(task: dict, fallback_index: int, tiers: ModelTiers, client: FireworksClient) -> dict:
    task_id = get_task_id(task, fallback_index)
    prompt = get_prompt(task)

    if prompt is None:
        log(f"[warn] task {task_id} has no recognizable prompt field. Keys present: {list(task.keys())}")
        return {"task_id": task_id, "answer": ""}

    try:
        category = classify(prompt)

        local_solver = _LOCAL_SOLVERS.get(category)
        if local_solver is not None:
            local_answer = local_solver(prompt)
            if local_answer is not None:
                log(f"[info] task {task_id} ({category.value}) -> zero-cost local solve")
                return {"task_id": task_id, "answer": local_answer}

        candidates = tiers.candidates_for(category)
        messages = build_messages(category, prompt)
        max_tokens = max_tokens_for(category)
        answer, _tokens_used, model_used = client.chat_completion_with_fallback(
            candidates, messages, max_tokens
        )
        sanitized = sanitize_answer(category, answer)
        if sanitized != answer:
            log(f"[info] task {task_id}: sanitizer adjusted output")
        log(f"[info] task {task_id} ({category.value}) -> {model_used}")
        return {"task_id": task_id, "answer": sanitized}

    except Exception as exc:  # noqa: BLE001 - never let one bad task crash the batch
        log(f"[warn] task {task_id} failed: {exc}")
        _debug_lines.append(traceback.format_exc())
        return {"task_id": task_id, "answer": ""}


def main() -> int:
    try:
        with open(INPUT_PATH, "r") as f:
            tasks = json.load(f)
    except Exception as exc:
        log(f"[fatal] could not read {INPUT_PATH}: {exc}")
        _write_debug_log()
        return 1

    log(f"[info] loaded {len(tasks)} tasks from {INPUT_PATH}")
    if tasks:
        log(f"[info] first task keys: {list(tasks[0].keys())}")

    try:
        allowed_models = os.environ["ALLOWED_MODELS"].split(",")
        tiers = ModelTiers([m.strip() for m in allowed_models if m.strip()])
        client = FireworksClient()
        log(f"[info] ALLOWED_MODELS parsed as: {tiers.allowed_models}")
    except Exception as exc:
        log(f"[fatal] setup failed: {exc}")
        _write_debug_log()
        return 1

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        # Use enumerate for the future->id mapping instead of indexing into
        # task dicts again here - avoids a second crash point on unexpected schemas.
        futures = {
            pool.submit(solve_task, task, i, tiers, client): i
            for i, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001 - defensive: should be unreachable
                # since solve_task already catches everything, but guards
                # against something escaping it (e.g. a bug in solve_task itself).
                fallback_id = get_task_id(tasks[idx], idx)
                log(f"[warn] task {fallback_id} raised outside solve_task: {exc}")
                results.append({"task_id": fallback_id, "answer": ""})

    order = {get_task_id(task, i): i for i, task in enumerate(tasks)}
    results.sort(key=lambda r: order.get(r["task_id"], len(tasks)))

    empty_count = sum(1 for r in results if not r["answer"].strip())
    log(f"[info] done. {len(results)} results written, {empty_count} empty answers.")

    try:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
    except Exception as exc:
        log(f"[fatal] could not write {OUTPUT_PATH}: {exc}")
        _write_debug_log()
        return 1

    _write_debug_log()
    return 0


def _write_debug_log():
    try:
        os.makedirs(os.path.dirname(DEBUG_LOG_PATH), exist_ok=True)
        with open(DEBUG_LOG_PATH, "w") as f:
            f.write("\n".join(_debug_lines))
    except Exception:
        pass  # debug log is best-effort, never let it fail the real run


if __name__ == "__main__":
    sys.exit(main())
