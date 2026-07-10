"""
Validates results.json against input tasks.
"""
import json
import sys


def validate(tasks_path: str, results_path: str) -> bool:
    ok = True

    # 1. Valid JSON
    try:
        with open(tasks_path) as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"FAIL: could not parse {tasks_path}: {e}")
        return False

    try:
        with open(results_path) as f:
            results = json.load(f)
    except Exception as e:
        print(f"FAIL: {results_path} is not valid JSON: {e}")
        return False

    # 2. Results must be a list of {task_id, answer} objects.
    if not isinstance(results, list):
        print("FAIL: results.json must be a JSON list")
        ok = False

    task_ids = {t["task_id"] for t in tasks}
    result_ids = set()
    for i, r in enumerate(results):
        if not isinstance(r, dict) or "task_id" not in r or "answer" not in r:
            print(f"FAIL: result[{i}] missing 'task_id' or 'answer' field: {r}")
            ok = False
            continue
        result_ids.add(r["task_id"])
        if not isinstance(r["answer"], str):
            print(f"FAIL: result for {r['task_id']} has non-string answer: {type(r['answer'])}")
            ok = False
        if r["answer"].strip() == "":
            print(f"WARN: result for {r['task_id']} has an empty answer")

    # 3. Every input task must have a corresponding output
    missing = task_ids - result_ids
    if missing:
        print(f"FAIL: {len(missing)} task(s) have no result: {sorted(missing)}")
        ok = False

    extra = result_ids - task_ids
    if extra:
        print(f"WARN: {len(extra)} result(s) reference unknown task_id(s): {sorted(extra)}")

    if ok:
        print(f"PASS: {len(results)}/{len(tasks)} tasks answered, valid JSON, all IDs present.")
    return ok


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python validate_submission.py <tasks.json> <results.json>")
        sys.exit(1)
    success = validate(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
