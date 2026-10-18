"""Run the eval set against one system and score it by execution accuracy.

    python evals/run_evals.py --system baseline
    python evals/run_evals.py --system pipeline --split test
    python evals/run_evals.py --check-gold      # gold SQL runs and matches itself
    python evals/run_evals.py --report          # print the summary table from saved results

Systems (each adds one thing to the previous):
    baseline           one LLM call, bare schema (table/column names and types)
    baseline_context   one LLM call + schema notes + business glossary
    pipeline_no_retry  planner -> SQL writer -> validator, no retries
    pipeline           full graph: + retry loop that feeds errors back to the writer
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.db import run_query  # noqa: E402
from evals.compare import results_match  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SYSTEMS = ["baseline", "baseline_context", "pipeline_no_retry", "pipeline"]


def load_questions(split: str = "all", ids: list[str] | None = None) -> list[dict]:
    qs = yaml.safe_load((HERE / "questions.yaml").read_text())
    if split != "all":
        qs = [q for q in qs if q["split"] == split]
    if ids:
        qs = [q for q in qs if q["id"] in ids]
    return qs


def answer(system: str, question: str) -> dict:
    """Returns {sql, df, error, usage: [...], retries, validator_errors: [...]}."""
    if system.startswith("baseline"):
        from agent.baseline import generate_sql

        sql, usage = generate_sql(question, with_context=(system == "baseline_context"))
        out = {"sql": sql, "df": None, "error": None, "usage": [usage], "retries": 0, "validator_errors": []}
        try:
            out["df"] = run_query(sql)
        except Exception as e:  # noqa: BLE001 - any failure counts as a wrong answer
            out["error"] = f"{type(e).__name__}: {e}"
        return out

    from agent.graph import run

    state = run(question, max_retries=0 if system == "pipeline_no_retry" else None, report=False)
    return {
        "sql": state.get("sql"),
        "df": state.get("result"),
        "error": state.get("error"),
        "usage": state.get("usage", []),
        "retries": state.get("attempts", 1) - 1,
        "validator_errors": state.get("error_history", []),
    }


def evaluate_one(system: str, q: dict) -> dict:
    start = time.perf_counter()
    record = {"id": q["id"], "difficulty": q["difficulty"], "split": q["split"], "question": q["question"]}
    try:
        out = answer(system, q["question"])
    except Exception as e:  # noqa: BLE001
        out = {"sql": None, "df": None, "error": f"{type(e).__name__}: {e}", "usage": [], "retries": 0,
               "validator_errors": []}
    record.update(
        sql=out["sql"],
        error=out["error"],
        retries=out["retries"],
        validator_errors=out["validator_errors"],
        latency_s=round(time.perf_counter() - start, 2),
        cost_usd=round(sum(u["cost_usd"] for u in out["usage"]), 5),
        llm_calls=len(out["usage"]),
    )
    if out["df"] is None:
        record.update(correct=False, reason=f"no result: {out['error']}")
    else:
        gold = run_query(q["gold_sql"])
        ok, reason = results_match(out["df"], gold, numeric_only=q.get("compare") == "numeric_only")
        record.update(correct=ok, reason=reason)
    return record


def summarize(records: list[dict]) -> dict:
    def acc(rs):
        return round(100 * sum(r["correct"] for r in rs) / len(rs), 1) if rs else None

    by = lambda key, val: [r for r in records if r[key] == val]  # noqa: E731
    return {
        "n": len(records),
        "accuracy": acc(records),
        "easy": acc(by("difficulty", "easy")),
        "medium": acc(by("difficulty", "medium")),
        "hard": acc(by("difficulty", "hard")),
        "dev": acc(by("split", "dev")),
        "test": acc(by("split", "test")),
        "avg_cost_usd": round(sum(r["cost_usd"] for r in records) / len(records), 4),
        "avg_latency_s": round(sum(r["latency_s"] for r in records) / len(records), 1),
        "questions_with_retries": sum(r["retries"] > 0 for r in records),
    }


def run(system: str, split: str, ids: list[str] | None, workers: int) -> dict:
    qs = load_questions(split, ids)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(lambda q: evaluate_one(system, q), qs))
    for r in records:
        mark = "PASS" if r["correct"] else "FAIL"
        print(f"{mark} {r['id']:<4} {r['question'][:60]:<60} {'' if r['correct'] else r['reason'][:80]}")
    summary = summarize(records)
    print(json.dumps(summary, indent=2))
    if split == "all" and not ids:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = {"system": system, "summary": summary, "records": records}
        (RESULTS_DIR / f"{system}.json").write_text(json.dumps(out, indent=2, default=str))
    return summary


def check_gold() -> None:
    bad = 0
    for q in load_questions():
        gold = run_query(q["gold_sql"])
        ok, reason = results_match(gold, gold)
        if not ok or gold.empty:
            bad += 1
            print(f"{q['id']}: {reason if not ok else 'empty result'}")
    print(f"gold check: {bad} problems")


def report() -> None:
    header = "| System | Overall | Easy | Medium | Hard | Test split | Avg cost | Avg latency |"
    lines = [header, "|---|---|---|---|---|---|---|---|"]
    for system in SYSTEMS:
        path = RESULTS_DIR / f"{system}.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text())["summary"]
        lines.append(
            f"| {system} | {s['accuracy']}% | {s['easy']}% | {s['medium']}% | {s['hard']}% | {s['test']}% "
            f"| ${s['avg_cost_usd']:.3f} | {s['avg_latency_s']}s |"
        )
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=SYSTEMS, default="pipeline")
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all")
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--check-gold", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.check_gold:
        check_gold()
    elif args.report:
        report()
    else:
        run(args.system, args.split, args.ids, args.workers)
