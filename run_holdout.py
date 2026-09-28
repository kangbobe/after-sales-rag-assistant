"""Run new, simulated questions once against the local model and save raw results.

The script reports retrieval matches only. Human review must judge model replies.
Requests in vector mode use the account configured in the local .env file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from rag_upgrade import (
    embedding_client,
    generate_answer_result,
    lexical_search,
    load_rules,
    vector_search,
)


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run new simulated support scenarios")
    parser.add_argument("--mode", choices=("lexical", "vector"), default="vector")
    parser.add_argument("--policies", type=Path, default=ROOT / "data" / "policies.md")
    parser.add_argument("--cases", type=Path, default=ROOT / "holdout_cases.json")
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    rules = load_rules(args.policies)
    client = embedding_client() if args.mode == "vector" else None
    output = ROOT / f"holdout_results_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    report = {
        "model": "qwen3.7-flash (or local QWEN_MODEL)" if client else "none",
        "mode": args.mode,
        "policy_sha256": hashlib.sha256(args.policies.read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "manual_review_required": True,
        "cases": [],
    }
    if client:
        import os
        report["model"] = os.getenv("QWEN_MODEL", "qwen3.7-flash")
    print(f"Running {len(cases)} NEW simulated cases in {args.mode} mode.", flush=True)

    for case in cases:
        started = time.perf_counter()
        try:
            if client:
                candidates = vector_search(
                    case["question"], rules, case["business"], client,
                    cache_dir=ROOT / ".rag_cache",
                )
                result = generate_answer_result(
                    case["question"], candidates, client, report["model"]
                )
            else:
                candidates = lexical_search(case["question"], rules, case["business"])
                result = None
            expected = case["expected_rule"]
            top1 = candidates[0]["id"] if candidates else None
            report["cases"].append({
                **case,
                "retrieved_top1": top1,
                "retrieval_top1_matches": top1 == expected if expected is not None else None,
                "candidates": candidates,
                "reply": result["text"] if result else "",
                "raw_reply": result["raw_text"] if result else "",
                "status": result["status"] if result else "retrieval_only",
                "evidence": result["evidence"] if result else [],
                "human_answer_review": None,
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            })
            print(f"{case['id']}: top1={top1 or 'none'}; manual answer review pending", flush=True)
        except Exception as exc:
            report["cases"].append({
                "id": case["id"],
                "status": "api_error",
                "error_type": type(exc).__name__,
                "http_status": getattr(exc, "status_code", None),
            })
            print(f"{case['id']}: stopped after {type(exc).__name__}", flush=True)
            break
        finally:
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Results saved to {output.name}. Upload this JSON for human review; keep .env private.")


if __name__ == "__main__":
    main()
