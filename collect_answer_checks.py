"""Collect model replies and their actual evidence for human review.

Run once locally: python3 collect_answer_checks.py
This calls paid APIs. It never marks generated answers automatically correct.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from rag_upgrade import embedding_client, generate_answer_result, load_rules, vector_search

ROOT = Path(__file__).resolve().parent
CASES = [
    ("A01", "R", "我同意卖手机了，但回收款一直没收到，能帮我查到账时间吗？", "R04", "不得附带退货或寄出前隐私提醒；不得虚构付款状态。"),
    ("A02", "R", "我不认可你们的屏幕检测结果，请帮我复核一下。", "R02", "建议申请复核，但不能说已经代为提交申请。"),
    ("A03", "B", "我买的二手手机现在送到哪里了？", "B06", "不得虚构当前位置或配送时间。"),
    ("A04", "B", "签收第7天手机开不了机，能不能退？", "B02", "不得跳过检测直接保证退货。"),
    ("A05", "B", "签收第8天手机开不了机，能不能直接退？", "B03", "不能套用7天内政策直接承诺退货。"),
    ("A06", "?", "我的手机订单出了问题，可以退吗？", "G01", "先澄清买入还是卖出。"),
    ("A07", "R", "请给我推荐一家新加坡餐厅。", None, "明显无关问题应直接拒答且不返回候选；此检查不覆盖所有手册外问法。"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", type=Path, default=ROOT / "data" / "policies.md")
    parser.add_argument("--only", help="只收集指定编号，逗号分隔，例如 A01,A02,A05")
    args = parser.parse_args()
    selected = CASES
    if args.only:
        wanted = {part.strip().upper() for part in args.only.split(",") if part.strip()}
        available = {case[0] for case in CASES}
        if not wanted or wanted - available:
            raise SystemExit("未知案例编号：" + ",".join(sorted(wanted - available or wanted)))
        selected = [case for case in CASES if case[0] in wanted]
    path = args.policies
    if not path.exists() and path == ROOT / "data" / "policies.md":
        path = ROOT / "sample_data" / "policies.md"
    rules = load_rules(path)
    try:
        client = embedding_client()
    except Exception as exc:
        raise SystemExit(f"无法初始化模型连接：{type(exc).__name__}。请检查依赖及本机 .env。") from None
    model = os.getenv("QWEN_MODEL", "qwen3.7-flash")
    output = ROOT / f"answer_checks_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}.json"
    rows = []
    report = {"model": model, "policy_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "policy_file": path.name, "manual_review_required": True, "cases": rows}
    print(f"将采集{len(selected)}条回复；仅记录结果，不自动判定答案正确。", flush=True)
    for test_id, business, question, expected, review_note in selected:
        started = time.perf_counter()
        try:
            candidates = vector_search(question, rules, business, client, cache_dir=ROOT / ".rag_cache")
            result = generate_answer_result(question, candidates, client, model)
            rows.append({"id": test_id, "question": question, "business": business, "expected_rule": expected,
                         "retrieval_top1_matches": (bool(candidates and candidates[0]["id"] == expected) if expected is not None else None),
                         "review_note": review_note, "candidates": candidates, "reply": result["text"],
                         "raw_reply": result["raw_text"], "status": result["status"], "evidence": result["evidence"],
                         "elapsed_ms": round((time.perf_counter() - started) * 1000)})
            print(f"{test_id} 已采集，首条规则：{candidates[0]['id'] if candidates else '无'}；回复待人工审核。", flush=True)
        except Exception as exc:
            rows.append({"id": test_id, "question": question, "status": "api_error", "error_type": type(exc).__name__, "http_status": getattr(exc, "status_code", None)})
            print(f"{test_id} 请求失败（{type(exc).__name__}），已停止并保留之前的结果。", flush=True)
            break
        finally:
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"记录已保存：{output}\n把这个 JSON 文件上传到对话即可，不需要上传 .env。")


if __name__ == "__main__":
    main()
