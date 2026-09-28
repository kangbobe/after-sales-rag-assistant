"""Compare retrieval top-1 against a labeled set; generation quality is separate."""
import argparse
import json
from pathlib import Path

from rag_upgrade import embedding_client, lexical_search, load_rules, vector_search


def input_path(path: Path) -> Path:
    """Find files beside this script when Cursor's terminal opens elsewhere."""
    if path.exists():
        return path
    beside_script = Path(__file__).resolve().parent / path
    if beside_script.exists():
        return beside_script
    raise SystemExit(
        f"找不到文件：{path}\n"
        "请确认 ZIP 已解压，并且 evaluate_rag_upgrade.py、"
        "sample_cases.json 和 sample_data/ 在同一个项目文件夹。"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policies", type=Path, default=Path("data/policies.md"))
    p.add_argument("--cases", type=Path, default=Path("sample_cases.json"))
    p.add_argument("--mode", choices=("lexical", "vector"), default="lexical")
    args = p.parse_args()
    rules = load_rules(input_path(args.policies))
    cases = json.loads(input_path(args.cases).read_text(encoding="utf-8"))
    client = embedding_client() if args.mode == "vector" else None
    passed = 0
    for case in cases:
        found = (vector_search(case["question"], rules, case["business"], client) if client
                 else lexical_search(case["question"], rules, case["business"]))
        actual = found[0]["id"] if found else "无"
        success = actual == case["expected"]
        passed += success
        print(f"{case['id']} {('通过' if success else '失败')} 预期:{case['expected']} 实际:{actual}")
    print(f"检索首条正确：{passed}/{len(cases)}（测试资料为模拟；不是模型回复准确率）")


if __name__ == "__main__":
    main()
