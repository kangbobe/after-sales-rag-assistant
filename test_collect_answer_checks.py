import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import collect_answer_checks as collector


class CollectionTests(unittest.TestCase):
    def test_partial_results_survive_failure_without_provider_error_body(self):
        policies = Path(__file__).parent / "sample_data" / "policies.md"
        evidence = [{"id": "R04", "text": "查看订单状态"}]
        result = {"text": "查看订单状态。[R04]", "raw_text": "查看订单状态。[R04]", "evidence": evidence, "status": "needs_review"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(collector, "ROOT", root), patch.object(collector, "embedding_client", return_value=object()), patch.object(collector, "vector_search", side_effect=[evidence, RuntimeError("provider private error body")]), patch.object(collector, "generate_answer_result", return_value=result), patch("sys.argv", ["collect_answer_checks.py", "--policies", str(policies)]), contextlib.redirect_stdout(io.StringIO()):
                collector.main()
            output = next(root.glob("answer_checks_*.json"))
            raw = output.read_text(encoding="utf-8")
            report = json.loads(raw)
            self.assertEqual(len(report["cases"]), 2)
            self.assertEqual(report["cases"][0]["raw_reply"], result["raw_text"])
            self.assertEqual(report["cases"][1]["status"], "api_error")
            self.assertNotIn("provider private error body", raw)


if __name__ == "__main__":
    unittest.main()
