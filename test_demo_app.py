"""Streamlit UI flow checks; only offline data and a stubbed API are used."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest


class DemoAppTests(unittest.TestCase):
    def test_offline_question_and_evidence(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
        self.assertFalse(app.exception)
        app.sidebar.radio[1].set_value("离线预览（不调用 API）").run()
        app.chat_input[0].set_value("回收款怎么还没到账？").run()
        self.assertFalse(app.exception)
        row = app.session_state["turns"][0]
        self.assertEqual(row["mode"], "offline")
        self.assertEqual([r["id"] for r in row["answer_evidence"]], ["R04"])

    def test_online_ui_uses_only_primary_evidence_with_mock_api(self):
        calls = []
        class Completions:
            @staticmethod
            def create(**kwargs):
                calls.append(kwargs)
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="请查看订单的成交与支付状态。[R04]"))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        candidates = [{"id": "R04", "text": "回收款需查看成交与支付状态。"}, {"id": "R06", "text": "备份资料。"}, {"id": "R03", "text": "退回设备。"}]
        with patch("rag_upgrade.embedding_client", return_value=client), patch("rag_upgrade.vector_search", return_value=candidates):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
            app.chat_input[0].set_value("回收款还没有到账").run()
        self.assertFalse(app.exception)
        row = app.session_state["turns"][0]
        self.assertEqual(row["status"], "needs_review")
        self.assertIn("[R04]", row["reply"])
        self.assertEqual(len(row["candidates"]), 3)
        self.assertEqual(len(row["answer_evidence"]), 1)
        self.assertNotIn("[R06]", calls[0]["messages"][1]["content"])

    def test_unknown_business_does_not_call_api(self):
        with patch("rag_upgrade.embedding_client", side_effect=AssertionError("Must not call API")):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
            app.sidebar.radio[0].set_value("还不清楚").run()
            app.chat_input[0].set_value("可以退吗？").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["turns"][0]["status"], "clarification")

    def test_offline_outside_scope_does_not_show_unrelated_policy(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
        app.sidebar.radio[1].set_value("离线预览（不调用 API）").run()
        app.chat_input[0].set_value("请推荐新加坡餐厅").run()
        self.assertFalse(app.exception)
        row = app.session_state["turns"][0]
        self.assertEqual(row["status"], "out_of_scope_refusal")
        self.assertEqual(row["candidates"], [])
        self.assertEqual(row["answer_evidence"], [])


if __name__ == "__main__":
    unittest.main()
