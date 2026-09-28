"""Regression cases from the 2026-09-28 seven-answer local run.

Tests the guard, not the quality of future model generations.
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rag_upgrade import generate_answer_result, load_rules


class Completion:
    def __init__(self, reply):
        self.reply = reply
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


class ProductionAnswerCases(unittest.TestCase):
    def run_case(self, rule_id, reply, question="模拟客户问句"):
        return generate_answer_result(question, [{"id": rule_id, "text": "对应模拟条款"}], Completion(reply), "mock")

    def test_promised_followup_on_recycling_payment_is_blocked(self):
        reply = "我们需要先核对订单中的成交和支付状态来为您跟进[R04]。建议您补充订单信息。"
        result = self.run_case("R04", reply)
        self.assertEqual(result["status"], "action_rejected")
        self.assertNotIn("为您跟进", result["text"])
        self.assertIn("[R04]", result["text"])

    def test_promised_manual_recheck_is_blocked(self):
        result = self.run_case("R02", "确认具体异议后，我会立即为您提交人工复核申请[R02]。")
        self.assertEqual(result["status"], "action_rejected")
        self.assertIn("联系人工", result["text"])

    def test_promised_handoff_is_blocked(self):
        result = self.run_case("B06", "如果长期未更新，我会为您转接人工专员核实[B06]。")
        self.assertEqual(result["status"], "action_rejected")
        self.assertNotIn("我会为您", result["text"])

    def test_future_processing_commitment_is_blocked(self):
        result = self.run_case("B02", "请您提交检测申请，后续将依据检测报告的实际结论为您推进处理。[B02]")
        self.assertEqual(result["status"], "action_rejected")
        self.assertIn("先申请检测", result["text"])

    def test_citation_whitespace_normalized_and_redundant_question_removed(self):
        result = self.run_case("B03", "签收第8天应申请检测，确认后可申请维修[ B03]。请问您是买入还是卖出业务？")
        self.assertEqual(result["status"], "needs_review")
        self.assertIn("[B03]", result["text"])
        self.assertNotIn("买入还是卖出", result["text"])
        self.assertIn("[ B03]", result["raw_text"])

    def test_obviously_unrelated_question_is_refused_without_policy_or_model(self):
        result = self.run_case("R02", "餐厅推荐不在我们的服务范围内，暂无法提供协助。", "请推荐新加坡餐厅")
        self.assertEqual(result["status"], "out_of_scope_refusal")
        self.assertIn("二手手机售后", result["text"])
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["raw_text"], "")

    def test_bad_refusal_to_phone_after_sales_is_not_trusted(self):
        result = self.run_case("B02", "退货不在我们的服务范围内，暂无法处理。", "手机签收后第5天坏了，能退吗？")
        self.assertEqual(result["status"], "citation_rejected")

    def test_wrong_citation_remains_blocked(self):
        result = self.run_case("R04", "按物流规定处理。[R05]")
        self.assertEqual(result["status"], "citation_rejected")

    def test_section_heading_not_appended_to_prior_policy(self):
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "policies.md"
            path.write_text("## [B06] 物流\n到订单页查看物流。\n## C. 共用规则\n## [G01] 未知\n先询问。", encoding="utf-8")
            rules = load_rules(path)
            self.assertEqual([r["id"] for r in rules], ["B06", "G01"])
            self.assertNotIn("共用规则", rules[0]["text"])


if __name__ == "__main__":
    unittest.main()
