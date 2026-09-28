"""Offline checks: no API key, network, or third-party packages required."""
import tempfile
import unittest
from pathlib import Path

from rag_upgrade import eligible_rules, lexical_search, load_rules, cosine, answer, vector_search, recycle_inspection_dispute, generate_answer_result


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.rules = [
            {"id": "G01", "text": "不清楚是卖出还是买入时先询问业务类型"},
            {"id": "R04", "text": "回收款到账需要核实订单状态，不得承诺具体打款日期"},
            {"id": "R05", "text": "寄出旧手机的物流异常转人工查验，不得承诺赔偿"},
            {"id": "R01", "text": "旧手机估价和检测后的最终报价可能不同"},
            {"id": "R02", "text": "质疑检测结果时请指出具体检测项目并申请人工复核"},
            {"id": "B02", "text": "签收后 7 天内功能故障须经检测确认，才可申请退货或维修"},
            {"id": "B03", "text": "签收第 8 天到 365 天功能故障须经检测确认，可申请维修"},
            {"id": "B06", "text": "买入订单的物流进度和当前位置需要在订单页面查看"},
        ]

    def test_load_saved_markdown_and_reject_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policies.md"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "空文件"):
                load_rules(path)
            path.write_text("## [R04] 回收款\n订单需要人工核实。\n\nB02：签收后故障\n先检测。", encoding="utf-8")
            rules = load_rules(path)
            self.assertEqual([r["id"] for r in rules], ["R04", "B02"])
            self.assertIn("人工核实", rules[0]["text"])

    def test_day_boundaries_filter_ineligible_rule(self):
        for day, correct, incorrect in [(5, "B02", "B03"), (7, "B02", "B03"), (8, "B03", "B02"), (20, "B03", "B02")]:
            with self.subTest(day=day):
                ids = {r["id"] for r in eligible_rules(self.rules, f"收到 {day} 天开不了机", "B")}
                self.assertIn(correct, ids)
                self.assertNotIn(incorrect, ids)

    def test_ambiguous_business_returns_only_clarification(self):
        chosen = lexical_search("我这个手机订单有问题，可以退吗？", self.rules, "?")
        self.assertEqual([r["id"] for r in chosen], ["G01"])

    def test_recycle_payment_does_not_hit_buying_policy(self):
        chosen = lexical_search("卖掉了手机但回收款还没到账", self.rules, "R")
        self.assertEqual(chosen[0]["id"], "R04")
        self.assertTrue(all(r["id"].startswith("R") for r in chosen))

    def test_cosine_is_scale_invariant_and_dimension_checked(self):
        self.assertAlmostEqual(cosine([1.0, 0.0], [10.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine([1.0, 0.0], [0.0, 1.0]), 0.0)
        with self.assertRaises(ValueError):
            cosine([1.0], [1.0, 2.0])

    def test_unknown_citation_is_not_delivered(self):
        class Response:
            choices = [type("Choice", (), {"message": type("Message", (), {"content": "已查到订单，今天到账。[R99]"})()})()]

        class Client:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        return Response()

        result = answer("钱呢？", [self.rules[1]], Client(), "mock")
        self.assertNotIn("已查到订单", result)
        self.assertIn("无法查询", result)

    def test_vector_search_uses_semantic_vectors_and_day_filter(self):
        class Item:
            def __init__(self, index, vector):
                self.index, self.embedding = index, vector

        class Client:
            base_url = "https://example.invalid/v1"

            def __init__(self):
                self.embeddings = self

            def create(self, *, model, input):
                vectors = []
                for text in input:
                    if text.startswith("B02") or "送到哪里" in text:
                        vector = [1.0, 0.0]
                    elif text.startswith("B03"):
                        vector = [0.0, 1.0]
                    else:
                        vector = [0.0, 0.0]
                    vectors.append(vector)
                return type("Response", (), {"data": [Item(i, v) for i, v in enumerate(vectors)]})()

        with tempfile.TemporaryDirectory() as tmp:
            found = vector_search("收到第 8 天开不了机，送到哪里？", self.rules, "B", Client(), cache_dir=Path(tmp))
            self.assertEqual(found[0]["id"], "B03")
            self.assertNotIn("B02", {r["id"] for r in found})

    def test_delivery_intent_promotes_b06_without_promoting_repair_shipping(self):
        # The mock deliberately puts B02 closer than B06: verify the explicit
        # business constraint solves the failure observed on the user's Mac.
        class Item:
            def __init__(self, index, vector):
                self.index, self.embedding = index, vector

        class Client:
            base_url = "https://example.invalid/v1"

            def __init__(self):
                self.embeddings = self

            def create(self, *, model, input):
                data = []
                for index, text in enumerate(input):
                    vector = [1.0, 0.0] if text.startswith("B02") or "送到哪里" in text else [0.0, 1.0]
                    data.append(Item(index, vector))
                return type("Response", (), {"data": data})()

        with tempfile.TemporaryDirectory() as tmp:
            got = vector_search("我买的二手手机现在送到哪里了？", self.rules, "B", Client(), cache_dir=Path(tmp))
            self.assertEqual(got[0]["id"], "B06")
            other = lexical_search("手机坏了要寄到哪里维修？", self.rules, "B")
            self.assertNotEqual(other[0]["id"], "B06")

    def test_disputed_inspection_promotes_r02_but_not_price_or_shipping(self):
        question = "你们说我的屏幕有问题，可我寄出时明明是好的。"
        self.assertTrue(recycle_inspection_dispute(question, "R"))
        self.assertEqual(lexical_search(question, self.rules, "R")[0]["id"], "R02")
        self.assertTrue(recycle_inspection_dispute("质检说电池损坏，但我寄出前电池明明正常", "R"))
        self.assertTrue(recycle_inspection_dispute("我不认可检测结果，想复核屏幕", "R"))
        self.assertFalse(recycle_inspection_dispute("检测完怎么从1200元降到900元？", "R"))
        self.assertFalse(recycle_inspection_dispute("快递没更新，手机是不是丢了？", "R"))

        class Item:
            def __init__(self, index, vector):
                self.index, self.embedding = index, vector

        class Client:
            base_url = "https://example.invalid/v1"

            def __init__(self):
                self.embeddings = self

            def create(self, *, model, input):
                data = []
                for index, text in enumerate(input):
                    # Simulate the observed failure: R05 wins raw similarity.
                    vector = [1.0, 0.0] if text.startswith("R05") or "明明" in text else [0.0, 1.0]
                    data.append(Item(index, vector))
                return type("Response", (), {"data": data})()

        with tempfile.TemporaryDirectory() as tmp:
            found = vector_search(question, self.rules, "R", Client(), cache_dir=Path(tmp))
            self.assertEqual(found[0]["id"], "R02")

    def test_payment_reply_cannot_use_unrelated_retrieval_candidates(self):
        from types import SimpleNamespace
        captured = []

        class Completions:
            @staticmethod
            def create(**kwargs):
                captured.append(kwargs)
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="请先备份手机。[R06]"))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        candidates = [
            {"id": "R04", "text": "回收款进度需查看订单中的成交与支付状态。"},
            {"id": "R06", "text": "寄出前备份资料。"},
            {"id": "R03", "text": "设备退回与运费需要核实。"},
        ]
        result = generate_answer_result("回收款何时到账？", candidates, client, "mock")
        model_context = captured[0]["messages"][1]["content"]
        self.assertIn("[R04]", model_context)
        self.assertNotIn("[R06]", model_context)
        self.assertNotIn("[R03]", model_context)
        self.assertEqual(result["status"], "citation_rejected")
        self.assertEqual(result["raw_text"], "请先备份手机。[R06]")
        self.assertNotIn("备份", result["text"])

    def test_ambiguous_business_asks_question_without_api(self):
        result = generate_answer_result("订单出问题了", [self.rules[0]], None, "unused")
        self.assertEqual(result["status"], "clarification")
        self.assertIn("出售", result["text"])
        self.assertIn("购买", result["text"])


if __name__ == "__main__":
    unittest.main()
