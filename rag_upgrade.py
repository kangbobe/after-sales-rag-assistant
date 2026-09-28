"""Standalone RAG experiment for a *simulated* used-phone customer service manual.

No company policy is implied. Place this file beside data/policies.md and run it
from a Cursor terminal. Offline mode requires only Python's standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path


RULE_HEADER = re.compile(r"^\s*(?:#{1,6}\s*)?(?:\[([RBG]\d{2})\]|([RBG]\d{2}))(?!\d)\s*(?:[:：|｜、.\-]\s*)?(.*)$")
DAY = re.compile(r"(?:签收(?:后)?|收到(?:后)?|买了|才|已经|用了)?\s*第?\s*(\d+)\s*天")
FAULT = ("开不了机", "无法开机", "充不了电", "故障", "坏了", "黑屏", "没声音", "不能用", "维修")
DELIVERY_TERMS = ("物流", "快递", "配送", "运输", "派送", "发货", "送到哪", "寄到哪", "到哪里", "到哪儿")


def load_rules(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8-sig")
    if not raw.strip():
        raise ValueError(f"手册是空文件：{path}；请先在 Cursor 中保存 policies.md。")
    rules: list[dict] = []
    active: dict | None = None
    for line in raw.splitlines():
        match = RULE_HEADER.match(line)
        if match:
            rule_id = match.group(1) or match.group(2)
            if any(rule["id"] == rule_id for rule in rules):
                raise ValueError(f"手册中规则编号重复：{rule_id}")
            active = {"id": rule_id, "text": match.group(3).strip()}
            rules.append(active)
        elif line.lstrip().startswith("#"):
            # Markdown section headings are boundaries, not policy text.
            active = None
        elif active is not None and line.strip():
            active["text"] += "\n" + line.strip()
    if not rules:
        raise ValueError("没有识别到规则编号。每条从 [R01]、R01：、## B01 等开头。")
    return rules


def eligible_rules(rules: list[dict], question: str, business: str) -> list[dict]:
    if business == "?":
        return [r for r in rules if r["id"] == "G01"]
    candidates = [r for r in rules if r["id"].startswith(business)]
    day = DAY.search(question)
    if business == "B" and day and any(word in question for word in FAULT):
        n = int(day.group(1))
        if n <= 0 or n > 365:
            candidates = [r for r in candidates if r["id"] not in {"B02", "B03"}]
        else:
            wrong = "B03" if n <= 7 else "B02"
            candidates = [r for r in candidates if r["id"] != wrong]
    return candidates


def bigrams(s: str) -> set[str]:
    chars = re.findall(r"[\u4e00-\u9fff]", s)
    return {chars[i] + chars[i + 1] for i in range(len(chars) - 1)}


def buyer_delivery_intent(question: str, business: str) -> bool:
    # "寄到哪里维修" / "退货快递" are different questions from buying-order delivery.
    other_intent = FAULT + ("退货", "退款", "退回", "寄回", "换货")
    return business == "B" and any(t in question for t in DELIVERY_TERMS) and not any(t in question for t in other_intent)


def recycle_inspection_dispute(question: str, business: str) -> bool:
    """Recognize a complaint about the inspection finding, not shipping damage."""
    if business != "R":
        return False
    if any(t in question for t in ("快递", "物流", "运输", "途中", "丢了", "丢失", "没更新")) and not any(t in question for t in ("检测", "质检", "验机")):
        return False
    dispute = ("明明", "有误", "误判", "不对", "不认可", "争议", "质疑", "复核", "检测错", "质检错")
    inspection = ("检测", "质检", "验机", "检测项", "你们说我的", "你们说屏幕")
    contradicted_condition = any(t in question for t in ("寄出时", "寄出前", "寄送前")) and any(t in question for t in ("明明", "是好的", "没有问题", "没问题", "正常", "完好")) and any(t in question for t in ("屏幕", "外观", "电池", "摄像头", "功能"))
    return contradicted_condition or (any(t in question for t in inspection) and any(t in question for t in dispute))


def rank_with_business_intent(question: str, business: str, scored: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
    """Promote an explicit delivery policy when the buyer asks about delivery.

    Keep similarity as the tie-breaker and retain other retrieved policies.
    This is a transparent business rule; it is not a claim about embedding quality.
    """
    delivery = buyer_delivery_intent(question, business)
    dispute = recycle_inspection_dispute(question, business)
    return sorted(
        scored,
        key=lambda item: (int((delivery and item[1]["id"] == "B06") or (dispute and item[1]["id"] == "R02")), item[0]),
        reverse=True,
    )

def _clearly_outside_scope(question: str) -> bool:
    outside = ("餐厅", "饭店", "天气", "酒店", "旅游", "股票", "电影", "航班")
    business = (
        "手机", "二手", "回收", "订单", "物流", "签收", "退款",
        "退货", "维修", "售后", "检测", "报价", "打款", "寄出", "设备",
    )
    return (
        any(word in question for word in outside)
        and not any(word in question for word in business)
    )

def lexical_search(question: str, rules: list[dict], business: str, top_k: int = 3) -> list[dict]:
    if _clearly_outside_scope(question):
        return []
    if business == "?":
        return eligible_rules(rules, question, business)
    q = bigrams(question)
    matches = []
    for rule in eligible_rules(rules, question, business):
        score = len(q & bigrams(rule["text"]))
        if business == "R" and rule["id"] == "R04" and any(x in question for x in ("款", "打款", "到账", "钱还没到")):
            score += 20
        if business == "B" and rule["id"] in {"B02", "B03"} and DAY.search(question) and any(x in question for x in FAULT):
            score += 20
        if score or (rule["id"] == "B06" and buyer_delivery_intent(question, business)) or (rule["id"] == "R02" and recycle_inspection_dispute(question, business)):
            matches.append((score, rule))
    matches = rank_with_business_intent(question, business, matches)
    return [rule for _, rule in matches[:top_k]]


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("问题向量与手册向量维度不一致。")
    norm = math.sqrt(sum(v * v for v in a) * sum(v * v for v in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0


def embedding_client():
    # Keep .env on your own computer; this script does not log or cache keys.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # An exported environment variable also works.
    from openai import OpenAI

    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("没有找到 DASHSCOPE_API_KEY；请在 Cursor 终端设置环境变量。不要上传密钥。")
    return OpenAI(api_key=key, base_url=os.getenv("QWEN_BASE_URL", "https://maas.qianwenaiapi.com/compatible-mode/v1"), timeout=60.0, max_retries=1)


def embed(client, inputs: list[str], model: str) -> list[list[float]]:
    vectors = []
    for start in range(0, len(inputs), 10):
        response = client.embeddings.create(model=model, input=inputs[start : start + 10])
        vectors.extend([item.embedding for item in sorted(response.data, key=lambda d: d.index)])
    if len(vectors) != len(inputs):
        raise RuntimeError("Embedding 返回的向量数量不匹配。")
    return vectors


def document_vectors(client, rules: list[dict], model: str, cache_dir: Path) -> dict[str, list[float]]:
    # The cache key includes the rule content, model and endpoint, never the API key.
    payload = json.dumps([rules, model, str(client.base_url)], ensure_ascii=False, sort_keys=True)
    key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"policy_vectors_{key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    vectors = embed(client, [f"{r['id']} {r['text']}" for r in rules], model)
    mapping = {rule["id"]: vector for rule, vector in zip(rules, vectors)}
    cache_path.write_text(json.dumps(mapping), encoding="utf-8")
    return mapping


def vector_search(question: str, rules: list[dict], business: str, client, model: str = "text-embedding-v4", cache_dir: Path = Path(".rag_cache"), top_k: int = 3) -> list[dict]:
    if _clearly_outside_scope(question):
        return []
    if business == "?":
        return eligible_rules(rules, question, business)
    candidates = eligible_rules(rules, question, business)
    if not candidates:
        return []
    vectors = document_vectors(client, rules, model, cache_dir)
    q_vector = embed(client, [question], model)[0]
    scored = [(cosine(q_vector, vectors[rule["id"]]), rule) for rule in candidates]
    return [rule for _, rule in rank_with_business_intent(question, business, scored)[:top_k]]


def select_answer_rules(chosen: list[dict]) -> list[dict]:
    # This version handles one main issue at a time. Top-3 are candidates,
    # not three independent reasons to add extra advice to the answer.
    return chosen[:1]


def _unsupported_action(reply: str) -> bool:
    """Block obvious claims of action; manual review still catches subtler ones."""
    patterns = (
        r"我(?:会|将)(?:立即|马上|尽快)?(?:为您)?(?:提交|申请|转接|核对|查询|跟进|推进)",
        r"(?:我们|客服)(?:会|将|需要先).{0,40}为您(?:提交|申请|转接|核对|查询|跟进|推进)",
        r"(?:已经|已)(?:为您)?(?:提交|申请|转接|核对|查到)",
        r"为您(?:转接|提交|申请)(?:人工|复核|工单)",
        r"(?:将|会).{0,35}为您推进处理",
    )
    return any(re.search(pattern, reply) for pattern in patterns)


def _refusal_without_policy_claim(question: str, reply: str) -> bool:
    # 只对明显与手机售后无关的问题，允许模型不引用手册而直接拒答。
    outside = ("餐厅", "饭店", "天气", "酒店", "旅游", "股票", "电影", "航班")
    scope_refusal = bool(re.search(
        r"(?:不(?:在|属于).{0,16}(?:服务范围|业务范围|售后范围)"
        r"|(?:服务范围|业务范围).{0,16}仅限于.{0,20}(?:手机|售后|回收))",
        reply,
    ))
    return (
        any(topic in question for topic in outside)
        and scope_refusal
        and any(word in reply for word in ("无法", "不能", "暂无法"))
    )

def _strip_redundant_business_question(reply: str) -> str:
    return re.sub(r"(?:请问|请确认|请先确认).{0,30}(?:买入|购买|买).{0,18}(?:卖出|出售|卖).{0,16}[？?]\s*$", "", reply).strip()


def _action_fallback(rule_id: str) -> str:
    replies = {
        "R02": "建议您先查看检测说明，并指出有争议的具体项目；如需复核，请联系人工客服申请。[R02] 我目前无法在此代您提交申请。",
        "R04": "请先查看订单中的成交与支付状态。[R04] 我目前无法查询订单进展或确认到账时间；如需进一步核实，建议联系人工客服。",
        "B06": "目前无法直接查询设备所在位置。请查看订单中的物流信息；如有异常或长期未更新，建议联系人工核实。[B06]",
        "B02": "签收后7天内出现疑似非人为功能故障，请先申请检测；检测确认后可按规则申请退货或维修，暂不能保证直接退货。[B02]",
        "B03": "签收第8天至第365天出现疑似非人为功能故障，可先申请检测；检测确认后可申请维修，不能保证直接退货。[B03]",
    }
    return replies.get(rule_id, "目前无法在对话中代您查询或办理申请，建议联系人工核实。")


def generate_answer_result(question: str, chosen: list[dict], client, model: str) -> dict:
    if _clearly_outside_scope(question):
        return {
            "text": "抱歉，我只能回答二手手机售后相关问题。",
            "raw_text": "",
            "evidence": [],
            "status": "out_of_scope_refusal",
        }
    evidence = select_answer_rules(chosen)
    if not evidence:
        return {"text": "目前没有足够信息确认，建议人工核实。", "raw_text": "", "evidence": [], "status": "no_evidence"}
    if evidence[0]["id"] == "G01":
        return {"text": "请问您是在出售旧手机，还是购买二手手机后遇到了问题？", "raw_text": "", "evidence": evidence, "status": "clarification"}
    context = "\n\n".join(f"[{r['id']}] {r['text']}" for r in evidence)
    business_hint = {"R": "用户已选择卖旧手机业务。", "B": "用户已选择买二手手机业务。"}.get(evidence[0]["id"][0], "")
    system = (
        "你是二手手机业务学习演示中的售后客服。仅按所给规则回答。"
        + business_hint +
        "回复要像真实客服对用户说话，不要说'本项目'、'模拟手册'、'编造'、'规则缺少依据'等内部措辞。"
        "你没有查单、提交申请或转接人工的工具；不能声称或承诺'我会为您提交/转接/跟进'，也不能保证到账日期、赔偿金额或未提供的办理渠道。"
        "不要在当前聊天索取订单号、手机号、密码、验证码、地址或支付信息。业务已确认时不要再问用户买入还是卖出。"
        "无法查询订单时说'目前无法查询您的订单进展或确认到账时间'；资料不足时说'目前无法确认，建议人工核实'。"
        "用户说手机无法开机时，只能称为'疑似功能故障'；未经检测不能说'属于功能故障'或'已确认故障'。"
        "每项事实在同一句末引用直接支持它的规则编号，例如 [B02]；仅可引用所给编号，勿引用无关规则。"
        "用户未说清买入还是卖出时先询问业务类型，不猜测处理政策。"
        "只回答用户这次提出的问题。不要补充未询问的退货、隐私、备份等事项；不要以泛泛的服务邀约结尾。"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": f"手册内容：\n{context}\n\n用户问题：{question}"}],
    )
    result = response.choices[0].message.content or ""
    normalized = re.sub(r"\[\s*([RBG]\d{2})\s*\]", r"[\1]", result)
    normalized = _strip_redundant_business_question(normalized)
    citations = set(re.findall(r"\[([RBG]\d{2})\]", normalized))
    allowed = {rule["id"] for rule in evidence}
    if _unsupported_action(normalized):
        return {"text": _action_fallback(evidence[0]["id"]), "raw_text": result, "evidence": evidence, "status": "action_rejected"}
    if not citations and _refusal_without_policy_claim(question, normalized):
        return {"text": normalized, "raw_text": result, "evidence": evidence, "status": "out_of_scope_refusal"}
    if citations - allowed or (not citations and allowed):
        return {"text": "回复中的依据暂时无法核对，建议人工核实。", "raw_text": result, "evidence": evidence, "status": "citation_rejected"}
    return {"text": normalized, "raw_text": result, "evidence": evidence, "status": "needs_review"}


def answer(question: str, chosen: list[dict], client, model: str) -> str:
    return generate_answer_result(question, chosen, client, model)["text"]


def main() -> None:
    parser = argparse.ArgumentParser(description="模拟客服 RAG：离线关键词 / 在线向量检索 + 千问回答")
    parser.add_argument("question")
    parser.add_argument("--business", choices=("R", "B", "?"), required=True, help="R 卖旧手机，B 买二手手机，? 不清楚")
    parser.add_argument("--policies", type=Path, default=Path("data/policies.md"))
    parser.add_argument("--mode", choices=("lexical", "vector"), default="lexical")
    parser.add_argument("--answer", action="store_true", help="追加大模型回答；需要 API Key")
    args = parser.parse_args()
    rules = load_rules(args.policies)
    client = embedding_client() if args.mode == "vector" or args.answer else None
    selected = (vector_search(args.question, rules, args.business, client) if args.mode == "vector"
                else lexical_search(args.question, rules, args.business))
    print("命中的规则：", ", ".join(f"[{r['id']}]" for r in selected) or "无")
    if args.answer:
        print("客服回复：", answer(args.question, selected, client, os.getenv("QWEN_MODEL", "qwen3.7-flash")))


if __name__ == "__main__":
    main()
