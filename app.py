"""Local demo: python3 -m streamlit run app.py"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from rag_upgrade import (
    embedding_client, generate_answer_result, lexical_search, load_rules,
    select_answer_rules, vector_search,
)


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="循答 · 二手设备客服助手", page_icon="♻️", layout="wide")
st.markdown("""<style>
 .stApp {background:#faf8f3;color:#29382f;}
 [data-testid="stSidebar"] {background:#f0eee6;}
 [data-testid="stChatMessage"] {border:1px solid #e0e5dc;border-radius:16px;}
 .block-container {max-width:1100px;padding-top:2rem;}
 h1,h2,h3 {color:#274d37;}
</style>""", unsafe_allow_html=True)
st.title("循答 · 二手设备客服助手")
st.caption("个人学习演示｜模拟回收与二手手机售后，非爱回收官方系统")

policy_path = ROOT / "data" / "policies.md"
using_sample = not policy_path.exists()
if using_sample:
    policy_path = ROOT / "sample_data" / "policies.md"
try:
    rules = load_rules(policy_path)
    policy_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()
except (OSError, ValueError) as exc:
    st.error(f"手册读取失败：{exc}")
    st.stop()

if "turns" not in st.session_state:
    st.session_state.turns = []

examples = {
    "R": ["我昨天同意卖手机了，怎么钱还没到？你帮我查一下。", "你们说我的屏幕有问题，可我寄出时明明是好的。", "估价1200元，检测后怎么变成900元了？"],
    "B": ["我买的二手手机现在送到哪里了？", "我收到二手手机才5天，现在开不了机，能退吗？", "买了20天，手机突然充不了电，可以直接退货吗？"],
    "?": ["我这个手机订单有问题，可以退吗？"],
}
example_prompt = None
with st.sidebar:
    st.subheader("选择业务")
    business_label = st.radio("客户正在办理", ["卖旧手机", "买二手手机", "还不清楚"])
    business = {"卖旧手机": "R", "买二手手机": "B", "还不清楚": "?"}[business_label]
    mode = st.radio("运行方式", ["在线 AI 回复", "离线预览（不调用 API）"])
    st.caption("在线模式使用本机 .env 中的 Key；每次提问会调用模型。请使用模拟客户问题。")
    example = st.selectbox("示例问题", examples[business])
    if st.button("用这条问题试用", use_container_width=True):
        example_prompt = example
    if st.button("清空本次记录", use_container_width=True):
        st.session_state.turns = []
        st.rerun()
    st.divider()
    st.caption(f"已载入 {len(rules)} 条规则 · {'示例手册' if using_sample else '你的项目手册'}")
    with st.expander("使用边界"):
        st.write("每次处理一个主要问题。历史记录用于展示，不自动作为下一问的上下文。没有真实订单、付款或人工工单接口。")
        st.write("模型回复仍需人工审核。找到编号不等于其中每句话都正确。")

if using_sample:
    st.info("当前使用包内的模拟手册；放入项目 data/policies.md 后，会自动使用你的手册。")
st.caption("候选检索用于寻找资料；当前单问题版本只把首条规则交给模型回答，避免附带无关建议。")

def render_turn(turn: dict) -> None:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["reply"])
        if turn["mode"] == "offline":
            if turn["status"] == "offline_excerpt":
                st.caption("离线预览：以上为规则原文摘录，没有调用大模型。")
            else:
                st.caption("离线范围判断或业务澄清，没有调用大模型。")
        else:
            st.caption(f"{turn['business_label']} · 耗时 {turn['elapsed_ms'] / 1000:.1f} 秒 · 回复待人工审核")
        if turn["status"] == "citation_rejected":
            st.warning("这次模型输出未通过引用编号检查，已显示兜底回复。")
        elif turn["status"] == "action_rejected":
            st.warning("模型声称能查询、代办或转人工，但当前没有相应接口；已显示基于规则的安全回复。")
        with st.expander("查看本次回答依据", expanded=False):
            for rule in turn["answer_evidence"]:
                st.markdown(f"**[{rule['id']}]**")
                st.write(rule["text"])
            if not turn["answer_evidence"]:
                st.write("未找到依据。")
        with st.expander("查看检索候选", expanded=False):
            st.caption("候选并非都适用于当前问题。候选 2、3 没有交给回答模型。")
            for rule in turn["candidates"]:
                st.write(f"[{rule['id']}] {rule['text']}")

for turn in st.session_state.turns:
    render_turn(turn)
if not st.session_state.turns:
    st.info("从左侧选一个示例，或在下方输入客户问题。")
else:
    st.download_button(
        "下载本次演示记录（JSON）",
        json.dumps(st.session_state.turns, ensure_ascii=False, indent=2),
        file_name="support_demo_results.json", mime="application/json",
    )

typed = st.chat_input("输入一个完整的客户问题，例如：我同意出售了，回收款怎么还没到？", max_chars=2000)
question = typed or example_prompt
if question and question.strip():
    question = question.strip()
    started = time.perf_counter()
    try:
        with st.spinner("正在查找规则并整理回复…"):
            if mode.startswith("离线"):
                candidates = lexical_search(question, rules, business)
                evidence = select_answer_rules(candidates)
                if business == "?" or not candidates:
                    result = generate_answer_result(question, candidates, None, "offline")
                else:
                    result = {"text": ("\n\n".join(f"[{r['id']}] {r['text']}" for r in evidence) or "未找到依据。"), "raw_text": "", "evidence": evidence, "status": "offline_excerpt"}
                model = "none"
            else:
                # Clarification can be handled without an API request.
                client = None if business == "?" else embedding_client()
                candidates = vector_search(question, rules, business, client, cache_dir=ROOT / ".rag_cache")
                model = os.getenv("QWEN_MODEL", "qwen3.7-flash")
                result = generate_answer_result(question, candidates, client, model)
        st.session_state.turns.append({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "question": question, "business": business, "business_label": business_label,
            "mode": "offline" if mode.startswith("离线") else "online",
            "model": model, "policy_sha256": policy_hash,
            "reply": result["text"], "raw_reply": result["raw_text"],
            "status": result["status"], "answer_evidence": result["evidence"],
            "candidates": candidates, "elapsed_ms": round((time.perf_counter() - started) * 1000),
        })
        st.rerun()
    except Exception as exc:
        # Avoid exposing provider error bodies or configuration secrets.
        status = getattr(exc, "status_code", None)
        st.error(f"暂时无法完成请求（{type(exc).__name__}{' / ' + str(status) if status else ''}）。请检查本机依赖、网络及 .env 配置。")
