import html
import os
import re
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

from retrieval_pipe import PolicyRetrievalPipeline

# 与原应用保持一致的环境配置
os.environ["HF_ENDPOINT"] = "http://hf-mirror.com"
load_dotenv()


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".odt",
    ".rtf",
    ".html",
    ".htm",
    ".xhtml",
}


def scan_policy_files(data_dir: str = "./policy_data") -> List[str]:
    """扫描 data_dir 中可作为输入的政策文件，返回相对路径列表。"""
    files: List[str] = []
    if not os.path.isdir(data_dir):
        return files

    for root, _, filenames in os.walk(data_dir):
        for name in filenames:
            ext = Path(name).suffix.lower()
            if ext in ALLOWED_EXTENSIONS:
                full_path = os.path.join(root, name)
                rel_path = os.path.relpath(full_path, ".")
                files.append(rel_path)
    files.sort()
    return files


def save_uploaded_files(uploaded_files, base_dir: str = "./policy_data/uploads") -> List[str]:
    """将通过 Streamlit 上传的文件保存到本地，返回保存后的相对路径列表。"""
    saved_paths: List[str] = []
    if not uploaded_files:
        return saved_paths

    os.makedirs(base_dir, exist_ok=True)

    for uf in uploaded_files:
        filename = uf.name
        # 简单防止路径穿越
        filename = os.path.basename(filename)
        save_path = os.path.join(base_dir, filename)
        with open(save_path, "wb") as f:
            f.write(uf.getbuffer())
        rel_path = os.path.relpath(save_path, ".")
        saved_paths.append(rel_path)

    return saved_paths


def build_pipeline(
    llm_model: str,
    vision_retriever: str,
    top_k: int,
    force_reindex: bool,
    qa_prompt: str,
    qwen_server_url: str | None = None,
    qwen_model_name: str | None = None,
    doubao_api_key: str | None = None,
    openai_api_key: str | None = None,
) -> PolicyRetrievalPipeline:
    """根据前端配置构建 PolicyRetrievalPipeline 实例。"""
    api_keys = {}
    if doubao_api_key:
        api_keys["doubao"] = doubao_api_key
    if openai_api_key:
        api_keys["openai"] = openai_api_key

    pipeline = PolicyRetrievalPipeline(
        data_dir="./policy_data",
        output_dir="./policy_outputs",
        llm_model=llm_model,
        vision_retriever=vision_retriever,
        api_keys=api_keys,
        top_k=top_k,
        force_reindex=force_reindex,
        qa_prompt=qa_prompt,
        extra_config=None,
        qwen_server_url=qwen_server_url,
        qwen_model_name=qwen_model_name,
    )
    return pipeline


def inject_global_style() -> None:
    """注入全局样式，使界面更接近设计稿风格。"""
    st.markdown(
        """
        <style>
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        #MainMenu,
        footer {
            display: none;
        }

        .stApp {
            background-color: #eef2f5;
        }

        .block-container {
            max-width: 1240px !important;
            padding-top: 0px !important;
            padding-bottom: 22px !important;
        }

        .app-header {
            width: 100%;
            margin-top: 14px;
            background: linear-gradient(90deg, #0f8a74 0%, #0b7e6b 55%, #0f8a74 100%);
            border-radius: 10px;
            padding: 10px 14px;
            color: #ffffff;
            box-shadow: 0 6px 18px rgba(14, 29, 52, 0.18);
            display: flex;
            align-items: center;
            gap: 14px;
            box-sizing: border-box;
        }

        .app-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            min-width: 250px;
        }

        .app-emblem {
            width: 34px;
            height: 34px;
            border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, #f7d166 0%, #d18c1b 30%, #b51c1c 70%);
            border: 2px solid rgba(255, 255, 255, 0.75);
            box-sizing: border-box;
        }

        .app-title {
            font-size: 18px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }

        .app-search {
            flex: 1;
            background-color: rgba(255, 255, 255, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.25);
            border-radius: 999px;
            padding: 9px 14px;
            font-size: 13px;
            color: rgba(255, 255, 255, 0.92);
            box-sizing: border-box;
        }

        .app-actions {
            min-width: 116px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 10px;
        }

        .app-icon {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background-color: rgba(255, 255, 255, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
        }

        .toolbar {
            background-color: #ffffff;
            border: 1px solid #e6ecf2;
            border-radius: 10px;
            padding: 10px 12px;
            margin-top: 10px;
            box-shadow: 0 6px 14px rgba(15, 40, 73, 0.06);
        }

        .toolbar-current {
            font-size: 13px;
            color: #1b2b3a;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            padding-top: 6px;
        }

        .card {
            background-color: #ffffff;
            border: 1px solid #e6ecf2;
            border-radius: 10px;
            box-shadow: 0 6px 14px rgba(15, 40, 73, 0.06);
            margin-top: 10px;
            overflow: hidden;
        }

        .card-header {
            background: linear-gradient(180deg, #f7fafc 0%, #f2f6f9 100%);
            padding: 10px 12px;
            border-bottom: 1px solid #e6ecf2;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            box-sizing: border-box;
        }

        .card-title {
            font-size: 15px;
            font-weight: 700;
            color: #152434;
        }

        .card-body {
            padding: 12px;
        }

        .tag {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            color: #0b7e6b;
            background-color: rgba(15, 138, 116, 0.08);
            border: 1px solid rgba(15, 138, 116, 0.22);
            padding: 4px 8px;
            border-radius: 999px;
            white-space: nowrap;
        }

        .muted {
            color: #6a7a8a;
            font-size: 12px;
        }

        .one-line {
            font-size: 14px;
            font-weight: 700;
            color: #122235;
            line-height: 1.6;
        }

        .kv {
            display: flex;
            flex-wrap: wrap;
            gap: 10px 14px;
            margin-top: 10px;
        }
        .kv span {
            font-size: 12px;
            color: #556575;
        }

        .pill {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 999px;
            background-color: #f3f6fb;
            border: 1px solid #e2e8f0;
            font-size: 12px;
            color: #31475d;
        }

        .grid-3 {
            display: grid;
            grid-template-columns: 1fr 1fr 1.2fr;
            gap: 10px;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1.1fr 1fr;
            gap: 10px;
        }

        .panel {
            border: 1px solid #e6ecf2;
            border-radius: 10px;
            padding: 10px 10px;
            background: #fbfdff;
        }

        .panel-title {
            font-size: 13px;
            font-weight: 700;
            color: #152434;
            margin-bottom: 6px;
        }

        .ul {
            padding-left: 18px;
            margin: 0;
        }
        .ul li {
            margin: 6px 0;
            color: #23364a;
            font-size: 13px;
            line-height: 1.5;
        }

        .segbar {
            display: flex;
            gap: 4px;
            margin-top: 6px;
        }
        .seg {
            height: 8px;
            width: 18px;
            border-radius: 2px;
            background: #dfe7ee;
        }
        .seg.on {
            background: #0f8a74;
        }

        .avatar {
            width: 100%;
            height: 150px;
            border-radius: 10px;
            background: linear-gradient(180deg, #f2f6f9 0%, #e8eef6 100%);
            border: 1px solid #e2e8f0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #7b8ea2;
            font-size: 13px;
            margin-bottom: 10px;
        }

        div[data-testid="stButton"] > button {
            border-radius: 8px;
            border: 1px solid #dbe4ee;
            padding: 8px 12px;
            font-size: 13px;
            line-height: 1;
            background-color: #f7fafc;
            color: #173046;
        }

        div[data-testid="stButton"] > button[kind="primary"] {
            background-color: #0f8a74;
            border-color: #0f8a74;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_header() -> None:
    """渲染顶部导航条区域。"""
    st.markdown(
        """
        <div class="app-header">
          <div class="app-brand">
            <div class="app-emblem"></div>
            <div class="app-title">农业政策智能解读平台</div>
          </div>
          <div class="app-search">输入政策标题、文号、发文机关等内容进行检索</div>
          <div class="app-actions">
            <div class="app-icon">🔔</div>
            <div class="app-icon">👤</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _text_to_items(text: str, max_items: int = 6) -> List[str]:
    s = (text or "").strip()
    if not s:
        return []

    parts = [p.strip() for p in re.split(r"\r?\n+", s) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r"[。；;]\s*", s) if p.strip()]
    parts = [p.lstrip("-•* ").strip() for p in parts if p.strip()]
    return parts[:max_items]


def _pick_current_title(all_inputs: List[str]) -> str:
    if not all_inputs:
        return "未选择政策"
    first = all_inputs[0]
    try:
        return Path(first).name
    except Exception:
        return str(first)


def _escape(text: str) -> str:
    return html.escape(text or "")


def _one_liner(text: str, max_len: int = 120) -> str:
    s = (text or "").strip()
    if not s:
        return "暂无"

    first = re.split(r"[。\n]", s, maxsplit=1)[0].strip()
    if not first:
        first = s
    if len(first) > max_len:
        return first[: max_len - 1].rstrip() + "…"
    return first


def _extract_dates(text: str) -> List[str]:
    s = text or ""
    dates = re.findall(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", s)
    seen = set()
    out: List[str] = []
    for d in dates:
        if d not in seen:
            out.append(d)
            seen.add(d)
    return out


def _card_open(title: str, right_html: str = "") -> None:
    st.markdown(
        f"""
        <div class="card">
          <div class="card-header">
            <div class="card-title">{_escape(title)}</div>
            {right_html}
          </div>
          <div class="card-body">
        """,
        unsafe_allow_html=True,
    )


def _card_close() -> None:
    st.markdown(
        """
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="农业政策智能解读平台", layout="wide")

    inject_global_style()
    render_top_header()

    if "settings_open" not in st.session_state:
        st.session_state["settings_open"] = False

    if "uploaded_paths" not in st.session_state:
        st.session_state["uploaded_paths"] = []
    if "saved_upload_names" not in st.session_state:
        st.session_state["saved_upload_names"] = []

    existing_files = scan_policy_files("./policy_data")

    def _sync_uploaded_paths() -> None:
        uploaded_files = st.session_state.get("uploaded_files", None)
        if not uploaded_files:
            return

        saved_names = set(st.session_state.get("saved_upload_names", []))
        new_files = []
        all_names = []
        for uf in uploaded_files:
            name = os.path.basename(getattr(uf, "name", ""))
            if not name:
                continue
            all_names.append(name)
            if name not in saved_names:
                new_files.append(uf)

        if new_files:
            new_paths = save_uploaded_files(new_files)
            for p in new_paths:
                if p not in st.session_state["uploaded_paths"]:
                    st.session_state["uploaded_paths"].append(p)

        st.session_state["saved_upload_names"] = sorted(saved_names.union(all_names))

    _sync_uploaded_paths()

    selected_files = st.session_state.get("selected_files", [])
    url_text = st.session_state.get("url_text", "")
    url_inputs = [line.strip() for line in (url_text or "").splitlines() if line.strip()]
    uploaded_paths = st.session_state.get("uploaded_paths", [])

    all_inputs: List[str] = []
    all_inputs.extend(selected_files)
    all_inputs.extend(uploaded_paths)
    all_inputs.extend(url_inputs)

    current_title = _pick_current_title(all_inputs)

    st.markdown("<div class='toolbar'>", unsafe_allow_html=True)
    tb_left, tb_center, tb_right = st.columns([2.6, 5.1, 1.1])
    with tb_left:
        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button("上传政策文件", use_container_width=True):
                st.session_state["settings_open"] = not st.session_state["settings_open"]
        with b2:
            run_button = st.button("提取政策信息", type="primary", use_container_width=True)
    with tb_center:
        st.markdown(
            f"<div class='toolbar-current'><b>当前解读：</b>{_escape(current_title)}</div>",
            unsafe_allow_html=True,
        )
    with tb_right:
        st.markdown("<span class='tag'>官方来源</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("上传政策文件 / 输入链接 / 参数配置", expanded=st.session_state["settings_open"]):
        left, right = st.columns([2.1, 1.4])

        with left:
            st.markdown("**选择已有政策文件（policy_data）**")
            st.multiselect(
                "从 policy_data 目录中选择已有文件 (Word / HTML / PDF)",
                options=existing_files,
                default=selected_files,
                key="selected_files",
            )

            st.file_uploader(
                "上传政策文件（保存到 policy_data/uploads）",
                type=[ext.lstrip(".") for ext in ALLOWED_EXTENSIONS],
                accept_multiple_files=True,
                key="uploaded_files",
            )

            st.markdown("**政策网页 URL（每行一个，可选）**")
            url_text = st.text_area(
                "",
                height=110,
                placeholder="https://www.moa.gov.cn/...",
                key="url_text",
                label_visibility="collapsed",
            )

        with right:
            llm_model = st.selectbox(
                "视觉 LLM 模型",
                options=["doubao", "gpt4", "qwen"],
                index=0,
                help="需与后端 visual_rag 配置一致。",
                key="llm_model",
            )

            vision_retriever = st.selectbox(
                "视觉检索模型",
                options=["colpali", "colqwen", "nemo"],
                index=0,
                key="vision_retriever",
            )

            top_k = st.slider("Top-K 检索页数", min_value=1, max_value=10, value=5, step=1, key="top_k")
            force_reindex = st.checkbox("强制重新构建索引 (force_reindex)", value=False, key="force_reindex")

            qa_prompt_default = "请基于给定政策文本，客观提取和归纳关键信息。请务必用中文回答问题。"
            qa_prompt = st.text_area("问答系统提示词 (prompt)", value=qa_prompt_default, height=100, key="qa_prompt")

            env_ark_key = os.getenv("ARK_API_KEY", "")
            doubao_api_key = st.text_input("Doubao ARK_API_KEY", value=env_ark_key, type="password", key="doubao_api_key")

            env_openai_key = os.getenv("OPENAI_API_KEY", "")
            openai_api_key = st.text_input("OpenAI API Key", value=env_openai_key, type="password", key="openai_api_key")

            env_qwen_url = os.getenv("QWEN_VL_SERVER_URL", "")
            qwen_server_url = st.text_input("Qwen-VL vLLM 服务地址", value=env_qwen_url, placeholder="例如：http://localhost:8001", key="qwen_server_url")
            qwen_model_name = st.text_input(
                "Qwen-VL 模型名称",
                value=os.getenv("QWEN_VL_MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct"),
                key="qwen_model_name",
            )

    selected_files = st.session_state.get("selected_files", [])
    url_text = st.session_state.get("url_text", "")
    url_inputs = [line.strip() for line in (url_text or "").splitlines() if line.strip()]
    uploaded_paths = st.session_state.get("uploaded_paths", [])

    all_inputs = []
    all_inputs.extend(selected_files)
    all_inputs.extend(uploaded_paths)
    all_inputs.extend(url_inputs)

    llm_model = st.session_state.get("llm_model", "doubao")
    vision_retriever = st.session_state.get("vision_retriever", "colpali")
    top_k = st.session_state.get("top_k", 5)
    force_reindex = st.session_state.get("force_reindex", False)
    qa_prompt = st.session_state.get("qa_prompt", "请基于给定政策文本，客观提取和归纳关键信息。请务必用中文回答问题。")
    doubao_api_key = st.session_state.get("doubao_api_key", os.getenv("ARK_API_KEY", ""))
    openai_api_key = st.session_state.get("openai_api_key", os.getenv("OPENAI_API_KEY", ""))
    qwen_server_url = st.session_state.get("qwen_server_url", os.getenv("QWEN_VL_SERVER_URL", ""))
    qwen_model_name = st.session_state.get("qwen_model_name", os.getenv("QWEN_VL_MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct"))

    if "policy_results" not in st.session_state:
        st.session_state["policy_results"] = None

    # ----------------- 运行 Pipeline -----------------
    if run_button:
        if not all_inputs:
            st.error("没有有效输入，请先选择文件或填写 URL。")
        else:
            # 基于选择的 LLM 模型做一些基本校验
            if llm_model == "doubao" and not doubao_api_key:
                st.error("已选择 doubao 模型，但 Doubao ARK_API_KEY 为空，请在右侧填写。")
                return
            if llm_model == "gpt4" and not openai_api_key:
                st.error("已选择 gpt4 模型，但 OpenAI API Key 为空，请在右侧填写。")
                return
            if llm_model == "qwen" and not qwen_server_url:
                st.error("已选择 qwen 模型，但 Qwen-VL vLLM 服务地址为空，请在右侧填写。")
                return

            try:
                with st.spinner("正在构建检索 Pipeline 并执行 7 维度问答，这可能需要较长时间，请稍候..."):
                    pipeline = build_pipeline(
                        llm_model=llm_model,
                        vision_retriever=vision_retriever,
                        top_k=top_k,
                        force_reindex=force_reindex,
                        qa_prompt=qa_prompt,
                        qwen_server_url=qwen_server_url or None,
                        qwen_model_name=qwen_model_name or None,
                        doubao_api_key=doubao_api_key or None,
                        openai_api_key=openai_api_key or None,
                    )

                    results = pipeline.retrieve_policy_info(all_inputs)
                    st.session_state["policy_results"] = results
            except Exception as e:
                st.error(f"运行检索 Pipeline 时发生错误：{e}")

    # ----------------- 结果展示：截图风格卡片布局 -----------------
    results = st.session_state.get("policy_results")

    main_left, main_right = st.columns([3.2, 1.25])

    with main_left:
        # 1) 政策要点总览
        right_html = "<span class='tag'>官方来源</span>"
        _card_open("【政策要点总览】", right_html=right_html)

        if results:
            conclusion = _one_liner(results.get("what", {}).get("answer", ""))
            bullet_text = (results.get("threshold", {}).get("answer", "") or "") + "\n" + (results.get("compliance", {}).get("answer", "") or "")
            bullets = _text_to_items(bullet_text, max_items=5)
            when_text = results.get("when", {}).get("answer", "")
            dates = _extract_dates(when_text)
        else:
            conclusion = "请先上传政策文件并点击“提取政策信息”"
            bullets = []
            dates = []

        st.markdown(
            f"<div class='one-line'>一句话结论：{_escape(conclusion)}</div>",
            unsafe_allow_html=True,
        )

        if bullets:
            items_html = "".join([f"<li>{_escape(x)}</li>" for x in bullets])
            st.markdown(f"<ul class='ul'>{items_html}</ul>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='muted'>暂无要点，提取后将展示关键摘要与注意事项。</div>", unsafe_allow_html=True)

        pub_date = dates[0] if dates else "-"
        deadline = dates[1] if len(dates) > 1 else "-"
        st.markdown(
            """
            <div class="kv">
              <span><b>基本信息：</b></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='kv'><span>政策来源：<span class='pill'>{_escape(current_title)}</span></span><span>发布时间：<span class='pill'>{_escape(pub_date)}</span></span><span>截止时间：<span class='pill'>{_escape(deadline)}</span></span></div>",
            unsafe_allow_html=True,
        )

        _card_close()

        # 2) 支持内容与申报规则
        _card_open("【支持内容与申报规则】")

        if results:
            who_items = _text_to_items(results.get("who", {}).get("answer", ""), max_items=6)
            ban_items = _text_to_items(results.get("compliance", {}).get("answer", ""), max_items=5)
            money_items = _text_to_items(
                (results.get("how_much", {}).get("answer", "") or "") + "\n" + (results.get("what", {}).get("answer", "") or ""),
                max_items=6,
            )
            material_items = _text_to_items(results.get("how", {}).get("answer", ""), max_items=10)
            threshold_items = _text_to_items(results.get("threshold", {}).get("answer", ""), max_items=4)
        else:
            who_items, ban_items, money_items, material_items, threshold_items = [], [], [], [], []

        who_html = "".join([f"<li>{_escape(x)}</li>" for x in (who_items or ["（提取后展示支持对象）"])])
        ban_html = "".join([f"<li>{_escape(x)}</li>" for x in (ban_items or ["（提取后展示不适用情形）"])])
        money_html = "".join([f"<li>{_escape(x)}</li>" for x in (money_items or ["（提取后展示扶持方式与资金规则）"])])
        threshold_pills = " ".join([f"<span class='pill'>{_escape(x)}</span>" for x in threshold_items])

        st.markdown(
            f"""
            <div class="grid-3">
              <div class="panel">
                <div class="panel-title">支持对象</div>
                <ul class="ul">{who_html}</ul>
              </div>
              <div class="panel">
                <div class="panel-title">不适用 / 负面清单</div>
                <ul class="ul">{ban_html}</ul>
              </div>
              <div class="panel">
                <div class="panel-title">扶持方式与资金规则</div>
                <ul class="ul">{money_html}</ul>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='kv'><span><b>核心申报条件：</b></span>" + (threshold_pills or "<span class='muted'>（提取后展示）</span>") + "</div>", unsafe_allow_html=True)

        with st.expander("申报材料清单（点击展开）", expanded=False):
            if material_items:
                for it in material_items:
                    st.markdown(f"- {_escape(it)}", unsafe_allow_html=True)
            else:
                st.caption("提取后将展示申报流程与所需材料。")

        _card_close()

        # 3) 影响解读与行动建议
        _card_open("【影响解读与行动建议】")

        if results:
            impact_items = _text_to_items(results.get("what", {}).get("answer", ""), max_items=3)
            action_items = _text_to_items(results.get("how", {}).get("answer", ""), max_items=3)
            when_text = results.get("when", {}).get("answer", "")
            dates = _extract_dates(when_text)
            start_date = dates[0] if dates else "-"
            end_date = dates[1] if len(dates) > 1 else "-"
        else:
            impact_items, action_items = [], []
            start_date, end_date = "-", "-"

        impact_html = "".join([f"<li>{_escape(x)}</li>" for x in (impact_items or ["（提取后展示政策影响与适用范围）"])])
        action_html = "".join([f"<li>{_escape(x)}</li>" for x in (action_items or ["（提取后展示可执行行动建议）"])])

        st.markdown(
            f"""
            <div class="grid-2">
              <div class="panel">
                <div class="panel-title">政策影响</div>
                <div class="muted">对财政支出、产业链、申报成本等的影响（示意）</div>
                <div style="margin-top:8px">
                  <div class="muted">对财政支出：</div>
                  <div class="segbar"><div class="seg on"></div><div class="seg on"></div><div class="seg on"></div><div class="seg"></div><div class="seg"></div></div>
                  <div class="muted" style="margin-top:8px">对产业链：</div>
                  <div class="segbar"><div class="seg on"></div><div class="seg on"></div><div class="seg on"></div><div class="seg on"></div><div class="seg"></div></div>
                </div>
                <div class="kv"><span>申报窗口：<span class="pill">{_escape(start_date)}</span></span><span>截止：<span class="pill">{_escape(end_date)}</span></span></div>
                <ul class="ul">{impact_html}</ul>
              </div>
              <div class="panel">
                <div class="panel-title">AI 行动建议</div>
                <div class="muted">基于政策要点与申报规则生成的行动建议（示意）</div>
                <ul class="ul">{action_html}</ul>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        _card_close()

        if results:
            with st.expander("查看原始结果 JSON"):
                st.json(results)

    with main_right:
        # 数字人播报
        _card_open("数字人播报")
        st.markdown("<div class='avatar'>数字人形象占位</div>", unsafe_allow_html=True)
        st.button("▶ 播放解读", use_container_width=True)
        st.radio("", options=["1分钟快读", "3分钟深度解读"], horizontal=True, label_visibility="collapsed")
        _card_close()

        # 解读目录
        _card_open("解读目录")
        st.markdown(
            """
            <ul class="ul">
              <li>政策要点总览</li>
              <li>支持内容与申报规则</li>
              <li>影响解读与行动建议</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )
        _card_close()

        # 关联政策
        _card_open("关联政策")
        if all_inputs:
            show_items = all_inputs[:5]
            items_html = "".join([f"<li>{_escape(str(x))}</li>" for x in show_items])
            st.markdown(f"<ul class='ul'>{items_html}</ul>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='muted'>上传或选择多个政策文件后，将在此展示关联政策。</div>", unsafe_allow_html=True)
        _card_close()


if __name__ == "__main__":
    main()
