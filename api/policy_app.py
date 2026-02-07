import os
os.environ["HF_ENDPOINT"] = "http://hf-mirror.com"
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

from retrieval_pipe import PolicyRetrievalPipeline, POLICY_QUESTIONS


# 预加载 .env 中的环境变量（例如 ARK_API_KEY、OPENAI_API_KEY、QWEN_VL_SERVER_URL 等）
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


def main() -> None:
    st.set_page_config(page_title="政策文件视觉RAG检索助手", layout="wide")

    st.title("政策文件视觉 RAG 检索助手")
    st.markdown(
        """
        该界面支持对 Word / HTML / PDF / 政策网页链接 进行统一处理，并从 7 个维度抽取政策关键信息。
        """
    )

    # ----------------- 侧边栏：参数配置 -----------------
    with st.sidebar:
        st.header("参数配置")

        llm_model = st.selectbox(
            "选择视觉 LLM 模型",
            options=["doubao", "gpt4", "qwen"],
            index=0,
            help="需与后端 visual_rag 配置一致。",
        )

        vision_retriever = st.selectbox(
            "选择视觉检索模型",
            options=["colpali", "colqwen", "nemo"],
            index=0,
        )

        top_k = st.slider("Top-K 检索页数", min_value=1, max_value=10, value=5, step=1)
        force_reindex = st.checkbox("强制重新构建索引 (force_reindex)", value=False)

        qa_prompt_default = "请基于给定政策文本，客观提取和归纳关键信息。请务必用中文回答问题。"
        qa_prompt = st.text_area(
            "问答系统提示词 (prompt)",
            value=qa_prompt_default,
            height=120,
        )

        st.subheader("API / 服务配置")

        # Doubao (火山方舟) API
        env_ark_key = os.getenv("ARK_API_KEY", "")
        doubao_api_key = st.text_input(
            "Doubao ARK_API_KEY",
            value=env_ark_key,
            type="password",
            help="如果 .env 中已配置 ARK_API_KEY，这里会自动读取，可按需修改。",
        )

        # OpenAI API（仅当选择 GPT-4 时实际使用）
        env_openai_key = os.getenv("OPENAI_API_KEY", "")
        openai_api_key = st.text_input(
            "OpenAI API Key",
            value=env_openai_key,
            type="password",
        )

        # Qwen-VL vLLM 服务
        env_qwen_url = os.getenv("QWEN_VL_SERVER_URL", "")
        qwen_server_url = st.text_input(
            "Qwen-VL vLLM 服务地址",
            value=env_qwen_url,
            placeholder="例如：http://localhost:8001",
        )
        qwen_model_name = st.text_input(
            "Qwen-VL 模型名称",
            value=os.getenv("QWEN_VL_MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct"),
        )

        st.markdown("---")
        st.caption("注意：请确保后端模型与 GPU 环境已正确部署。")

    # ----------------- 主区域：文件与输入 -----------------

    st.header("选择/上传政策文件")

    existing_files = scan_policy_files("./policy_data")
    selected_files = st.multiselect(
        "从 policy_data 目录中选择已有文件 (Word / HTML / PDF)",
        options=existing_files,
        default=[],
    )

    uploaded_files = st.file_uploader(
        "上传新的政策文件 (将被保存到 policy_data/uploads)",
        type=[ext.lstrip(".") for ext in ALLOWED_EXTENSIONS],
        accept_multiple_files=True,
    )

    uploaded_paths = save_uploaded_files(uploaded_files) if uploaded_files else []

    st.subheader("输入政策网页 URL (每行一个，可选)")
    url_text = st.text_area(
        "可填写农业农村部等官网的政策链接，系统将自动转为 PDF 进行处理。",
        height=100,
        placeholder="https://www.moa.gov.cn/...",
    )
    url_inputs = [line.strip() for line in url_text.splitlines() if line.strip()]

    all_inputs: List[str] = []
    all_inputs.extend(selected_files)
    all_inputs.extend(uploaded_paths)
    all_inputs.extend(url_inputs)

    if all_inputs:
        st.info("本次将处理以下输入：")
        st.write(all_inputs)
    else:
        st.warning("请至少选择一个本地文件或输入一个 URL。")

    # ----------------- 运行 Pipeline -----------------

    run_button = st.button("运行 7 维度政策信息抽取", type="primary")

    if "policy_results" not in st.session_state:
        st.session_state["policy_results"] = None

    if run_button:
        if not all_inputs:
            st.error("没有有效输入，请先选择文件或填写 URL。")
        else:
            # 基于选择的 LLM 模型做一些基本校验
            if llm_model == "doubao" and not doubao_api_key:
                st.error("已选择 doubao 模型，但 Doubao ARK_API_KEY 为空，请在侧边栏中填写。")
                return
            if llm_model == "gpt4" and not openai_api_key:
                st.error("已选择 gpt4 模型，但 OpenAI API Key 为空，请在侧边栏中填写。")
                return
            if llm_model == "qwen" and not qwen_server_url:
                st.error("已选择 qwen 模型，但 Qwen-VL vLLM 服务地址为空，请在侧边栏中填写。")
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

    # ----------------- 结果展示 -----------------

    results = st.session_state.get("policy_results")
    if results:
        st.header("政策 7 维度抽取结果")

        # 使用标签页分别展示七个维度
        tabs = st.tabs(list(POLICY_QUESTIONS.keys()))
        for key, tab in zip(POLICY_QUESTIONS.keys(), tabs):
            info = results.get(key, {})
            with tab:
                st.subheader(f"[{key}] {info.get('question', '')}")
                st.markdown("**答案：**")
                st.write(info.get("answer", ""))
                st.markdown("**分析 (Chain of Thought)：**")
                st.write(info.get("analysis", ""))

        st.markdown("---")
        with st.expander("查看原始结果 JSON"):
            st.json(results)


if __name__ == "__main__":
    main()
