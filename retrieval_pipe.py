import os
from pathlib import Path
from typing import Dict, List, Optional

from utils import PDFConverter
from visual_rag import VisualRAGPipeline


POLICY_QUESTIONS: Dict[str, str] = {
    "who": "根据政策文件，申报主体是谁？谁可以申报？请概括主体类型和适用范围。",
    "what": "根据政策文件，资金主要用于哪些具体方向或环节？",
    "how_much": "根据政策文件，补贴标准和资金上限是多少？",
    "threshold": "根据政策文件，申报门槛或硬性指标有哪些？例如规模要求、资质条件等。",
    "compliance": "根据政策文件，合规性要求有哪些？是否有禁止情形或红线条款？",
    "when": "根据政策文件，本次申报的时间节点和截止日期是什么？是否有分批次安排？",
    "how": "根据政策文件，申报流程和需要提交的材料有哪些？请按流程步骤进行梳理。",
}


class PolicyRetrievalPipeline:
    """统一完成：Word/网页 -> PDF -> VisDoMRAG 检索 -> 7 个维度问答。"""

    def __init__(
        self,
        data_dir: str = "./policy_data",
        output_dir: str = "./policy_outputs",
        llm_model: str = "doubao",
        vision_retriever: str = "colpali",
        api_keys: Optional[Dict[str, str]] = None,
        top_k: int = 5,
        force_reindex: bool = False,
        qa_prompt: str = "请基于给定政策文本，客观提取和归纳关键信息。",
        extra_config: Optional[Dict] = None,
        qwen_server_url: Optional[str] = None,
        qwen_model_name: Optional[str] = None,
    ) -> None:
        self.data_dir = os.path.abspath(data_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.docs_dir = os.path.join(self.data_dir, "docs")
        os.makedirs(self.docs_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.converter = PDFConverter()
        self._pdf_entries: List[str] = []  # 相对 data_dir 的路径，例如 "docs/file.pdf"

        base_config = {
            "data_dir": self.data_dir,
            "output_dir": self.output_dir,
            "llm_model": llm_model,
            "vision_retriever": vision_retriever,
            "top_k": top_k,
            "api_keys": api_keys or {},
            "force_reindex": force_reindex,
            "qa_prompt": qa_prompt,
            "pdf_files": [],
            "qwen_server_url": qwen_server_url,
            "qwen_model_name": qwen_model_name,
        }
        if extra_config:
            base_config.update(extra_config)

        self._config = base_config
        self._visdom: Optional[VisualRAGPipeline] = None

    # ---------------------- 文档准备：统一转 PDF ----------------------

    def _slugify(self, text: str) -> str:
        """将 URL 或任意字符串转为适合作为文件名的 slug。"""
        import re

        text = text.strip().lower()
        text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        if not text:
            text = "doc"
        # 防止过长
        return text[:80]

    def _ensure_visdom(self) -> None:
        if self._visdom is None:
            if not self._pdf_entries:
                raise ValueError("尚未添加任何 PDF 文档")
            self._config["pdf_files"] = self._pdf_entries
            self._visdom = VisualRAGPipeline(self._config)

    def add_inputs(self, inputs: List[str]) -> List[str]:
        """接收文件路径或 URL，统一转为 PDF 并返回生成的 PDF 绝对路径列表。"""
        generated_pdfs: List[str] = []

        for item in inputs:
            if os.path.exists(item):
                ext = Path(item).suffix.lower()
                if ext in PDFConverter.WORD_EXTENSIONS or ext in PDFConverter.HTML_EXTENSIONS:
                    pdf_name = Path(item).stem + ".pdf"
                    pdf_rel = os.path.join("docs", pdf_name)
                    pdf_abs = os.path.join(self.docs_dir, pdf_name)
                    ok = self.converter.convert(item, pdf_abs)
                    if not ok:
                        raise RuntimeError(f"转换为 PDF 失败: {item}")
                elif ext == ".pdf":
                    pdf_name = Path(item).name
                    pdf_rel = os.path.join("docs", pdf_name)
                    pdf_abs = os.path.join(self.docs_dir, pdf_name)
                    if os.path.abspath(item) != os.path.abspath(pdf_abs):
                        import shutil

                        os.makedirs(self.docs_dir, exist_ok=True)
                        shutil.copy2(item, pdf_abs)
                else:
                    raise ValueError(f"不支持的文件类型: {item}")

                if pdf_rel not in self._pdf_entries:
                    self._pdf_entries.append(pdf_rel)
                generated_pdfs.append(pdf_abs)

            else:
                if item.startswith("http://") or item.startswith("https://"):
                    slug = self._slugify(item)
                    pdf_name = f"{slug}.pdf"
                    pdf_rel = os.path.join("docs", pdf_name)
                    pdf_abs = os.path.join(self.docs_dir, pdf_name)
                    ok = self.converter.url_to_pdf(item, pdf_abs)
                    if not ok:
                        raise RuntimeError(f"URL 转 PDF 失败: {item}")
                    if pdf_rel not in self._pdf_entries:
                        self._pdf_entries.append(pdf_rel)
                    generated_pdfs.append(pdf_abs)
                else:
                    raise FileNotFoundError(f"路径不存在且不是 URL: {item}")

        return generated_pdfs

    # ---------------------- 检索与问答 ----------------------

    def retrieve_policy_info(self, inputs: List[str]) -> Dict[str, Dict[str, str]]:
        """主入口：给定若干 Word/HTML/PDF/URL，返回 7 个维度的结构化答案。

        返回结构：
        {
            "who": {"question": "...", "answer": "...", "analysis": "..."},
            ...
        }
        """
        self.add_inputs(inputs)
        self._ensure_visdom()

        results: Dict[str, Dict[str, str]] = {}
        for key, question in POLICY_QUESTIONS.items():
            response = self._visdom.answer_question(question)
            if response is None:
                results[key] = {"question": question, "answer": "", "analysis": ""}
                continue
            results[key] = {
                "question": question,
                "answer": response.get("answer", ""),
                "analysis": response.get("analysis", ""),
            }
        return results


__all__ = ["PolicyRetrievalPipeline", "POLICY_QUESTIONS"]
