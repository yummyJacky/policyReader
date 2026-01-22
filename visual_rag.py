import os
import base64
import json
import logging
import traceback
from io import BytesIO
import time
import re

import numpy as np
import torch
from pdf2image import convert_from_path
from tqdm import tqdm
from transformers import AutoModel
from openai import OpenAI
import requests


logger = logging.getLogger("VisDoMRAG")


class VisualRAGEngine:
    def __init__(
        self,
        *,
        data_dir: str,
        output_dir: str,
        llm_model: str,
        vision_retriever: str,
        top_k: int,
        force_reindex: bool,
        qa_prompt: str,
        pdf_files,
        client=None,
        llm=None,
        process_vision_info=None,
        qwen_server_url=None,
        qwen_model_name=None,
    ):
        """Engine responsible for visual retrieval and visual QA.

        All necessary configuration is passed explicitly instead of via a parent object.
        """

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.llm_model = llm_model
        self.vision_retriever = vision_retriever
        self.top_k = top_k
        self.force_reindex = force_reindex
        self.qa_prompt = qa_prompt
        self.pdf_files = pdf_files

        # Optional LLM-related handles
        if client is not None:
            self.client = client
        if llm is not None:
            self.llm = llm
        if process_vision_info is not None:
            self.process_vision_info = process_vision_info

        # Optional Qwen-VL vLLM HTTP configuration
        if qwen_server_url is not None:
            self.qwen_server_url = qwen_server_url
        if qwen_model_name is not None:
            self.qwen_model_name = qwen_model_name

        # Visual retrieval resources (supporting ColPali/ColQwen and Nvidia NemoRetriever)
        self.vision_retrieval_file = (
            f"{self.data_dir}/retrieval/retrieval_{self.vision_retriever}.csv"
        )

        # Model / processor handles (lazy-loaded on first use)
        self.vision_model = None
        self.vision_processor = None

        # Interactive visual index state
        self._visual_index_built = False
        self._page_embeddings = None  # Tensor of shape [n_pages, n_tokens, dim]
        self._page_ids = []
        self._page_info = {}

    def _get_config_pdf_paths(self):
        """Resolve configured PDF entries to absolute paths under data_dir.

        This mirrors the behavior previously implemented in the higher-level
        pipeline, but is now self-contained inside the engine.
        """
        paths = {}
        for entry in self.pdf_files:
            if os.path.isabs(entry):
                pdf_path = entry
            else:
                if "/" in entry:
                    pdf_path = os.path.join(self.data_dir, entry)
                else:
                    pdf_path = os.path.join(self.data_dir, entry)
            if not os.path.exists(pdf_path) and not entry.endswith(".pdf"):
                alt_entry = f"{entry}.pdf"
                if os.path.isabs(alt_entry):
                    alt_path = alt_entry
                else:
                    if "/" in alt_entry:
                        alt_path = os.path.join(self.data_dir, alt_entry)
                    else:
                        alt_path = os.path.join(self.data_dir, alt_entry)
                if os.path.exists(alt_path):
                    pdf_path = alt_path
            if not os.path.exists(pdf_path):
                logger.warning(f"PDF file not found: {pdf_path}")
                continue
            base = os.path.basename(pdf_path)
            doc_id = os.path.splitext(base)[0]
            if doc_id not in paths:
                paths[doc_id] = pdf_path
        return paths

    def _ensure_vision_model(self):
        """Lazily load visual retrieval model/processor based on vision_retriever."""
        if self.vision_model is not None:
            return

        # Keep original ColPali / ColQwen flows, and add Nemo as an alternative
        if self.vision_retriever in ["colpali", "colqwen"]:
            try:
                if self.vision_retriever == "colpali":
                    from colpali_engine.models import ColPali, ColPaliProcessor

                    logger.info("Loading ColPali model for visual indexing")
                    self.vision_model = ColPali.from_pretrained(
                        "vidore/colpali-v1.2",
                        torch_dtype=torch.bfloat16,
                        device_map="cuda",
                    ).eval()
                    self.vision_processor = ColPaliProcessor.from_pretrained(
                        "vidore/colpali-v1.2"
                    )
                else:  # colqwen
                    from colpali_engine.models import ColQwen2, ColQwen2Processor

                    logger.info("Loading ColQwen model for visual indexing")
                    self.vision_model = ColQwen2.from_pretrained(
                        "vidore/colqwen2-v1.0",
                        torch_dtype=torch.bfloat16,
                        device_map="cuda",
                    ).eval()
                    self.vision_processor = ColQwen2Processor.from_pretrained(
                        "vidore/colqwen2-v0.1"
                    )
            except ImportError:
                raise ImportError(
                    "ColPali/ColQwen models not found. Please install colpali_engine."
                )
        elif self.vision_retriever in ["nemo", "nvidia", "nemo_retriever"]:
            try:
                logger.info(
                    "Loading Nvidia NemoRetriever model 'nvidia/llama-nemoretriever-colembed-1b-v1' for visual indexing"
                )
                self.vision_model = AutoModel.from_pretrained(
                    "nvidia/llama-nemoretriever-colembed-1b-v1",
                    device_map="cuda",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    revision="1f0fdea7f5b19532a750be109b19072d719b8177",
                ).eval()
            except ImportError as e:
                raise ImportError(
                    f"Failed to import transformers for Nvidia NemoRetriever: {e}. Please install 'transformers'."
                )
            except Exception as e:
                logger.error("Error loading Nvidia NemoRetriever model: %s", str(e))
                raise
        else:
            raise ValueError(f"Unsupported visual retriever: {self.vision_retriever}")

    def _build_interactive_visual_index(self):
        # Ensure visual model is ready for interactive index as well
        self._ensure_vision_model()
        if self._visual_index_built:
            return
        if not self.pdf_files:
            logger.warning("pdf_files not provided for interactive visual index")
            self._visual_index_built = False
            return
        pdf_paths = self._get_config_pdf_paths()
        if not pdf_paths:
            logger.warning("No valid PDF files for interactive visual index")
            self._visual_index_built = False
            return
        # ColPali / ColQwen interactive index
        if self.vision_retriever in ["colpali", "colqwen"]:
            page_embeddings = {}
            page_info = {}
            for doc_id, pdf_path in tqdm(
                pdf_paths.items(), desc="Processing PDFs for interactive visual index"
            ):
                try:
                    pages = convert_from_path(pdf_path)
                except Exception as e:
                    logger.error(
                        f"Error converting PDF {pdf_path} to images: {str(e)}"
                    )
                    traceback.print_exc()
                    continue
                for page_idx, page_img in enumerate(pages):
                    page_id = f"{doc_id}_{page_idx}"
                    try:
                        processed_image = self.vision_processor.process_images([page_img])
                        processed_image = {
                            k: v.to(self.vision_model.device)
                            for k, v in processed_image.items()
                        }
                        with torch.no_grad():
                            embedding = self.vision_model(**processed_image)
                        page_embeddings[page_id] = embedding.cpu()
                        page_info[page_id] = {
                            "doc_id": doc_id,
                            "page_idx": page_idx,
                            "pdf_path": pdf_path,
                        }
                    except Exception as e:
                        logger.error(
                            f"Error processing page {page_idx} of PDF {pdf_path}: {str(e)}"
                        )
                        traceback.print_exc()
                        continue
            self._page_embeddings = page_embeddings
            self._page_info = page_info
            self._page_ids = list(page_embeddings.keys())
            self._visual_index_built = True

        # Nemo interactive index
        elif self.vision_retriever in ["nemo", "nvidia", "nemo_retriever"]:
            page_embeddings_list = []
            page_ids = []
            page_info = {}
            for doc_id, pdf_path in tqdm(
                pdf_paths.items(), desc="Processing PDFs for interactive visual index"
            ):
                try:
                    pages = convert_from_path(pdf_path)
                except Exception as e:
                    logger.error(
                        f"Error converting PDF {pdf_path} to images: {str(e)}"
                    )
                    traceback.print_exc()
                    continue

                if not pages:
                    continue

                try:
                    with torch.no_grad():
                        passage_embeddings = self.vision_model.forward_passages(
                            pages, batch_size=8
                        )
                    passage_embeddings = passage_embeddings.cpu()
                except Exception as e:
                    logger.error(
                        "Error encoding pages for interactive visual index with NemoRetriever: %s",
                        str(e),
                    )
                    traceback.print_exc()
                    continue

                for page_idx, page_img in enumerate(pages):
                    page_id = f"{doc_id}_{page_idx}"
                    page_ids.append(page_id)
                    page_embeddings_list.append(passage_embeddings[page_idx])
                    page_info[page_id] = {
                        "doc_id": doc_id,
                        "page_idx": page_idx,
                        "pdf_path": pdf_path,
                    }

            if not page_embeddings_list:
                logger.warning("No visual embeddings available for interactive index")
                self._visual_index_built = False
                return

            self._page_embeddings = torch.stack(page_embeddings_list, dim=0)
            self._page_ids = page_ids
            self._page_info = page_info
            self._visual_index_built = True
        else:
            raise ValueError(f"Unsupported visual retriever: {self.vision_retriever}")

    def retrieve_visual_contexts_for_question(self, query):
        try:
            self._build_interactive_visual_index()
            if not self._visual_index_built:
                return []
            # ColPali / ColQwen interactive retrieval
            if self.vision_retriever in ["colpali", "colqwen"]:
                if not self._page_embeddings:
                    return []
                page_ids = list(self._page_embeddings.keys())
                if not page_ids:
                    logger.warning(
                        "No visual embeddings available for interactive retrieval"
                    )
                    return []
                try:
                    processed_query = self.vision_processor.process_queries([query])
                    processed_query = {
                        k: v.to(self.vision_model.device)
                        for k, v in processed_query.items()
                    }
                    with torch.no_grad():
                        query_emb = self.vision_model(**processed_query)
                except Exception as e:
                    logger.error(
                        "Error generating query embedding for interactive visual retrieval: %s",
                        str(e),
                    )
                    traceback.print_exc()
                    return []
                ds = torch.cat(
                    [self._page_embeddings[pid] for pid in page_ids], dim=0
                )
                if len(ds) == 0:
                    logger.warning("No visual embeddings to score")
                    return []
                scores = self.vision_processor.score_multi_vector(query_emb, ds)
                scores = scores.flatten().numpy()
                top_indices = np.argsort(-scores)[: self.top_k]
                selected_page_ids = np.array(page_ids)[top_indices]
            # Nemo interactive retrieval
            elif self.vision_retriever in ["nemo", "nvidia", "nemo_retriever"]:
                if self._page_embeddings is None or not self._page_ids:
                    return []
                try:
                    with torch.no_grad():
                        query_emb = self.vision_model.forward_queries(
                            [query], batch_size=1
                        )
                    query_emb = query_emb.cpu()
                except Exception as e:
                    logger.error(
                        "Error generating query embedding for interactive visual retrieval: %s",
                        str(e),
                    )
                    traceback.print_exc()
                    return []

                try:
                    with torch.no_grad():
                        scores = self.vision_model.get_scores(
                            query_emb, self._page_embeddings
                        )
                    query_scores = scores[0].cpu().numpy()
                except Exception as e:
                    logger.error(
                        "Error scoring interactive visual retrieval with NemoRetriever: %s",
                        str(e),
                    )
                    traceback.print_exc()
                    return []

                if len(query_scores) == 0:
                    logger.warning("No visual embeddings to score")
                    return []

                top_indices = np.argsort(-query_scores)[: self.top_k]
                selected_page_ids = [
                    self._page_ids[idx]
                    for idx in top_indices
                    if 0 <= idx < len(self._page_ids)
                ]
            else:
                raise ValueError(f"Unsupported visual retriever: {self.vision_retriever}")

            pages = []
            for page_id in selected_page_ids:
                info = self._page_info.get(page_id)
                if not info:
                    continue
                pdf_path = info["pdf_path"]
                page_idx = info["page_idx"]
                try:
                    pdf_images = convert_from_path(pdf_path)
                    if page_idx >= len(pdf_images):
                        logger.warning(f"Page {page_idx} out of range for {pdf_path}")
                        continue
                    image = pdf_images[page_idx]
                    pages.append(
                        {
                            "image": image,
                            "document_id": page_id,
                            "page_number": page_idx,
                        }
                    )
                except Exception as e:
                    logger.error(f"Error loading PDF page for {page_id}: {str(e)}")
                    traceback.print_exc()
                    continue
            logger.info(
                "Retrieved %d visual contexts for interactive query", len(pages)
            )
            return pages
        except Exception as e:
            logger.error(
                "Error retrieving visual contexts for interactive query: %s", str(e)
            )
            traceback.print_exc()
            return []

    def encode_image(self, pil_image):
        """Encode a PIL image to base64 string."""
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def generate_visual_response(self, query, visual_contexts):
        """Generate a response based on visual contexts."""
        try:
            # Extract just the images
            images = [ctx["image"] for ctx in visual_contexts]

            # Create prompt
            prompt_template = f"""
            You are tasked with answering a question based on the relevant pages of a PDF document. Provide your response in the following format:
            ## Evidence:

            ## Chain of Thought:

            ## Answer:

            ___
            Instructions:

            1. Evidence Curation: Extract relevant elements (such as paragraphs, tables, figures, charts) from the provided pages and populate them in the "Evidence" section. For each element, include the type, content, and a brief explanation of its relevance.

            2. Chain of Thought: In the "Chain of Thought" section, list out each logical step you take to derive the answer, referencing the evidence where applicable. You should perform computations if you need to to get to the answer. 

            3. Answer: {self.qa_prompt}
            ___
            Question: {query}
            """
            base64_images = [self.encode_image(img) for img in images]

            if self.llm_model == "gpt4":
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            }
                            for base64_image in base64_images
                        ]
                        + [
                            {"type": "text", "text": prompt_template},
                        ],
                    }
                ]

                response = self.client.chat.completions.create(
                    model="chatgpt-4o-latest",
                    messages=messages,
                    max_tokens=3000,
                    temperature=0.7,
                )

                return response.choices[0].message.content

            elif self.llm_model == "doubao":
                response = self.llm.responses.create(
                    model="doubao-seed-1-6-flash-250828",
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                                }
                                for base64_image in base64_images
                            ]
                            + [
                                {
                                    "type": "input_text",
                                    "text": prompt_template,
                                }
                            ],
                        }
                    ],
                )
                return response.output[1].content[0].text
            elif self.llm_model == "qwen":
                # Use an external vLLM server exposing an OpenAI-compatible
                # /v1/chat/completions endpoint, similar to QwenVLCaptioner.
                server_url = getattr(self, "qwen_server_url", None)
                model_name = getattr(self, "qwen_model_name", None) or "Qwen/Qwen2.5-VL-7B-Instruct"

                if not server_url:
                    logger.warning("qwen_server_url not configured; cannot call Qwen-VL vLLM server")
                    return "Qwen-VL server not configured."

                s = server_url.strip()
                if not (s.startswith("http://") or s.startswith("https://")):
                    s = "http://" + s
                base = s.rstrip("/")
                if not base.endswith("/v1"):
                    base = f"{base}/v1"
                url = f"{base}/chat/completions"

                data_urls = [f"data:image/jpeg;base64,{b64}" for b64 in base64_images]

                payload = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": u}}
                                for u in data_urls
                            ]
                            + [
                                {
                                    "type": "text",
                                    "text": prompt_template,
                                }
                            ],
                        }
                    ],
                    "max_tokens": 2048,
                    "temperature": 0.1,
                }

                headers = {"Content-Type": "application/json"}

                response = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()

                choices = data.get("choices") or []
                if not choices:
                    return ""
                message = choices[0].get("message", {})
                content = message.get("content")

                if isinstance(content, str):
                    generated = content
                else:
                    parts = []
                    for block in content or []:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(block.get("text", ""))
                    generated = "".join(parts)

                return (generated or "").strip()

        except Exception as e:
            logger.error(f"Error generating visual response: {str(e)}")
            traceback.print_exc()
            return "Error generating response from visual contexts."


class VisualRAGPipeline:
    """Lightweight visual-only pipeline that wraps VisualRAGEngine.

    This is a simplified version of the original VisDoMRAG, keeping only
    configuration, LLM initialization and interactive visual QA.
    """

    def __init__(self, config):
        self.config = config
        self.data_dir = config["data_dir"]
        self.output_dir = config["output_dir"]
        self.llm_model = config["llm_model"]
        self.vision_retriever = config["vision_retriever"]
        self.top_k = config.get("top_k", 5)
        self.api_keys = config.get("api_keys", {})
        self.force_reindex = config.get("force_reindex", False)
        self.qa_prompt = config.get(
            "qa_prompt",
            "Answer the question objectively based on the context provided.",
        )
        self.pdf_files = config.get("pdf_files", [])

        # Create retrieval directory
        os.makedirs(f"{self.data_dir}/retrieval", exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Per-run identifiers and log file paths for timing & outputs
        self.run_id = time.strftime("%Y%m%d_%H%M%S")
        self.timing_log_path = os.path.join(
            self.data_dir,
            "retrieval",
            f"visrag_timing_{self.run_id}.jsonl",
        )
        self.output_log_path = os.path.join(
            self.output_dir,
            f"visrag_outputs_{self.run_id}.jsonl",
        )

        # Initialize LLM backend
        self._initialize_llm()

        # Visual engine (fully configured via explicit parameters)
        self.visual_engine = VisualRAGEngine(
            data_dir=self.data_dir,
            output_dir=self.output_dir,
            llm_model=self.llm_model,
            vision_retriever=self.vision_retriever,
            top_k=self.top_k,
            force_reindex=self.force_reindex,
            qa_prompt=self.qa_prompt,
            pdf_files=self.pdf_files,
            client=getattr(self, "client", None),
            llm=getattr(self, "llm", None),
            process_vision_info=getattr(self, "process_vision_info", None),
            qwen_server_url=getattr(self, "qwen_server_url", None),
            qwen_model_name=getattr(self, "qwen_model_name", None),
        )

    def _initialize_llm(self):
        """Initialize visual-capable LLM based on llm_model setting."""
        if self.llm_model == "doubao":
            if not self.api_keys.get("doubao"):
                raise ValueError("Doubao API key is required")
            self.llm = OpenAI(
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=self.api_keys["doubao"],
            )
            logger.info("Initialized doubao model")

        elif self.llm_model == "gpt4":
            if not self.api_keys.get("openai"):
                raise ValueError("OpenAI API key is required")
            self.client = OpenAI(api_key=self.api_keys["openai"])
            logger.info("Initialized GPT-4 (via OpenAI client)")

        elif self.llm_model == "qwen":
            # Configure Qwen-VL to use an external vLLM HTTP server instead of
            # loading the model locally. The server should expose an
            # OpenAI-compatible /v1/chat/completions endpoint.
            #
            # Config priority:
            #   1) config["qwen_server_url"], config["qwen_model_name"]
            #   2) env QWEN_VL_SERVER_URL, default model "Qwen/Qwen2.5-VL-7B-Instruct"
            self.qwen_server_url = self.config.get("qwen_server_url") or os.getenv(
                "QWEN_VL_SERVER_URL"
            )
            self.qwen_model_name = self.config.get("qwen_model_name") or "Qwen/Qwen2.5-VL-7B-Instruct"

            if not self.qwen_server_url:
                logger.warning(
                    "Qwen-VL vLLM server URL not configured; set qwen_server_url in config or QWEN_VL_SERVER_URL env."
                )
            else:
                logger.info(
                    "Using external Qwen-VL vLLM server at %s with model %s",
                    self.qwen_server_url,
                    self.qwen_model_name,
                )
        else:
            raise ValueError(f"Unsupported LLM model: {self.llm_model}")

    def generate_text_only(self, prompt, max_tokens: int = 512) -> str:
        """Use the underlying LLM in text-only mode.

        This bypasses visual retrieval and is suitable for summarization over
        existing文本，例如基于七个维度回答再次输出一句话结论。
        """

        try:
            if self.llm_model == "doubao":
                response = self.llm.responses.create(  # type: ignore[attr-defined]
                    model="doubao-seed-1-6-flash-250828",
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": prompt,
                                }
                            ],
                        }
                    ],
                )
                try:
                    return response.output[1].content[0].text  # type: ignore[union-attr,index]
                except Exception:
                    return ""

            if self.llm_model == "gpt4":
                response = self.client.chat.completions.create(  # type: ignore[attr-defined]
                    model="chatgpt-4o-latest",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.3,
                )
                return (response.choices[0].message.content or "").strip()  # type: ignore[union-attr]

            if self.llm_model == "qwen":
                server_url = getattr(self, "qwen_server_url", None)
                model_name = getattr(self, "qwen_model_name", None) or "Qwen/Qwen2.5-VL-7B-Instruct"

                if not server_url:
                    logger.warning("qwen_server_url not configured; cannot call Qwen-VL text-only endpoint")
                    return ""

                s = server_url.strip()
                if not (s.startswith("http://") or s.startswith("https://")):
                    s = "http://" + s
                base = s.rstrip("/")
                if not base.endswith("/v1"):
                    base = f"{base}/v1"
                url = f"{base}/chat/completions"

                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                }

                headers = {"Content-Type": "application/json"}

                resp = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()

                choices = data.get("choices") or []
                if not choices:
                    return ""
                message = choices[0].get("message", {})
                content = message.get("content")

                if isinstance(content, str):
                    generated = content
                else:
                    parts = []
                    for block in content or []:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(block.get("text", ""))
                    generated = "".join(parts)

                return (generated or "").strip()

            raise ValueError(f"Unsupported LLM model for text-only generation: {self.llm_model}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error in generate_text_only: {str(e)}")
            return ""

    def extract_sections(self, text):
        """Extract markdown sections (Evidence, Chain of Thought, Answer)."""
        sections = {}
        headings = ["Evidence", "Chain of Thought", "Answer"]

        for i in range(len(headings)):
            heading = headings[i]
            next_heading = headings[i + 1] if i + 1 < len(headings) else None

            if next_heading:
                pattern = rf"## {heading}:(.*?)(?=## {next_heading}:)"
            else:
                pattern = rf"## {heading}:(.*)"

            match = re.search(pattern, text, re.DOTALL)
            if match:
                sections[heading] = match.group(1).strip()
            else:
                sections[heading] = ""

        return sections

    def answer_question(self, question):
        """Interactive visual QA over configured PDFs for a natural-language question."""
        try:
            overall_start = time.time()
            logger.info("Retrieving visual contexts for interactive question...")

            retrieval_start = time.time()
            visual_contexts = self.visual_engine.retrieve_visual_contexts_for_question(
                question
            )
            retrieval_end = time.time()

            if not visual_contexts:
                logger.warning("No visual contexts found for interactive question")
                return None

            logger.info("Generating visual response...")
            generation_start = time.time()
            visual_response = self.visual_engine.generate_visual_response(
                question, visual_contexts
            )
            generation_end = time.time()

            vr_dict = self.extract_sections(visual_response)
            vr_dict.update(
                {
                    "question": question,
                    "document": [ctx["document_id"] for ctx in visual_contexts],
                    "pages": [ctx["page_number"] for ctx in visual_contexts],
                }
            )

            overall_end = time.time()

            total_time = overall_end - overall_start
            retrieval_time = retrieval_end - retrieval_start
            generation_time = generation_end - generation_start

            logger.info(
                "answer_question (visual only) completed in %.2f seconds (retrieval: %.2f, generation: %.2f)",
                total_time,
                retrieval_time,
                generation_time,
            )

            result = {
                "question": question,
                "answer": vr_dict.get("Answer", ""),
                "analysis": vr_dict.get("Chain of Thought", ""),
                "conclusion": "",
                "response": vr_dict,
                "timing": {
                    "total_seconds": total_time,
                    "retrieval_seconds": retrieval_time,
                    "generation_seconds": generation_time,
                },
            }

            # Append timing info to per-run retrieval log under data_dir/retrieval
            timing_record = {
                "question": question,
                "documents": vr_dict.get("document", []),
                "pages": vr_dict.get("pages", []),
                "total_seconds": total_time,
                "retrieval_seconds": retrieval_time,
                "generation_seconds": generation_time,
            }
            try:
                with open(self.timing_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(timing_record, ensure_ascii=False) + "\n")
            except Exception as log_err:
                logger.error("Failed to write timing log to %s: %s", self.timing_log_path, str(log_err))

            # Append summarized output record to per-run output log under output_dir
            output_record = {
                "question": question,
                "answer": result["answer"],
                "analysis": result["analysis"],
                "documents": vr_dict.get("document", []),
                "pages": vr_dict.get("pages", []),
                "total_seconds": total_time,
            }
            try:
                with open(self.output_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            except Exception as log_err:
                logger.error("Failed to write output log to %s: %s", self.output_log_path, str(log_err))

            return result
        except Exception as e:
            logger.error(f"Error answering interactive question: {str(e)}")
            traceback.print_exc()
            return None


__all__ = ["VisualRAGEngine", "VisualRAGPipeline"]
