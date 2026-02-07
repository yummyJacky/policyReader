import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
import json
import asyncio
import hashlib
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from dotenv import load_dotenv
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from retrieval_pipe import PolicyRetrievalPipeline, POLICY_QUESTIONS, build_dim_summary_text, generate_one_sentence_summary
from poster_pipeline import build_poster_records_from_answers
from policy_intent import SessionIntent, detect_intent


load_dotenv()


router = APIRouter()


class PolicyAPIError(Exception):
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


ERROR_CODES = {
    "TOKEN_LIMIT": 1000,
    "AUTH_FAILED": 1001,
    "TIMEOUT": 1002,
}


SESSIONS: Dict[str, Dict[str, Any]] = {}


def _get_or_create_session_id(request: Request) -> str:
    sid = request.cookies.get("policy_session")
    if sid:
        return sid
    return uuid.uuid4().hex


def _get_session_state(session_id: str) -> Dict[str, Any]:
    state = SESSIONS.get(session_id)
    if state is None:
        state = {}
        SESSIONS[session_id] = state
    return state


def _make_doc_tag(saved_rel: str) -> str:
    base = saved_rel.strip().lstrip("./")
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]


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


def _save_upload_file(uploaded_file: UploadFile, base_dir: str = "./policy_data/uploads") -> str:
    if uploaded_file.filename is None:
        raise HTTPException(status_code=400, detail="file.filename 为空")

    filename = os.path.basename(uploaded_file.filename)
    ext = Path(filename).suffix.lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    os.makedirs(base_dir, exist_ok=True)

    save_path = os.path.join(base_dir, filename)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.file.read())

    return os.path.relpath(save_path, ".")


def _build_policy_pipeline() -> PolicyRetrievalPipeline:
    llm_model = os.getenv("POLICY_LLM_MODEL", "doubao")
    vision_retriever = os.getenv("POLICY_VISION_RETRIEVER", "nemo")
    top_k = int(os.getenv("POLICY_TOP_K", "5"))

    api_keys: Dict[str, str] = {}
    ark_key = os.getenv("ARK_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if llm_model == "doubao":
        if not ark_key:
            raise HTTPException(status_code=400, detail="已选择 doubao 模型，但 ARK_API_KEY 为空")
        api_keys["doubao"] = ark_key

    if llm_model == "gpt4":
        if not openai_key:
            raise HTTPException(status_code=400, detail="已选择 gpt4 模型，但 OPENAI_API_KEY 为空")
        api_keys["openai"] = openai_key

    qwen_server_url = os.getenv("QWEN_VL_SERVER_URL")
    qwen_model_name = os.getenv("QWEN_VL_MODEL_NAME")
    if llm_model == "qwen" and not qwen_server_url:
        raise HTTPException(status_code=400, detail="已选择 qwen 模型，但 QWEN_VL_SERVER_URL 为空")

    return PolicyRetrievalPipeline(
        data_dir="./policy_data",
        output_dir="./policy_outputs",
        llm_model=llm_model,
        vision_retriever=vision_retriever,
        api_keys=api_keys,
        top_k=top_k,
        force_reindex=False,
        qa_prompt="请基于给定政策文本，客观提取和归纳关键信息。请务必用中文回答问题。",
        extra_config=None,
        qwen_server_url=qwen_server_url,
        qwen_model_name=qwen_model_name,
    )


def _json_line(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def _error_line(code: int, message: str) -> bytes:
    error_content = json.dumps({"code": code, "message": message}, ensure_ascii=False)
    return _json_line({
        "type": "ERROR",
        "messageType": "TEXT",
        "content": error_content
    })


def _make_public_file_url(request: Request, local_path: str) -> str:
    rel = local_path.strip().lstrip("./")
    base = str(request.base_url).rstrip("/")

    public_prefix = (os.getenv("POLICY_PUBLIC_FILES_PREFIX") or "/files").rstrip("/")
    outputs_prefix = (os.getenv("POLICY_OUTPUTS_PREFIX") or "policy_outputs/")

    if rel.startswith(outputs_prefix):
        rel = rel[len(outputs_prefix) :]
    rel = rel.lstrip("/")

    return f"{base}{public_prefix}/{rel}"


@router.post("/api/policy-intent")
async def policy_intent(message: str = Form(...)) -> Dict[str, Any]:
    intent = detect_intent(message)
    return {"intent": intent}


@router.post("/api/policy-interpre")
async def policy_interpret(
    request: Request,
    message: str = Form(...),
    file: Optional[UploadFile] = File(default=None),
) -> StreamingResponse:
    session_id = _get_or_create_session_id(request)
    session_state = _get_session_state(session_id)

    intent: SessionIntent = detect_intent(message)

    saved_rel: Optional[str] = None
    inputs: Optional[List[str]] = None
    from_cache = False

    if file is not None:
        saved_rel = _save_upload_file(file)
        inputs = [saved_rel]
        session_state.clear()
        session_state["saved_rel"] = saved_rel
        session_state["doc_tag"] = _make_doc_tag(saved_rel)
    else:
        cached_answers = session_state.get("dim_answers")
        cached_doc_tag = session_state.get("doc_tag")
        if cached_answers and cached_doc_tag:
            from_cache = True
        else:
            raise HTTPException(status_code=400, detail="请先上传政策文档以进行解析")

    async def _stream() -> AsyncIterator[bytes]:
        try:
            yield _json_line({
                "type": "RESPONSE",
                "messageType": "TEXT",
                "content": "开始处理...",
            })

            def _work() -> Dict[str, Any]:
                try:
                    if from_cache:
                        partial = session_state.get("dim_answers") or {}
                        dim_summary = session_state.get("dim_summary") or ""

                        out: Dict[str, Any] = {
                            "intent": intent,
                            "dim_answers": partial,
                            "dim_summary": dim_summary,
                            "doc_tag": session_state.get("doc_tag"),
                            "from_cache": True,
                        }

                        if intent in ["text_and_poster", "poster_only"]:
                            posters = build_poster_records_from_answers(partial)
                            out["posters"] = posters
                        else:
                            out["posters"] = {}

                        return out

                    pipeline = _build_policy_pipeline()
                    pipeline.add_inputs(inputs or [])
                    pipeline._ensure_visdom()  # type: ignore[attr-defined]

                    vis_pipeline = pipeline._visdom  # type: ignore[attr-defined]

                    # 核心逻辑：固定使用 POLICY_QUESTIONS 进行检索和回答
                    partial: Dict[str, Dict[str, str]] = {}
                    for key, question in POLICY_QUESTIONS.items():
                        try:
                            resp = pipeline._visdom.answer_question(question)  # type: ignore[attr-defined]
                            if resp is None:
                                partial[key] = {"question": question, "answer": "", "analysis": ""}
                            else:
                                partial[key] = {
                                    "question": question,
                                    "answer": resp.get("answer", ""),
                                    "analysis": resp.get("analysis", ""),
                                }
                        except Exception as exc:  # noqa: BLE001
                            partial[key] = {"question": question, "answer": "", "analysis": f"调用出错: {exc}"}

                    partial["summary"] = generate_one_sentence_summary(vis_pipeline, partial)
                    dim_summary = build_dim_summary_text(partial)
                    
                    out: Dict[str, Any] = {
                        "intent": intent,
                        "dim_answers": partial,
                        "dim_summary": dim_summary,
                        "doc_tag": session_state.get("doc_tag") or (inputs[0] if inputs else None),
                        "from_cache": False,
                    }

                    session_state["dim_answers"] = partial
                    session_state["dim_summary"] = dim_summary

                    # 如果意图包含海报，则生成海报（完整流水线）
                    if intent in ["text_and_poster", "poster_only"]:
                        posters = build_poster_records_from_answers(partial)
                        out["posters"] = posters
                    else:
                        out["posters"] = {}

                    return out
                except PolicyAPIError:
                    raise
                except asyncio.TimeoutError:
                    raise PolicyAPIError(ERROR_CODES["TIMEOUT"], "响应超时")
                except Exception as exc:
                    exc_msg = str(exc).lower()
                    # 识别模型内部抛出的上下文超限错误（针对豆包/OpenAI/Qwen等常见错误关键字）
                    if any(x in exc_msg for x in ["context_length_exceeded", "token limit", "context window", "maximum context length", "429"]):
                        raise PolicyAPIError(ERROR_CODES["TOKEN_LIMIT"], "上下文 token 超出限制")
                    
                    if any(x in exc_msg for x in ["auth", "api_key", "401", "unauthorized"]):
                        raise PolicyAPIError(ERROR_CODES["AUTH_FAILED"], "认证失败")
                    raise PolicyAPIError(500, f"解析执行失败: {exc}")

            try:
                result = await asyncio.to_thread(_work)
            except PolicyAPIError as e:
                yield _error_line(e.code, e.message)
                return
            except Exception as exc:  # noqa: BLE001
                yield _error_line(500, f"解析执行失败: {exc}")
                return

            intent = result.get("intent")

            if intent in ["text_only", "text_and_poster"]:
                # 文本解读输出
                summary_answer = (result.get("dim_answers", {}).get("summary") or {}).get("answer", "")
                if summary_answer:
                    yield _json_line({"type": "RESPONSE", "messageType": "TEXT", "content": summary_answer})
                
                dim_summary = result.get("dim_summary") or ""
                if dim_summary:
                    yield _json_line({"type": "RESPONSE", "messageType": "TEXT", "content": dim_summary})

            if intent in ["text_and_poster", "poster_only"]:
                # 海报图片输出
                posters = result.get("posters") or {}
                idx = 0
                for dim_key, info in posters.items():
                    image_path = info.get("image_result")
                    if not image_path:
                        continue
                    idx += 1
                    yield _json_line({
                        "type": "RESPONSE",
                        "messageType": "IMAGE",
                        "content": {
                            "title": f"图{idx}({dim_key})",
                            "url": _make_public_file_url(request, str(image_path)),
                            "doc_tag": result.get("doc_tag"),
                            "from_cache": bool(result.get("from_cache")),
                        },
                    })

            yield b"[DONE]\n"

        except Exception as e:
            yield _error_line(500, f"服务器内部错误: {e}")

    resp = StreamingResponse(_stream(), media_type="text/plain; charset=utf-8")
    resp.set_cookie(
        key="policy_session",
        value=session_id,
        httponly=True,
        samesite="lax",
    )
    return resp
