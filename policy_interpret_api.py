import os
import json
import hashlib
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from retrieval_pipe import PolicyRetrievalPipeline, POLICY_QUESTIONS, build_dim_summary_text, generate_one_sentence_summary
from poster_pipeline import build_poster_for_dimension
from policy_intent import SessionIntent, detect_intent


load_dotenv()


router = APIRouter()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://js2.blockelite.cn:17672",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory="./policy_outputs"), name="files")


@router.get("/api/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}


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
    # SSE 格式要求以 data: 开头，并以两个换行符结尾
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _text_chunks(text: str, *, chunk_size: int = 200) -> List[str]:
    if not text:
        return []
    s = text.strip("\n")
    if not s:
        return []
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]


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


@router.post("/api/policy-interpre")
def policy_interpret(
    request: Request,
    message: str = Form(...),
    file: Optional[UploadFile] = File(default=None),
) -> StreamingResponse:
    session_id = _get_or_create_session_id(request)
    session_state = _get_session_state(session_id)

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

    def _stream() -> Iterator[bytes]:
        try:
            intent: SessionIntent = detect_intent(message)

            text_header_sent = False
            image_header_sent = False

            if intent in ["text_only", "text_and_poster"]:
                yield _json_line({"type": "RESPONSE", "messageType": "TEXT"})
                text_header_sent = True
                yield _json_line({"content": "开始处理..."})

            partial: Dict[str, Dict[str, str]] = session_state.get("dim_answers") or {}
            doc_tag = session_state.get("doc_tag")

            if not from_cache:
                pipeline = _build_policy_pipeline()
                pipeline.add_inputs(inputs or [])
                pipeline._ensure_visdom()  # type: ignore[attr-defined]
                vis_pipeline = pipeline._visdom  # type: ignore[attr-defined]

                for dim_key, question in POLICY_QUESTIONS.items():
                    try:
                        if intent in ["text_only", "text_and_poster"]:
                            if not text_header_sent:
                                yield _json_line({"type": "RESPONSE", "messageType": "TEXT"})
                                text_header_sent = True
                            yield _json_line({"content": f"正在处理：{question}"})

                        resp = vis_pipeline.answer_question(question)
                        if resp is None:
                            dim_res = {"question": question, "answer": "", "analysis": ""}
                        else:
                            dim_res = {
                                "question": question,
                                "answer": resp.get("answer", ""),
                                "analysis": resp.get("analysis", ""),
                            }
                        partial[dim_key] = dim_res

                        if intent in ["text_only", "text_and_poster"]:
                            # 逐维度输出：先输出问题，再流式输出答案内容
                            if not text_header_sent:
                                yield _json_line({"type": "RESPONSE", "messageType": "TEXT"})
                                text_header_sent = True
                            yield _json_line({"content": f"【{question}】"})
                            for chunk in _text_chunks(dim_res.get("answer", "")):
                                yield _json_line({"content": chunk})
                    except Exception as exc:  # noqa: BLE001
                        partial[dim_key] = {
                            "question": question,
                            "answer": "",
                            "analysis": f"调用出错: {exc}",
                        }

                        if intent in ["text_only", "text_and_poster"]:
                            if not text_header_sent:
                                yield _json_line({"type": "RESPONSE", "messageType": "TEXT"})
                                text_header_sent = True
                            yield _json_line({"content": f"【{question}】"})
                            yield _json_line({"content": f"调用出错: {exc}"})

                    # 如果需要海报，做到“每生成一张就推一张”
                    if intent in ["text_and_poster", "poster_only"]:
                        try:
                            if intent == "text_and_poster":
                                if not text_header_sent:
                                    yield _json_line({"type": "RESPONSE", "messageType": "TEXT"})
                                    text_header_sent = True
                                yield _json_line({"content": f"正在生成海报：{dim_key}"})

                            poster_info = build_poster_for_dimension(dim_key, partial.get(dim_key) or {})
                            if poster_info and poster_info.get("image_result"):
                                if not image_header_sent:
                                    yield _json_line({"type": "RESPONSE", "messageType": "IMAGE"})
                                    image_header_sent = True
                                yield _json_line({
                                    "content": {
                                        "title": f"{dim_key}",
                                        "url": _make_public_file_url(
                                            request, str(poster_info["image_result"])
                                        ),
                                        "doc_tag": doc_tag,
                                        "from_cache": from_cache,
                                    }
                                })
                        except Exception:
                            pass

                try:
                    partial["summary"] = generate_one_sentence_summary(vis_pipeline, partial)
                except Exception as exc:  # noqa: BLE001
                    partial["summary"] = {
                        "question": "",
                        "answer": "",
                        "analysis": f"调用出错: {exc}",
                    }

                dim_summary = build_dim_summary_text(partial)
                session_state["dim_answers"] = partial
                session_state["dim_summary"] = dim_summary

                if intent in ["text_only", "text_and_poster"]:
                    summary_text = (partial.get("summary") or {}).get("answer", "")
                    if summary_text:
                        if not text_header_sent:
                            yield _json_line({"type": "RESPONSE", "messageType": "TEXT"})
                            text_header_sent = True
                        yield _json_line({"content": "【总结】"})
                        for chunk in _text_chunks(summary_text):
                            yield _json_line({"content": chunk})
                    if dim_summary:
                        if not text_header_sent:
                            yield _json_line({"type": "RESPONSE", "messageType": "TEXT"})
                            text_header_sent = True
                        yield _json_line({"content": "【维度汇总】"})
                        for chunk in _text_chunks(dim_summary):
                            yield _json_line({"content": chunk})

            else:
                # 直接把缓存的维度答案逐条推给前端

                if intent in ["text_only", "text_and_poster"]:
                    for dim_key, dim_res in partial.items():
                        if dim_key == "summary":
                            continue
                        q = dim_res.get("question", "")
                        a = dim_res.get("answer", "")
                        if q:
                            if not text_header_sent:
                                yield _json_line({"type": "RESPONSE", "messageType": "TEXT"})
                                text_header_sent = True
                            yield _json_line({"content": f"【{q}】"})
                        for chunk in _text_chunks(a):
                            yield _json_line({"content": chunk})

                    cached_dim_summary = session_state.get("dim_summary") or ""
                    summary_text = (partial.get("summary") or {}).get("answer", "")
                    if summary_text:
                        if not text_header_sent:
                            yield _json_line({"type": "RESPONSE", "messageType": "TEXT"})
                            text_header_sent = True
                        yield _json_line({"content": "【总结】"})
                        for chunk in _text_chunks(summary_text):
                            yield _json_line({"content": chunk})
                    if cached_dim_summary:
                        if not text_header_sent:
                            yield _json_line({"type": "RESPONSE", "messageType": "TEXT"})
                            text_header_sent = True
                        yield _json_line({"content": "【维度汇总】"})
                        for chunk in _text_chunks(cached_dim_summary):
                            yield _json_line({"content": chunk})

                if intent in ["text_and_poster", "poster_only"]:
                    # 缓存模式下：按维度逐张生成并输出图片
                    for dim_key in POLICY_QUESTIONS.keys():
                        try:
                            poster_info = build_poster_for_dimension(dim_key, partial.get(dim_key) or {})
                            if poster_info and poster_info.get("image_result"):
                                if not image_header_sent:
                                    yield _json_line({"type": "RESPONSE", "messageType": "IMAGE"})
                                    image_header_sent = True
                                yield _json_line({
                                    "content": {
                                        "title": f"{dim_key}",
                                        "url": _make_public_file_url(
                                            request, str(poster_info["image_result"])
                                        ),
                                        "doc_tag": doc_tag,
                                        "from_cache": from_cache,
                                    }
                                })
                        except Exception:
                            continue

            yield b"data: [DONE]\n\n"

        except PolicyAPIError as e:
            yield _json_line({"type": "ERROR", "messageType": "TEXT"})
            yield _json_line({"content": json.dumps({"code": e.code, "message": e.message}, ensure_ascii=False)})
            yield b"data: [DONE]\n\n"
        except Exception as e:
            yield _json_line({"type": "ERROR", "messageType": "TEXT"})
            yield _json_line({"content": json.dumps({"code": 500, "message": f"服务器内部错误: {e}"}, ensure_ascii=False)})
            yield b"data: [DONE]\n\n"

    resp = StreamingResponse(_stream(), media_type="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    resp.headers["Connection"] = "keep-alive"
    resp.set_cookie(
        key="policy_session",
        value=session_id,
        httponly=True,
        samesite="lax",
    )
    return resp


app.include_router(router)

