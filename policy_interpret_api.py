import os
import json
import hashlib
import uuid
from pathlib import Path
import time
from typing import Any, Dict, Iterator, List, Optional
import concurrent.futures

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from retrieval_pipe import PolicyRetrievalPipeline, POLICY_QUESTIONS, build_dim_summary_text, generate_one_sentence_summary
from poster_pipeline import (
    build_poster_for_dimension,
    concat_poster_images,
    generate_cover_and_tail_from_single_image,
)
from policy_intent import SessionIntent, detect_intent

from realtime_logger import get_logger, setup_realtime_logging


load_dotenv()

setup_realtime_logging(log_path="./uvicorn.log")
logger = get_logger("policy_interpret_api")

_LONG_POSTER_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)


router = APIRouter()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://js2.blockelite.cn:17672",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://139.159.236.86:8090"
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
    sid = request.cookies.get("session_id")
    if sid:
        return sid
    return uuid.uuid4().hex


def _get_session_state(session_id: str) -> Dict[str, Any]:
    state = SESSIONS.get(session_id)
    if state is None:
        state = {}
        SESSIONS[session_id] = state
    return state


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

def _sse_ping() -> bytes:
    return b"data: ping\n\n"
    
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
async def policy_interpret(
    request: Request,
    message: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
) -> StreamingResponse:
    # 强制手动解析逻辑，因为 FastAPI 在混合 Form/File 时对 JSON 的自动支持很弱
    content_type = request.headers.get("content-type", "").lower()
    
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                # 尝试从各种可能的字段名中获取 message
                message = body.get("message") or body.get("prompt") or body.get("query") or body.get("text")
                if not message and "data" in body and isinstance(body["data"], dict):
                    message = body["data"].get("message")
        except Exception:
            logger.error("Failed to parse JSON body")
    
    # 如果是 Form 数据，FastAPI 已经将 message 注入到变量中了，无需额外操作

    if message is None:
        logger.info(
            "Missing message field. content-type=%s",
            request.headers.get("content-type"),
        )
        raise HTTPException(
            status_code=422,
            detail="缺少 message 字段（支持 form-data 的 message 或 JSON 的 {\"message\": ...}）",
        )

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
    else:
        cached_answers = session_state.get("dim_answers")
        cached_saved_rel = session_state.get("saved_rel")
        if cached_answers and cached_saved_rel:
            from_cache = True
            saved_rel = str(cached_saved_rel)
            inputs = [saved_rel]
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

            partial: Dict[str, Dict[str, str]] = session_state.get("dim_answers") or {}
            posters_generated: Dict[str, Dict[str, Any]] = {}
            first_style_ref: str | None = None
            first_dim_key: str | None = None
            try:
                first_dim_key = next(iter(POLICY_QUESTIONS.keys()))
            except Exception:
                first_dim_key = None

            if not from_cache:
                logger.info("Starting new policy interpretation for session %s", session_id)
                pipeline = _build_policy_pipeline()
                pipeline.add_inputs(inputs or [])
                pipeline._ensure_visdom()  # type: ignore[attr-defined]
                vis_pipeline = pipeline._visdom  # type: ignore[attr-defined]

                poster_futures = []
                # 使用线程池异步生成海报
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                    for dim_key, question in POLICY_QUESTIONS.items():
                        try:
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
                                # 仅流式输出答案内容
                                for chunk in _text_chunks(dim_res.get("answer", "")):
                                    yield _json_line({"content": chunk})

                            # 立即提交海报生成任务
                            if intent in ["text_and_poster", "poster_only"]:
                                logger.info("Submitting async poster task for dim: %s", dim_key)
                                def _gen_poster(dk, dr):
                                    logger.info("Starting poster generation for dim: %s", dk)
                                    res = build_poster_for_dimension(dk, dr, style_reference=first_style_ref)
                                    logger.info("Finished poster generation for dim: %s", dk)
                                    return dk, res

                                if first_dim_key and dim_key == first_dim_key and first_style_ref is None:
                                    try:
                                        dk0, poster0 = _gen_poster(dim_key, dim_res)
                                        if poster0 and poster0.get("image_result"):
                                            posters_generated[dk0] = poster0
                                            first_style_ref = str(poster0.get("image_result"))
                                    except Exception:
                                        logger.exception("First-dimension poster generation failed")
                                else:
                                    poster_futures.append(pool.submit(_gen_poster, dim_key, dim_res))

                            # 检查是否有已完成的任务，及时回收结果
                            done_futures = [f for f in poster_futures if f.done()]
                            for f in done_futures:
                                try:
                                    dk, poster_info = f.result()
                                    if poster_info and poster_info.get("image_result"):
                                        posters_generated[dk] = poster_info
                                except Exception:
                                    logger.exception("Async poster generation failed (in loop)")
                                poster_futures.remove(f)

                        except Exception as exc:  # noqa: BLE001
                            partial[dim_key] = {
                                "question": question,
                                "answer": "",
                                "analysis": f"调用出错: {exc}",
                            }

                            if intent in ["text_only", "text_and_poster"]:
                                yield _json_line({"content": f"调用出错: {exc}"})
                            logger.exception("dim=%s question=%s answer_question failed", dim_key, question)

                    # 等待所有异步海报任务完成
                    for future in concurrent.futures.as_completed(poster_futures):
                        try:
                            dk, poster_info = future.result()
                            if poster_info and poster_info.get("image_result"):
                                posters_generated[dk] = poster_info
                        except Exception:
                            logger.exception("Async poster generation failed")

                try:
                    partial["summary"] = generate_one_sentence_summary(vis_pipeline, partial)
                    logger.info("Interpretation finished and cached for session %s. Partial keys: %s", session_id, list(partial.keys()))
                except Exception as exc:  # noqa: BLE001
                    partial["summary"] = {
                        "question": "",
                        "answer": "",
                        "analysis": f"总结生成出错: {exc}",
                    }
                    logger.error("generate_one_sentence_summary failed: %s", exc)

                dim_summary = build_dim_summary_text(partial)
                session_state["dim_answers"] = partial
                session_state["dim_summary"] = dim_summary
                logger.info("Session %s state updated and summary built.", session_id)

            else:
                logger.info("Using cached interpretation for session %s", session_id)
                # 直接把缓存的维度答案逐条推给前端
                if intent in ["text_only", "text_and_poster"]:
                    for dim_key, dim_res in partial.items():
                        if dim_key == "summary":
                            continue
                        a = dim_res.get("answer", "")
                        for chunk in _text_chunks(a):
                            yield _json_line({"content": chunk})

                if intent in ["text_and_poster", "poster_only"]:
                    # 缓存模式下：后端生成但不实时推送单张图片
                    logger.info("Regenerating posters from cache for session %s", session_id)
                    for dim_key in POLICY_QUESTIONS.keys():
                        try:
                            logger.info("Regenerating poster for dim: %s", dim_key)
                            poster_info = build_poster_for_dimension(dim_key, partial.get(dim_key) or {})
                            if poster_info and poster_info.get("image_result"):
                                posters_generated[dim_key] = poster_info
                        except Exception as e:
                            logger.exception("dim=%s build_poster_for_dimension failed (cache mode)", dim_key, exc_info=True)
                            continue

            # 生成可下载的长图：封面 + 各维度 + 尾页
            if intent in ["text_and_poster", "poster_only"] and posters_generated:
                try:
                    logger.info("Starting long poster concatenation...")
                    title = "政策解读"
                    if saved_rel:
                        try:
                            title = Path(saved_rel).stem or title
                        except Exception:
                            title = title

                    summary_text = ""
                    try:
                        summary_text = ((partial.get("summary") or {}).get("answer") or "").strip()
                    except Exception:
                        summary_text = ""

                    def _build_long_poster() -> str | None:
                        cover_path, tail_path = generate_cover_and_tail_from_single_image(
                            title=title,
                            summary=summary_text,
                            output_dir="./policy_outputs/posters",
                        )
                        logger.info("Cover and tail generated: %s, %s", cover_path, tail_path)

                        out_name = f"long_{session_id}.png"
                        out_path = os.path.join("./policy_outputs/posters", out_name)
                        return concat_poster_images(
                            posters_generated,
                            out_path,
                            cover_image=cover_path or None,
                            tail_image=tail_path or None,
                        )

                    future = _LONG_POSTER_EXECUTOR.submit(_build_long_poster)
                    last_ping = 0.0
                    try:
                        while not future.done():
                            now = time.time()
                            if now - last_ping >= 2.0:
                                yield _sse_ping()
                                last_ping = now
                            time.sleep(0.2)
                    except GeneratorExit:
                        try:
                            future.cancel()
                        except Exception:
                            pass
                        logger.info("Client disconnected during long poster generation. session=%s", session_id)
                        raise

                    long_path = future.result()

                    if long_path:
                        logger.info("Long poster generated at: %s", long_path)
                        if not image_header_sent:
                            yield _json_line({"type": "RESPONSE", "messageType": "IMAGE"})
                            image_header_sent = True
                        yield _json_line(
                            {
                                "content": {
                                    "title": "download_long_poster",
                                    "url": _make_public_file_url(request, str(long_path)),
                                    "from_cache": from_cache,
                                }
                            }
                        )
                        session_state["long_poster"] = long_path
                except Exception:
                    logger.exception("build long poster failed for session %s", session_id)

            logger.info("Stream finished for session %s", session_id)
            yield b"data: [DONE]\n\n"

        except GeneratorExit:
            logger.info("SSE stream closed by client. session=%s", session_id)
            raise

        except PolicyAPIError as e:
            logger.exception("PolicyAPIError")
            yield _json_line({"type": "ERROR", "messageType": "TEXT"})
            yield _json_line({"content": json.dumps({"code": e.code, "message": e.message}, ensure_ascii=False)})
            yield b"data: [DONE]\n\n"
        except Exception as e:
            logger.exception("Unhandled exception in policy_interpret")
            yield _json_line({"type": "ERROR", "messageType": "TEXT"})
            yield _json_line({"content": json.dumps({"code": 500, "message": f"服务器内部错误: {e}"}, ensure_ascii=False)})
            yield b"data: [DONE]\n\n"

    resp = StreamingResponse(_stream(), media_type="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    resp.headers["Connection"] = "keep-alive"

    return resp


app.include_router(router)

