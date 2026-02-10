import os
import json
import hashlib
import uuid
from pathlib import Path
import time
import asyncio
import threading
from typing import Any, Dict, AsyncIterator, Iterator, List, Optional
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

# ────────────────────── 模块级线程池 ──────────────────────
_POSTER_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)
_LONG_POSTER_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)
# 检索也放到线程池，避免阻塞事件循环
_RETRIEVAL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)


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
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _sse_ping() -> bytes:
    return b": keepalive\n\n"


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


# ────────────────────── 海报生成（同步，在线程池中执行） ──────────────────────

def _build_poster_sync(
    dk: str,
    dr: dict,
    style_ref: Optional[str],
) -> tuple:
    """在线程池中执行的同步海报生成函数。"""
    try:
        logger.info("Poster task started for dim: %s (style_ref=%s)", dk, bool(style_ref))
        result = build_poster_for_dimension(dk, dr, style_reference=style_ref)
        logger.info("Poster task finished for dim: %s, has_image=%s", dk, bool(result and result.get("image_result")))
        return dk, result
    except Exception:
        logger.exception("Poster generation failed for dim: %s", dk)
        return dk, None


def _build_long_poster_sync(
    session_id: str,
    title: str,
    summary_text: str,
    cover_style_ref: Optional[str],
    posters_generated: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """在线程池中执行的同步长图生成函数。"""
    try:
        cover_path, tail_path = generate_cover_and_tail_from_single_image(
            title=title,
            summary=summary_text,
            output_dir="./policy_outputs/posters",
            style_reference=cover_style_ref,
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
    except Exception:
        logger.exception("Long poster build failed for session %s", session_id)
        return None


def _retrieval_one_dim_sync(vis_pipeline, dim_key: str, question: str) -> dict:
    """在线程池中执行的单维度检索。"""
    try:
        resp = vis_pipeline.answer_question(question)
        if resp is None:
            return {"question": question, "answer": "", "analysis": ""}
        return {
            "question": question,
            "answer": resp.get("answer", ""),
            "analysis": resp.get("analysis", ""),
        }
    except Exception as exc:
        logger.exception("dim=%s answer_question failed", dim_key)
        return {
            "question": question,
            "answer": "",
            "analysis": f"调用出错: {exc}",
        }


# ────────────────────── 异步等待 future，期间发 ping ──────────────────────

async def _await_future_with_ping(
    future: concurrent.futures.Future,
    loop: asyncio.AbstractEventLoop,
    ping_interval: float = 1.0,
):
    """异步等待一个线程池 future 完成，期间每隔 ping_interval 秒返回控制权让调用者发 ping。"""
    while not future.done():
        await asyncio.sleep(ping_interval)
    return future.result()


async def _await_futures_with_ping(
    futures: Dict[str, concurrent.futures.Future],
    posters_generated: Dict[str, Dict[str, Any]],
    ping_interval: float = 1.0,
) -> AsyncIterator[bytes]:
    """异步等待多个 futures 完成，期间 yield ping 保持连接。"""
    pending = set(futures.values())
    future_to_dk = {v: k for k, v in futures.items()}

    while pending:
        # 非阻塞检查哪些已完成
        done = {f for f in pending if f.done()}

        for f in done:
            dk = future_to_dk.get(f, "?")
            try:
                result_dk, poster_info = f.result()
                if poster_info and poster_info.get("image_result"):
                    posters_generated[result_dk] = poster_info
                    logger.info("Poster collected for dim: %s", result_dk)
            except Exception:
                logger.exception("Async poster generation failed for dim: %s", dk)

        pending -= done

        if pending:
            yield _sse_ping()
            await asyncio.sleep(ping_interval)


@router.post("/api/policy-interpre")
async def policy_interpret(
    request: Request,
    message: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
) -> StreamingResponse:
    content_type = request.headers.get("content-type", "").lower()

    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                message = body.get("message") or body.get("prompt") or body.get("query") or body.get("text")
                if not message and "data" in body and isinstance(body["data"], dict):
                    message = body["data"].get("message")
        except Exception:
            logger.error("Failed to parse JSON body")

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

    async def _stream() -> AsyncIterator[bytes]:
        try:
            loop = asyncio.get_running_loop()
            intent: SessionIntent = detect_intent(message)

            text_header_sent = False
            image_header_sent = False

            if intent in ["text_only", "text_and_poster"]:
                yield _json_line({"type": "RESPONSE", "messageType": "TEXT"})
                text_header_sent = True

            partial: Dict[str, Dict[str, str]] = session_state.get("dim_answers") or {}
            posters_generated: Dict[str, Dict[str, Any]] = {}

            if not from_cache:
                # ───────── 新文件：检索 + 并行海报生成 ─────────
                logger.info("Starting new policy interpretation for session %s", session_id)

                # 在线程池中初始化 pipeline（可能涉及 I/O）
                def _init_pipeline():
                    pipeline = _build_policy_pipeline()
                    pipeline.add_inputs(inputs or [])
                    pipeline._ensure_visdom()
                    return pipeline

                pipeline = await loop.run_in_executor(_RETRIEVAL_EXECUTOR, _init_pipeline)
                vis_pipeline = pipeline._visdom

                # 海报任务协调
                poster_futures: Dict[str, concurrent.futures.Future] = {}
                first_poster_style_ref: Optional[str] = None
                first_poster_done = False

                # ── 逐维度检索（在线程池中执行），完成后提交海报任务 ──
                for dim_key, question in POLICY_QUESTIONS.items():
                    # 在线程池中执行检索，不阻塞事件循环
                    dim_res = await loop.run_in_executor(
                        _RETRIEVAL_EXECUTOR,
                        _retrieval_one_dim_sync,
                        vis_pipeline, dim_key, question,
                    )
                    partial[dim_key] = dim_res

                    # 流式输出文本
                    if intent in ["text_only", "text_and_poster"]:
                        answer_text = dim_res.get("answer", "")
                        if dim_res.get("analysis", "").startswith("调用出错"):
                            yield _json_line({"content": dim_res["analysis"]})
                        else:
                            for chunk in _text_chunks(answer_text):
                                yield _json_line({"content": chunk})

                    # 提交海报任务
                    if intent in ["text_and_poster", "poster_only"]:
                        style_ref_for_this = first_poster_style_ref if first_poster_done else None

                        future = _POSTER_EXECUTOR.submit(
                            _build_poster_sync,
                            dim_key,
                            dict(dim_res),
                            style_ref_for_this,
                        )
                        poster_futures[dim_key] = future

                        # 如果是第一个海报，同步等待它完成以获取风格参照
                        # 但用 asyncio 等待，不阻塞事件循环
                        if not first_poster_done:
                            logger.info("Waiting for first poster (dim=%s) to get style reference...", dim_key)
                            while not future.done():
                                yield _sse_ping()
                                await asyncio.sleep(1.0)
                            try:
                                _, first_result = future.result()
                                if first_result and first_result.get("image_result"):
                                    first_poster_style_ref = str(first_result["image_result"])
                                    posters_generated[dim_key] = first_result
                                    logger.info("First poster ready, style_ref=%s", first_poster_style_ref)
                            except Exception:
                                logger.exception("First poster failed for dim=%s", dim_key)
                            first_poster_done = True

                        logger.info(
                            "Submitted poster task for dim: %s, continuing to next retrieval",
                            dim_key,
                        )

                    # 发送 ping 保持连接活跃
                    yield _sse_ping()

                # ── 生成摘要 ──
                try:
                    def _gen_summary():
                        return generate_one_sentence_summary(vis_pipeline, partial)

                    partial["summary"] = await loop.run_in_executor(
                        _RETRIEVAL_EXECUTOR, _gen_summary
                    )
                    logger.info(
                        "Interpretation finished for session %s. Partial keys: %s",
                        session_id, list(partial.keys()),
                    )
                except Exception as exc:
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

                # ── 收集剩余海报任务 ──
                remaining_futures = {
                    dk: f for dk, f in poster_futures.items()
                    if dk not in posters_generated
                }
                if remaining_futures:
                    logger.info(
                        "Waiting for %d remaining poster tasks...",
                        len(remaining_futures),
                    )
                    async for ping in _await_futures_with_ping(
                        remaining_futures, posters_generated, ping_interval=1.0
                    ):
                        yield ping

                    logger.info(
                        "All poster tasks collected. %d posters generated.",
                        len(posters_generated),
                    )

            else:
                # ───────── 缓存模式 ─────────
                logger.info("Using cached interpretation for session %s", session_id)

                if intent in ["text_only", "text_and_poster"]:
                    for dim_key, dim_res in partial.items():
                        if dim_key == "summary":
                            continue
                        a = dim_res.get("answer", "")
                        for chunk in _text_chunks(a):
                            yield _json_line({"content": chunk})

                if intent in ["text_and_poster", "poster_only"]:
                    logger.info("Regenerating posters from cache for session %s", session_id)

                    poster_futures_cache: Dict[str, concurrent.futures.Future] = {}
                    first_done_cache = False
                    style_ref_cache: Optional[str] = None

                    for dim_key in POLICY_QUESTIONS.keys():
                        dr = partial.get(dim_key) or {}
                        style_for_this = style_ref_cache if first_done_cache else None

                        future = _POSTER_EXECUTOR.submit(
                            _build_poster_sync, dim_key, dict(dr), style_for_this,
                        )
                        poster_futures_cache[dim_key] = future

                        if not first_done_cache:
                            while not future.done():
                                yield _sse_ping()
                                await asyncio.sleep(1.0)
                            try:
                                _, first_res = future.result()
                                if first_res and first_res.get("image_result"):
                                    style_ref_cache = str(first_res["image_result"])
                                    posters_generated[dim_key] = first_res
                            except Exception:
                                logger.exception("First cached poster failed dim=%s", dim_key)
                            first_done_cache = True

                    remaining = {
                        dk: f for dk, f in poster_futures_cache.items()
                        if dk not in posters_generated
                    }
                    async for ping in _await_futures_with_ping(
                        remaining, posters_generated, ping_interval=1.0
                    ):
                        yield ping

            # ───────── 生成长图 ─────────
            if intent in ["text_and_poster", "poster_only"] and posters_generated:
                try:
                    logger.info(
                        "Starting long poster concatenation with %d posters...",
                        len(posters_generated),
                    )
                    title = "政策解读"
                    if saved_rel:
                        try:
                            title = Path(saved_rel).stem or title
                        except Exception:
                            pass

                    summary_text = ""
                    try:
                        summary_text = ((partial.get("summary") or {}).get("answer") or "").strip()
                    except Exception:
                        pass

                    cover_style_ref: Optional[str] = None
                    for dk in POLICY_QUESTIONS.keys():
                        info = posters_generated.get(dk)
                        if info and info.get("image_result"):
                            cover_style_ref = str(info["image_result"])
                            break

                    # 提交长图生成到线程池
                    long_future = _LONG_POSTER_EXECUTOR.submit(
                        _build_long_poster_sync,
                        session_id,
                        title,
                        summary_text,
                        cover_style_ref,
                        posters_generated,
                    )

                    # 异步等待，期间发 ping
                    while not long_future.done():
                        yield _sse_ping()
                        await asyncio.sleep(1.0)

                    long_path = long_future.result()

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
                    else:
                        logger.error("Long poster generation returned None for session %s", session_id)
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