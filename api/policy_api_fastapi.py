import os
os.environ["HF_ENDPOINT"] = "http://hf-mirror.com"
import asyncio
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from retrieval_pipe import (
    POLICY_QUESTIONS,
    PolicyRetrievalPipeline,
    build_dim_summary_text,
    generate_one_sentence_summary,
)



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


def save_uploaded_files(uploaded_files: List[UploadFile], base_dir: str = "./policy_data/uploads") -> List[str]:
    saved_paths: List[str] = []
    if not uploaded_files:
        return saved_paths

    os.makedirs(base_dir, exist_ok=True)

    for uf in uploaded_files:
        filename = os.path.basename(uf.filename)
        ext = Path(filename).suffix.lower()
        if ext and ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

        save_path = os.path.join(base_dir, filename)
        with open(save_path, "wb") as f:
            f.write(uf.file.read())
        rel_path = os.path.relpath(save_path, ".")
        saved_paths.append(rel_path)

    return saved_paths


class JobConfig(BaseModel):
    llm_model: Literal["doubao", "gpt4", "qwen"] = "doubao"
    vision_retriever: Literal["colpali", "colqwen", "nemo"] = "colpali"
    top_k: int = Field(default=5, ge=1, le=10)
    force_reindex: bool = False
    qa_prompt: str = "请基于给定政策文本，客观提取和归纳关键信息。请务必用中文回答问题。"

    # keys / services
    doubao_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    qwen_server_url: Optional[str] = None
    qwen_model_name: Optional[str] = None


class CreateJobRequest(BaseModel):
    inputs: List[str] = Field(default_factory=list, description="相对路径或 URL")
    config: JobConfig = Field(default_factory=JobConfig)


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    error: Optional[str] = None
    result: Optional[Dict[str, Dict[str, str]]] = None


JOBS: Dict[str, Dict[str, Any]] = {}


def _build_pipeline(cfg: JobConfig) -> PolicyRetrievalPipeline:
    api_keys: Dict[str, str] = {}
    if cfg.doubao_api_key:
        api_keys["doubao"] = cfg.doubao_api_key
    if cfg.openai_api_key:
        api_keys["openai"] = cfg.openai_api_key

    return PolicyRetrievalPipeline(
        data_dir="./policy_data",
        output_dir="./policy_outputs",
        llm_model=cfg.llm_model,
        vision_retriever=cfg.vision_retriever,
        api_keys=api_keys,
        top_k=cfg.top_k,
        force_reindex=cfg.force_reindex,
        qa_prompt=cfg.qa_prompt,
        extra_config=None,
        qwen_server_url=cfg.qwen_server_url,
        qwen_model_name=cfg.qwen_model_name,
    )


async def _run_job(job_id: str, inputs: List[str], cfg: JobConfig) -> None:
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["error"] = None
    JOBS[job_id]["result"] = {}

    def _work() -> Dict[str, Dict[str, str]]:
        pipeline = _build_pipeline(cfg)

        pipeline.add_inputs(inputs)
        pipeline._ensure_visdom()  # type: ignore[attr-defined]

        partial: Dict[str, Dict[str, str]] = {}
        for key, question in POLICY_QUESTIONS.items():
            try:
                response = pipeline._visdom.answer_question(question)  # type: ignore[attr-defined]
                if response is None:
                    partial[key] = {"question": question, "answer": "", "analysis": ""}
                else:
                    partial[key] = {
                        "question": question,
                        "answer": response.get("answer", ""),
                        "analysis": response.get("analysis", ""),
                    }
            except Exception as exc:  # noqa: BLE001
                partial[key] = {"question": question, "answer": "", "analysis": f"调用出错: {exc}"}

            JOBS[job_id]["result"] = partial.copy()

        # 额外一步：在 7 个维度回答完成后，让模型基于这些回答输出一句话结论
        try:
            vis_pipeline = pipeline._visdom  # type: ignore[attr-defined]
            partial["summary"] = generate_one_sentence_summary(vis_pipeline, partial)
        except Exception as exc:  # noqa: BLE001
            # generate_one_sentence_summary 已经会将错误写入 analysis，这里仅兜底
            partial.setdefault("summary", {
                "question": "",
                "answer": "",
                "analysis": f"调用出错: {exc}",
            })

        # 生成 AI 行动建议时复用同一份 dim_summary 汇总文案
        dim_summary = build_dim_summary_text(partial)

        actions_prompt = (
            "下面是你刚才针对同一份政策文本、从七个维度给出的详细回答：\n"  # noqa: E501
            f"{dim_summary}\n\n"
            "请基于这些回答，面向申报主体给出3-6条AI行动建议，概括需要重点关注的注意事项、风险提示以及申报材料准备要点。"
            "要求：用中文作答，以列表形式输出，每条建议不超过50字。"
        )
        try:
            vis_pipeline = pipeline._visdom  # type: ignore[attr-defined]
            actions_text = vis_pipeline.generate_text_only(actions_prompt)  # type: ignore[attr-defined]

            partial["actions"] = {
                "question": actions_prompt,
                "answer": actions_text,
                "analysis": "",
            }
        except Exception as exc:  # noqa: BLE001
            partial["actions"] = {
                "question": actions_prompt,
                "answer": "",
                "analysis": f"调用出错: {exc}",
            }

        JOBS[job_id]["result"] = partial.copy()

        return partial

    try:
        result = await asyncio.to_thread(_work)
        JOBS[job_id]["status"] = "succeeded"
        JOBS[job_id]["result"] = result
    except Exception as e:  # noqa: BLE001
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)


app = FastAPI(title="Policy Reader API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/policy/files")
def list_policy_files() -> Dict[str, List[str]]:
    return {"files": scan_policy_files("./policy_data")}


@app.post("/api/policy/upload")
def upload_policy_files(files: List[UploadFile] = File(...)) -> Dict[str, List[str]]:
    saved = save_uploaded_files(files)
    return {"saved_paths": saved}


@app.post("/api/policy/jobs", response_model=JobStatusResponse)
async def create_job(req: CreateJobRequest, background_tasks: BackgroundTasks) -> JobStatusResponse:
    if not req.inputs:
        raise HTTPException(status_code=400, detail="inputs 不能为空")

    if req.config.llm_model == "doubao" and not (req.config.doubao_api_key or os.getenv("ARK_API_KEY")):
        raise HTTPException(status_code=400, detail="已选择 doubao 模型，但 doubao_api_key / ARK_API_KEY 为空")
    if req.config.llm_model == "gpt4" and not (req.config.openai_api_key or os.getenv("OPENAI_API_KEY")):
        raise HTTPException(status_code=400, detail="已选择 gpt4 模型，但 openai_api_key / OPENAI_API_KEY 为空")
    if req.config.llm_model == "qwen" and not (req.config.qwen_server_url or os.getenv("QWEN_VL_SERVER_URL")):
        raise HTTPException(status_code=400, detail="已选择 qwen 模型，但 qwen_server_url / QWEN_VL_SERVER_URL 为空")

    # 用环境变量做默认回填
    cfg = req.config
    if cfg.doubao_api_key is None:
        cfg.doubao_api_key = os.getenv("ARK_API_KEY")
    if cfg.openai_api_key is None:
        cfg.openai_api_key = os.getenv("OPENAI_API_KEY")
    if cfg.qwen_server_url is None:
        cfg.qwen_server_url = os.getenv("QWEN_VL_SERVER_URL")
    if cfg.qwen_model_name is None:
        cfg.qwen_model_name = os.getenv("QWEN_VL_MODEL_NAME")

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "status": "queued",
        "error": None,
        "result": None,
        "inputs": req.inputs,
    }

    background_tasks.add_task(_run_job, job_id, req.inputs, cfg)

    return JobStatusResponse(job_id=job_id, status="queued")


@app.get("/api/policy/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        error=job.get("error"),
        result=job.get("result"),
    )
