# 农业政策智能解读平台（FastAPI + React）

实现了一个基于 **FastAPI 后端 + React 前端** 的政策文件智能解读平台，
利用视觉检索（ColPali / NemoRetriever）与大模型，对上传或指定的政策文件进行解析，
从 7 个核心维度自动抽取结构化信息，并给出一句话总结与 AI 行动建议。

## 功能概览

- **政策文件接入**：
  - 选择已有政策 PDF / Word / HTML 文件（位于 `policy_data/`）
  - 上传本地文件（保存到 `policy_data/uploads/`）
  - 输入政策网页 URL，自动转为 PDF

- **视觉检索 + 问答**：
  - 统一完成：Word/网页 → PDF → 视觉检索（ColPali / ColQwen / Nemo）→ 7 个维度问答
  - 7 个维度包括：申报主体、资金用途、补贴标准、申报门槛、合规要求、时间节点、申报流程与材料

- **增量结果展示**：
  - 后端按问题顺序依次推理，实时写入任务状态
  - 前端轮询任务进度，逐条更新各模块的展示，不必等待所有问题完成

- **高级汇总能力**：
  - 在 7 维度回答基础上，由文本 LLM 再生成：
    - **一句话结论**：概括政策核心目标与支持方向
    - **AI 行动建议**：面向申报主体的 3–6 条注意事项 / 风险提示 / 材料准备要点

---

## 目录结构（核心部分）

```text
policyReader/
├─ policy_api_fastapi.py        # FastAPI 后端入口与接口定义
├─ retrieval_pipe.py            # PolicyRetrievalPipeline：7 维度问答管线
├─ visual_rag.py                # VisualRAGPipeline：视觉检索 + LLM 封装
├─ utils.py                     # PDF 转换等工具
└─ policy_frontend_react/       # React 前端工程（Vite）
   ├─ src/App.jsx               # 主页面与交互逻辑
   ├─ src/api.js                # 调用 FastAPI 的封装
   ├─ src/components/Card.jsx   # 通用卡片组件
   ├─ src/styles.css            # 页面样式
   └─ ...
```

---

## 环境准备

### 1. Python 后端环境

建议使用 Conda / venv 创建独立环境（Python 3.10+，项目当前在 3.11 下开发验证）：

```bash
# 推荐：使用 environment.yml 创建完整环境（包含视觉检索与大模型依赖）
conda env create -f environment.yml
conda activate policyReader


- `fastapi`, `uvicorn[standard]`：Web API 框架与服务
- `python-multipart`：文件上传支持
- `pydantic`：配置与响应模型
- `python-dotenv`：从 `.env` 读取环境变量

> 包含视觉检索与大模型在内的完整依赖已在 `environment.yml` 中列出，
> 建议优先使用 `conda env create -f environment.yml` 创建 `policyReader` 环境。

#### `utils.py` 所需系统依赖（Word/HTML/URL → PDF）

`utils.py` 中的 `PDFConverter` 会调用系统级工具完成 Word / HTML / URL 到 PDF 的转换，主要依赖：

- **LibreOffice / soffice**（用于 Word → PDF）  
  - `WordConverter` 首选通过 `libreoffice` / `soffice` 命令行进行转换。
  - 在 Debian / Ubuntu 等发行版上，可通过：`sudo apt install libreoffice` 安装。

- **wkhtmltopdf**（用于 HTML / URL → PDF，配合 `pdfkit`）  
  - `HTMLConverter` 优先使用 `pdfkit`，其底层需要系统可用的 `wkhtmltopdf` 可执行文件。
  - 在 Debian / Ubuntu 上可通过：`sudo apt install wkhtmltopdf` 安装，或使用官方二进制安装包，并确保其在 `PATH` 中。

- **Poppler 工具链（如未使用 Conda 环境）**  
  - `visual_rag.py` 中的 `pdf2image` 依赖 Poppler（Conda 环境中已通过 `poppler` / `poppler-data` 提供）。
  - 若在非 Conda 环境中使用 `pdf2image`，需要在系统中安装 Poppler，例如：`sudo apt install poppler-utils`。

> 上述系统依赖不包含在 `pip` 包中，需在宿主操作系统或基础镜像中预先安装。

### 2. Node.js 前端环境

前端基于 Vite + React，需安装 Node.js（推荐 18+）：

```bash
cd policy_frontend_react
npm install
```

---

## 环境变量配置

在项目根目录创建 `.env` 文件，用于后端读取 API Key 等配置，例如：

```env
# Doubao（火山方舟）API Key
ARK_API_KEY=your_doubao_api_key_here

# OpenAI API Key（仅在选择 gpt4 模型时使用）
OPENAI_API_KEY=your_openai_api_key_here

# Qwen-VL vLLM 服务（仅在选择 qwen 模型时使用）
QWEN_VL_SERVER_URL=http://localhost:8001
QWEN_VL_MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
```

前端在参数配置面板中可以显式填写这些字段；若前端传入 `null`，后端会自动从上述环境变量中回填默认值。

---

## 启动方式

### 1. 启动 FastAPI 后端

在项目根目录执行（确保已激活 Python 环境）：

```bash
python -m uvicorn policy_api_fastapi:app --host 0.0.0.0 --port 8000
```

- 后端默认监听 `http://0.0.0.0:8000`
- 已开启 CORS，允许前端（默认 `http://127.0.0.1:5173`）访问 `/api/*`

### 2. 启动 React 前端

```bash
cd policy_frontend_react
npm run dev
```

默认访问地址：`http://127.0.0.1:5173`

Vite 已在 `vite.config.js` 中配置代理：

- 前端发起的 `/api/...` 请求会被代理到 `http://127.0.0.1:8000`，
  因此在代码中不需要写死后端地址。

---

## Web 界面使用说明

打开前端页面后，主要流程如下：

1. **上传 / 选择 政策文件**
   - 点击顶部工具栏「上传政策文件」按钮，弹出参数配置面板
   - 在「政策来源」区域：
     - 从下拉列表中选择已有政策文件（扫描 `policy_data/`）
     - 通过文件选择框上传本地文档（保存到 `policy_data/uploads/`）
     - 可选：输入政策网页 URL（每行一个），后端会抓取并转为 PDF

2. **配置模型与参数**
   - 选择视觉 LLM：`doubao` / `gpt4` / `qwen`
   - 选择视觉检索模型：`colpali` / `colqwen` / `nemo`
   - 设置 `top_k`、`force_reindex`、自定义 QA Prompt
   - 根据所选模型填写 / 留空对应的 API Key 或服务地址

3. **触发政策解读**
   - 点击「提取政策信息」按钮
   - 后端创建异步任务，前端开始轮询任务状态（queued → running → succeeded/failed）
   - 结果会按问题维度依次返回，界面各模块会陆续填充内容

4. **查看解读结果（左侧主区域）**

   - **【政策要点总览】**
     - 一句话结论：后端在 7 维度回答基础上，由文本 LLM 输出的单句总结
     - 基本信息：政策来源、发布时间、截止时间、任务状态

   - **【支持内容与申报规则】**
     - 支持对象：可申报主体列表
     - 不适用 / 负面清单：限制条件与红线条款
     - 扶持方式与资金规则：补贴标准、资金用途等
     - 核心申报条件：点击展开查看关键门槛
     - 申报材料清单：点击展开查看详细流程与材料

   - **【影响解读与行动建议】**
     - 政策影响：对财政支出、产业链、申报成本的影响要点
     - 申报窗口与截止时间：从时间维度回答中抽取
     - AI 行动建议：基于 7 维度回答，由文本 LLM 生成的 3–6 条可执行建议
     - 支持查看「原始结果 JSON」，方便调试与核对模型输出

右侧区域包括数字人播报占位、解读目录、关联政策列表等辅助模块。

---

## 后端 API 简要说明

所有接口前缀均为 `/api`：

- `GET /api/health`
  - 健康检查，返回 `{ "status": "ok" }`

- `GET /api/policy/files`
  - 扫描 `policy_data/` 目录下可用的政策文件
  - 返回：`{ files: ["policy_data/xxx.pdf", ...] }`

- `POST /api/policy/upload`
  - 上传一个或多个政策文件，保存到 `policy_data/uploads/`
  - 返回：`{ saved_paths: ["policy_data/uploads/xxx.pdf", ...] }`

- `POST /api/policy/jobs`
  - 创建政策解读任务
  - 请求体示例：

    ```jsonc
    {
      "inputs": ["policy_data/docs/example.pdf"],
      "config": {
        "llm_model": "doubao",       // 或 gpt4 / qwen
        "vision_retriever": "nemo",  // colpali / colqwen / nemo
        "top_k": 3,
        "force_reindex": false,
        "qa_prompt": "请基于给定政策文本，客观提取和归纳关键信息。请务必用中文回答问题",
        "doubao_api_key": null,       // 为空时后端从 .env 中回填
        "openai_api_key": null,
        "qwen_server_url": null,
        "qwen_model_name": null
      }
    }
    ```

  - 响应：`{ "job_id": "...", "status": "queued" }`

- `GET /api/policy/jobs/{job_id}`
  - 查询任务状态与当前累积结果
  - 响应结构大致为：

    ```jsonc
    {
      "job_id": "...",
      "status": "running | succeeded | failed",
      "error": null,
      "result": {
        "who": { "question": "...", "answer": "...", "analysis": "..." },
        "what": { ... },
        "how_much": { ... },
        "threshold": { ... },
        "compliance": { ... },
        "when": { ... },
        "how": { ... },
        "summary": { "answer": "一句话结论..." },
        "actions": { "answer": "若干条 AI 行动建议..." }
      }
    }
    ```

前端会在任务执行期间持续轮询该接口，并根据 `result` 中已有的 key 增量更新页面。

---

## 注意事项

- **GPU / 模型资源**：
  - 视觉检索（ColPali / NemoRetriever）和多模态模型通常需要 GPU 与较大显存，首次运行时会从 Hugging Face 或 Nvidia 下载模型权重。
  - 项目中已将 `HF_ENDPOINT` 设置为国内镜像（如需自定义，请修改 `policy_api_fastapi.py` 顶部配置）。

- **长文本与 markdown**：
  - 后端返回的 `answer` 多为 markdown 文本，前端使用 `react-markdown + remark-gfm` 渲染。
  - 若需要控制条目数量或样式，可在 `App.jsx` 中调整 `toItems` 或样式。

- **生产部署**：
  - 当前配置主要面向本地开发与 Demo；如需线上部署，请：
    - 使用 `uvicorn`/`gunicorn` + 反向代理（Nginx）部署 FastAPI
    - 使用 `npm run build` 构建前端静态资源并由静态服务器托管

---

## 许可证

项目暂未显式指定开源许可证，如需在生产环境或商业场景中使用，请先与作者确认相关授权与合规要求。

