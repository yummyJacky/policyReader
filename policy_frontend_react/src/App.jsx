import React, { useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import Card from './components/Card.jsx'
import { createJob, getJob, listPolicyFiles, uploadPolicyFiles } from './api.js'

function toItems(text, maxItems = 6) {
  const s = (text || '').trim()
  if (!s) return []
  let parts = s.split(/\r?\n+/).map((x) => x.trim()).filter(Boolean)
  if (parts.length <= 1) {
    parts = s.split(/[。；;]\s*/).map((x) => x.trim()).filter(Boolean)
  }
  parts = parts.map((x) => x.replace(/^[-•*\s]+/, '').trim()).filter(Boolean)
  return parts
}

function oneLiner(text, maxLen = 120) {
  const s = (text || '').trim()
  if (!s) return '暂无'
  let first = s.split(/[。\n]/)[0].trim()
  if (!first) first = s
  if (first.length > maxLen) return first.slice(0, maxLen - 1).trimEnd() + '…'
  return first
}

function extractDates(text) {
  const s = text || ''
  const m = s.match(/\d{4}[-/.]\d{1,2}[-/.]\d{1,2}/g) || []
  const out = []
  const seen = new Set()
  for (const d of m) {
    if (!seen.has(d)) {
      out.push(d)
      seen.add(d)
    }
  }
  return out
}

function Markdown({ children }) {
  const content = (children || '').toString()
  if (!content.trim()) return null
  return <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
}

export default function App() {
  const [files, setFiles] = useState([])
  const [selectedFiles, setSelectedFiles] = useState([])
  const [urls, setUrls] = useState('')
  const [uploadedPaths, setUploadedPaths] = useState([])

  const [settingsOpen, setSettingsOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const [jobId, setJobId] = useState('')
  const [jobStatus, setJobStatus] = useState('')
  const [result, setResult] = useState(null)

  const [config, setConfig] = useState({
    llm_model: 'doubao',
    vision_retriever: 'nemo',
    top_k: 3,
    force_reindex: false,
    qa_prompt: '请基于给定政策文本，客观提取和归纳关键信息。请务必用中文回答问题。',
    doubao_api_key: '',
    openai_api_key: '',
    qwen_server_url: '',
    qwen_model_name: 'Qwen/Qwen2.5-VL-7B-Instruct'
  })

  const pollTimer = useRef(null)

  useEffect(() => {
    ;(async () => {
      try {
        const data = await listPolicyFiles()
        setFiles(data.files || [])
      } catch (e) {
        setError(String(e?.message || e))
      }
    })()
  }, [])

  useEffect(() => {
    if (!jobId) return

    async function poll() {
      try {
        const data = await getJob(jobId)
        setJobStatus(data.status)
        setResult(data.result || null)
        if (data.status === 'succeeded') {
          setBusy(false)
          clearInterval(pollTimer.current)
          pollTimer.current = null
        }
        if (data.status === 'failed') {
          setBusy(false)
          setError(data.error || '任务失败')
          clearInterval(pollTimer.current)
          pollTimer.current = null
        }
      } catch (e) {
        setBusy(false)
        setError(String(e?.message || e))
        clearInterval(pollTimer.current)
        pollTimer.current = null
      }
    }

    poll()
    pollTimer.current = setInterval(poll, 1200)

    return () => {
      if (pollTimer.current) {
        clearInterval(pollTimer.current)
        pollTimer.current = null
      }
    }
  }, [jobId])

  const urlInputs = useMemo(() => {
    return urls
      .split(/\r?\n+/)
      .map((x) => x.trim())
      .filter(Boolean)
  }, [urls])

  const inputs = useMemo(() => {
    return [...selectedFiles, ...uploadedPaths, ...urlInputs]
  }, [selectedFiles, uploadedPaths, urlInputs])

  const currentTitle = useMemo(() => {
    if (!inputs.length) return '未选择政策'
    const first = inputs[0]
    try {
      return String(first).split('/').pop()
    } catch {
      return String(first)
    }
  }, [inputs])

  async function onUploadFiles(fileList) {
    setError('')
    if (!fileList || fileList.length === 0) return

    try {
      setBusy(true)
      const data = await uploadPolicyFiles(Array.from(fileList))
      const saved = data.saved_paths || []
      setUploadedPaths((prev) => Array.from(new Set([...prev, ...saved])))
      const refreshed = await listPolicyFiles()
      setFiles(refreshed.files || [])
    } catch (e) {
      setError(String(e?.message || e))
    } finally {
      setBusy(false)
    }
  }

  async function onExtract() {
    setError('')
    setResult(null)

    if (!inputs.length) {
      setError('请先上传/选择政策文件或输入 URL')
      setSettingsOpen(true)
      return
    }

    try {
      setBusy(true)
      setJobStatus('queued')
      const cfgToSend = {
        ...config,
        doubao_api_key: config.doubao_api_key?.trim() ? config.doubao_api_key : null,
        openai_api_key: config.openai_api_key?.trim() ? config.openai_api_key : null,
        qwen_server_url: config.qwen_server_url?.trim() ? config.qwen_server_url : null,
        qwen_model_name: config.qwen_model_name?.trim() ? config.qwen_model_name : null
      }
      const payload = {
        inputs,
        config: cfgToSend
      }
      const data = await createJob(payload)
      setJobId(data.job_id)
      setJobStatus(data.status)
    } catch (e) {
      setBusy(false)
      setError(String(e?.message || e))
    }
  }

  const summary = useMemo(() => {
    if (!result) return { conclusion: '', bullets: [], dates: [] }

    const hasSummary = !!result.summary
    const hasWhat = !!result.what
    const hasThreshold = !!result.threshold
    const hasCompliance = !!result.compliance
    const hasWhen = !!result.when

    const conclusion = hasSummary ? oneLiner(result.summary?.answer || '') : ''

    let bullets = []
    if (hasThreshold && hasCompliance) {
      const bulletText = `${result.threshold?.answer || ''}\n${result.compliance?.answer || ''}`
      bullets = toItems(bulletText, 5)
    }

    const dates = hasWhen ? extractDates(result.when?.answer || '') : []

    return { conclusion, bullets, dates }
  }, [result])

  const support = useMemo(() => {
    if (!result) return { who: [], ban: [], money: [], materials: [], thresholds: [] }

    const hasWho = !!result.who
    const hasCompliance = !!result.compliance
    const hasHowMuch = !!result.how_much
    const hasWhat = !!result.what
    const hasHow = !!result.how
    const hasThreshold = !!result.threshold

    return {
      who: hasWho ? toItems(result.who?.answer || '', 6) : [],
      ban: hasCompliance ? toItems(result.compliance?.answer || '', 5) : [],
      money:
        hasHowMuch && hasWhat
          ? toItems(`${result.how_much?.answer || ''}\n${result.what?.answer || ''}`, 6)
          : [],
      materials: hasHow ? toItems(result.how?.answer || '', 10) : [],
      thresholds: hasThreshold ? toItems(result.threshold?.answer || '', 4) : []
    }
  }, [result])

  const impact = useMemo(() => {
    if (!result) return { impactItems: [], actionItems: [], dates: [] }

    const hasWhat = !!result.what
    const hasActions = !!result.actions
    const hasWhen = !!result.when

    return {
      impactItems: hasWhat ? toItems(result.what?.answer || '', 3) : [],
      actionItems: hasActions ? toItems(result.actions?.answer || '', 6) : [],
      dates: hasWhen ? extractDates(result.when?.answer || '') : []
    }
  }, [result])

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <div className="emblem" />
          <div className="title">农业政策智能解读平台</div>
        </div>
        <div className="search">
          <span className="placeholder">输入政策标题、文号、发文机关等内容进行检索</span>
        </div>
        <div className="headerActions">
          <div className="icon">🔔</div>
          <div className="icon">👤</div>
        </div>
      </header>

      <div className="toolbar">
        <div className="toolbarLeft">
          <button className="btn" onClick={() => setSettingsOpen((v) => !v)} disabled={busy}>
            上传政策文件
          </button>
          <button className="btn primary" onClick={onExtract} disabled={busy}>
            {busy ? '处理中…' : '提取政策信息'}
          </button>
        </div>
        <div className="toolbarCenter">
          <div className="current">
            <b>当前解读：</b>
            {currentTitle}
          </div>
        </div>
        <div className="toolbarRight">
          <span className="tag">官方来源</span>
        </div>
      </div>

      {settingsOpen && (
        <div className="modalMask" onClick={() => !busy && setSettingsOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modalHeader">
              <div className="modalTitle">上传政策文件 / 输入链接 / 参数配置</div>
              <button className="btn" onClick={() => setSettingsOpen(false)} disabled={busy}>
                关闭
              </button>
            </div>
            <div className="modalBody">
              <div className="modalGrid">
                <div className="modalPanel">
                  <div className="panelTitle">政策来源</div>

                  <label className="fieldLabel">选择已有政策文件（policy_data）</label>
                  <select
                    className="select"
                    multiple
                    value={selectedFiles}
                    onChange={(e) => {
                      const opts = Array.from(e.target.selectedOptions).map((o) => o.value)
                      setSelectedFiles(opts)
                    }}
                  >
                    {files.map((f) => (
                      <option key={f} value={f}>
                        {f}
                      </option>
                    ))}
                  </select>

                  <label className="fieldLabel">上传政策文件（保存到 policy_data/uploads）</label>
                  <input
                    className="input"
                    type="file"
                    multiple
                    onChange={(e) => onUploadFiles(e.target.files)}
                    disabled={busy}
                  />

                  {uploadedPaths.length > 0 && (
                    <div className="hint">已上传：{uploadedPaths.map((p) => p.split('/').pop()).join('、')}</div>
                  )}

                  <label className="fieldLabel">政策网页 URL（每行一个，可选）</label>
                  <textarea className="textarea" value={urls} onChange={(e) => setUrls(e.target.value)} placeholder="https://www.moa.gov.cn/..." />

                  <div className="hint">本次输入：{inputs.length ? inputs.length : 0} 条</div>
                </div>

                <div className="modalPanel">
                  <div className="panelTitle">模型与参数</div>

                  <div className="row">
                    <div className="col">
                      <label className="fieldLabel">视觉 LLM 模型</label>
                      <select className="select" value={config.llm_model} onChange={(e) => setConfig((p) => ({ ...p, llm_model: e.target.value }))}>
                        <option value="doubao">doubao</option>
                        <option value="gpt4">gpt4</option>
                        <option value="qwen">qwen</option>
                      </select>
                    </div>
                    <div className="col">
                      <label className="fieldLabel">视觉检索模型</label>
                      <select className="select" value={config.vision_retriever} onChange={(e) => setConfig((p) => ({ ...p, vision_retriever: e.target.value }))}>
                        <option value="colpali">colpali</option>
                        <option value="colqwen">colqwen</option>
                        <option value="nemo">nemo</option>
                      </select>
                    </div>
                  </div>

                  <div className="row">
                    <div className="col">
                      <label className="fieldLabel">Top-K</label>
                      <input
                        className="input"
                        type="number"
                        min={1}
                        max={10}
                        value={config.top_k}
                        onChange={(e) => setConfig((p) => ({ ...p, top_k: Number(e.target.value) || 5 }))}
                      />
                    </div>
                    <div className="col">
                      <label className="fieldLabel">force_reindex</label>
                      <div className="checkboxRow">
                        <input
                          type="checkbox"
                          checked={config.force_reindex}
                          onChange={(e) => setConfig((p) => ({ ...p, force_reindex: e.target.checked }))}
                        />
                        <span className="hint">强制重建索引</span>
                      </div>
                    </div>
                  </div>

                  <label className="fieldLabel">Prompt</label>
                  <textarea className="textarea" value={config.qa_prompt} onChange={(e) => setConfig((p) => ({ ...p, qa_prompt: e.target.value }))} />

                  <div className="divider" />

                  <label className="fieldLabel">Doubao ARK_API_KEY</label>
                  <input className="input" type="password" value={config.doubao_api_key} onChange={(e) => setConfig((p) => ({ ...p, doubao_api_key: e.target.value }))} />

                  <label className="fieldLabel">OpenAI API Key</label>
                  <input className="input" type="password" value={config.openai_api_key} onChange={(e) => setConfig((p) => ({ ...p, openai_api_key: e.target.value }))} />

                  <label className="fieldLabel">Qwen-VL vLLM 服务地址</label>
                  <input className="input" value={config.qwen_server_url} onChange={(e) => setConfig((p) => ({ ...p, qwen_server_url: e.target.value }))} placeholder="http://localhost:8001" />

                  <label className="fieldLabel">Qwen-VL 模型名称</label>
                  <input className="input" value={config.qwen_model_name} onChange={(e) => setConfig((p) => ({ ...p, qwen_model_name: e.target.value }))} />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {error && <div className="alert">{error}</div>}

      <main className="main">
        <div className="left">
          <Card title="【政策要点总览】" right={<span className="tag">官方来源</span>}>
            <div className="oneLine">
              一句话结论：
              {jobStatus === 'succeeded' && summary.conclusion
                ? summary.conclusion
                : '请先上传政策文件并点击“提取政策信息”'}
            </div>

            <div className="kv">
              <span>
                政策来源：<span className="pill">{currentTitle}</span>
              </span>
              <span>
                发布时间：<span className="pill">{summary.dates[0] || '-'}</span>
              </span>
              <span>
                截止时间：<span className="pill">{summary.dates[1] || '-'}</span>
              </span>
              {jobStatus && (
                <span>
                  任务状态：<span className="pill">{jobStatus}</span>
                </span>
              )}
            </div>
          </Card>

          <Card title="【支持内容与申报规则】">
            <div className="grid3">
              <div className="panel">
                <div className="panelTitle">支持对象</div>
                <ul className="ul">
                  {(support.who.length ? support.who : ['（提取后展示支持对象）']).map((x, idx) => (
                    <li key={idx}>
                       <Markdown>{x}</Markdown>
                    </li>
                  ))}
                </ul>
              </div>
              <div className="panel">
                <div className="panelTitle">不适用 / 负面清单</div>
                <ul className="ul">
                  {(support.ban.length ? support.ban : ['（提取后展示不适用情形）']).map((x, idx) => (
                    <li key={idx}>
                       <Markdown>{x}</Markdown>
                    </li>
                  ))}
                </ul>
              </div>
              <div className="panel">
                <div className="panelTitle">扶持方式与资金规则</div>
                <ul className="ul">
                  {(support.money.length ? support.money : ['（提取后展示扶持方式与资金规则）']).map((x, idx) => (
                    <li key={idx}>
                       <Markdown>{x}</Markdown>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="grid2" style={{ marginTop: 12 }}>
              <div className="panel">
                <details className="details">
                  <summary className="panelTitle">核心申报条件（点击展开）</summary>
                  <div className="detailsBody">
                    {(support.thresholds.length ? support.thresholds : ['提取后将展示核心申报条件。']).map((x, idx) => (
                      <div key={idx} className="detailLine">
                        <Markdown>{x}</Markdown>
                      </div>
                    ))}
                  </div>
                </details>
              </div>

              <div className="panel">
                <details className="details">
                  <summary className="panelTitle">申报材料清单（点击展开）</summary>
                  <div className="detailsBody">
                    {(support.materials.length ? support.materials : ['提取后将展示申报流程与所需材料。']).map((x, idx) => (
                      <div key={idx} className="detailLine">
                        <Markdown>{x}</Markdown>
                      </div>
                    ))}
                  </div>
                </details>
              </div>
            </div>
          </Card>

          <Card title="【影响解读与行动建议】">
            <div className="grid2">
              <div className="panel">
                <div className="panelTitle">政策影响</div>
                <div className="muted">对财政支出、产业链、申报成本等的影响（示意）</div>

                <div style={{ marginTop: 10 }}>
                  <div className="muted">对财政支出：</div>
                  <div className="segbar">
                    <div className="seg on" />
                    <div className="seg on" />
                    <div className="seg on" />
                    <div className="seg" />
                    <div className="seg" />
                  </div>
                  <div className="muted" style={{ marginTop: 10 }}>
                    对产业链：
                  </div>
                  <div className="segbar">
                    <div className="seg on" />
                    <div className="seg on" />
                    <div className="seg on" />
                    <div className="seg on" />
                    <div className="seg" />
                  </div>
                </div>

                <div className="kv">
                  <span>
                    申报窗口：<span className="pill">{impact.dates[0] || '-'}</span>
                  </span>
                  <span>
                    截止：<span className="pill">{impact.dates[1] || '-'}</span>
                  </span>
                </div>

                <ul className="ul">
                  {(impact.impactItems.length ? impact.impactItems : ['（提取后展示政策影响与适用范围）']).map((x, idx) => (
                    <li key={idx}>
                      <Markdown>{x}</Markdown>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="panel">
                <div className="panelTitle">AI 行动建议</div>
                <div className="muted">基于政策要点与申报规则生成的行动建议（示意）</div>
                <ul className="ul" style={{ marginTop: 8 }}>
                  {(impact.actionItems.length ? impact.actionItems : ['（提取后展示可执行行动建议）']).map((x, idx) => (
                    <li key={idx}>
                      <Markdown>{x}</Markdown>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {result && (
              <details className="details" style={{ marginTop: 12 }}>
                <summary>查看原始结果 JSON</summary>
                <pre className="json">{JSON.stringify(result, null, 2)}</pre>
              </details>
            )}
          </Card>
        </div>

        <div className="right">
          <Card title="数字人播报">
            <div className="avatar">数字人形象占位</div>
            <button className="btn primary" style={{ width: '100%' }} disabled>
              ▶ 播放解读（占位）
            </button>
            <div className="radioRow">
              <label className="radio">
                <input type="radio" name="mode" defaultChecked /> 1分钟快读
              </label>
              <label className="radio">
                <input type="radio" name="mode" /> 3分钟深度解读
              </label>
            </div>
          </Card>

          <Card title="解读目录">
            <ul className="ul">
              <li>政策要点总览</li>
              <li>支持内容与申报规则</li>
              <li>影响解读与行动建议</li>
            </ul>
          </Card>

          <Card title="关联政策">
            {inputs.length ? (
              <ul className="ul">
                {inputs.slice(0, 6).map((x, idx) => (
                  <li key={idx}>{String(x)}</li>
                ))}
              </ul>
            ) : (
              <div className="muted">上传或选择多个政策文件后，将在此展示关联政策。</div>
            )}
          </Card>
        </div>
      </main>
    </div>
  )
}
