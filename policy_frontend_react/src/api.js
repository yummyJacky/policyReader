async function jsonFetch(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {})
    }
  })

  if (!res.ok) {
    let detail = ''
    try {
      const data = await res.json()
      detail = data?.detail ? String(data.detail) : JSON.stringify(data)
    } catch {
      detail = await res.text()
    }
    throw new Error(detail || `HTTP ${res.status}`)
  }

  return res.json()
}

export async function health() {
  return jsonFetch('/api/health')
}

export async function listPolicyFiles() {
  return jsonFetch('/api/policy/files')
}

export async function uploadPolicyFiles(files) {
  const form = new FormData()
  for (const f of files) form.append('files', f)

  const res = await fetch('/api/policy/upload', {
    method: 'POST',
    body: form
  })

  if (!res.ok) {
    let detail = ''
    try {
      const data = await res.json()
      detail = data?.detail ? String(data.detail) : JSON.stringify(data)
    } catch {
      detail = await res.text()
    }
    throw new Error(detail || `HTTP ${res.status}`)
  }

  return res.json()
}

export async function createJob(payload) {
  return jsonFetch('/api/policy/jobs', {
    method: 'POST',
    body: JSON.stringify(payload)
  })
}

export async function getJob(jobId) {
  return jsonFetch(`/api/policy/jobs/${jobId}`)
}
