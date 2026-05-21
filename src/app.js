(function () {
  const $ = (s) => document.querySelector(s);

  // ── File extensions → language map ──

  const EXT_LANG = {
    ts: 'typescript', tsx: 'typescript', js: 'javascript', jsx: 'javascript',
    rs: 'rust', c: 'c', cpp: 'c++', cc: 'c++', cxx: 'c++', h: 'c', hpp: 'c++',
    py: 'python', go: 'go', java: 'java', html: 'html', htm: 'html',
    css: 'css', sql: 'sql', sh: 'bash', bash: 'bash', toml: 'toml',
    yaml: 'yaml', yml: 'yaml', json: 'json', md: 'markdown', txt: 'text',
    rb: 'ruby', swift: 'swift', kt: 'kotlin', cs: 'csharp', lua: 'lua',
    zig: 'zig', asm: 'asm', s: 'asm', vue: 'vue', svelte: 'svelte',
    astro: 'astro', graphql: 'graphql', gql: 'graphql', proto: 'protobuf',
    cmake: 'cmake', mk: 'makefile', makefile: 'makefile',
    dockerfile: 'dockerfile', xml: 'xml', ini: 'ini', cfg: 'config',
    conf: 'config', env: 'env', gitignore: 'gitignore',
    scss: 'scss', sass: 'sass', less: 'less',
  };

  function extToLang(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    return EXT_LANG[ext] || 'text';
  }

  // ── Token estimation ──
  // ~3.2 chars per token for code (conservative)
  function estimateTokens(text) {
    return Math.ceil(text.length / 3.2);
  }

  function formatTokens(n) {
    if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
    return String(n);
  }

  // ── File state ──

  let uploadedFiles = []; // { id, name, content, language, tokens }
  let fileIdCounter = 0;
  let modelCtx = 4096; // updated from API
  let modelName = '';   // updated from API
  let modelsData = [];  // cached from /api/models for settings panel
  let currentMode = 'write'; // 'write' or 'review'

  // ── Mode toggle ──

  document.querySelectorAll('.mode-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.mode-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      currentMode = btn.dataset.mode;

      // Update UI hints based on mode
      const writeBtn = $('#write-btn');
      const descInput = $('#write-desc');
      if (currentMode === 'review') {
        writeBtn.textContent = 'Review →';
        descInput.placeholder = 'What should be reviewed? (e.g. "Check for bugs and performance issues")';
      } else {
        writeBtn.innerHTML = 'Write &rarr;';
        descInput.placeholder = 'Describe what you need...';
      }
    });
  });

  function addFile(name, content) {
    const language = extToLang(name);
    const tokens = estimateTokens(content);
    const id = ++fileIdCounter;
    uploadedFiles.push({ id, name, content, language, tokens });
    renderFileList();
    updateBudget();
  }

  function removeFile(id) {
    uploadedFiles = uploadedFiles.filter((f) => f.id !== id);
    renderFileList();
    updateBudget();
  }

  function clearFiles() {
    uploadedFiles = [];
    renderFileList();
    updateBudget();
  }

  // ── Relevance scoring ──
  // Sort files so the most relevant to the description and target language come first

  function scoreFile(file, description, targetLang) {
    let score = 0;
    // Same language as target
    if (file.language === targetLang) score += 10;
    // Filename mentioned in description
    const fnameBase = file.name.replace(/\.[^.]+$/, '').toLowerCase();
    const descLower = description.toLowerCase();
    if (fnameBase.length > 2 && descLower.includes(fnameBase)) score += 20;
    // Words from description appear in file content
    const words = descLower.split(/\s+/).filter((w) => w.length > 3);
    for (const w of words) {
      if (file.content.includes(w)) score += 2;
    }
    return score;
  }

  function getSortedFiles(description, targetLang) {
    return [...uploadedFiles]
      .map((f) => ({ ...f, _score: scoreFile(f, description, targetLang) }))
      .sort((a, b) => b._score - a._score);
  }

  // ── Render file list ──

  function renderFileList() {
    const list = $('#file-list');
    list.innerHTML = '';
    for (const f of uploadedFiles) {
      const el = document.createElement('div');
      el.className = 'file-entry';
      el.innerHTML = `
        <svg class="file-icon" width="12" height="12" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
        </svg>
        <span class="file-name" title="${esc(f.name)}">${esc(f.name)}</span>
        <span class="file-tokens">${formatTokens(f.tokens)} tok</span>
        <button class="file-remove" title="Remove" data-id="${f.id}">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </button>`;
      list.appendChild(el);
    }

    // Attach remove handlers
    list.querySelectorAll('.file-remove').forEach((btn) => {
      btn.onclick = () => removeFile(+btn.dataset.id);
    });
  }

  // ── Budget bar ──

  function updateBudget() {
    const bar = $('#budget-bar');
    const fill = $('#budget-fill');
    const label = $('#budget-label');
    const detail = $('#budget-detail');

    const totalFileTokens = uploadedFiles.reduce((s, f) => s + f.tokens, 0);
    const ctxTokens = estimateTokens($('#write-ctx').value || '');
    const descTokens = estimateTokens($('#write-desc').value || '');
    const systemTokens = 80; // approximate system prompt

    const inputUsed = totalFileTokens + ctxTokens + descTokens + systemTokens;
    const remaining = Math.max(0, modelCtx - inputUsed);
    const pct = modelCtx > 0 ? Math.min((inputUsed / modelCtx) * 100, 100) : 0;

    if (totalFileTokens === 0 && ctxTokens === 0) {
      bar.classList.add('hidden');
      return;
    }

    bar.classList.remove('hidden');
    fill.style.width = pct + '%';
    fill.classList.remove('warn', 'over');

    if (remaining < 256) {
      fill.classList.add('over');
      label.textContent = `${formatTokens(inputUsed)} input · ${formatTokens(remaining)} left for response`;
      detail.innerHTML = `<span class="budget-over">Context nearly full — files will be truncated or dropped</span>`;
    } else if (remaining < modelCtx * 0.15) {
      fill.classList.add('warn');
      label.textContent = `${formatTokens(inputUsed)} input · ${formatTokens(remaining)} left for response`;
      detail.textContent = modelName
        ? `${modelName} · ${formatTokens(modelCtx)} context`
        : `${formatTokens(modelCtx)} context`;
    } else {
      label.textContent = `${formatTokens(inputUsed)} input · ${formatTokens(remaining)} left for response`;
      detail.textContent = modelName
        ? `${modelName} · ${formatTokens(modelCtx)} context`
        : `${formatTokens(modelCtx)} context`;
    }
  }

  // Listen for typing in context/desc to update budget
  $('#write-desc').addEventListener('input', updateBudget);
  $('#write-ctx').addEventListener('input', updateBudget);

  // ── Drag & drop ──

  const dropArea = $('#drop-area');
  const fileInput = $('#file-input');

  function handleDroppedFiles(fileList) {
    for (const file of fileList) {
      // Skip files > 2MB
      if (file.size > 2 * 1024 * 1024) continue;
      const reader = new FileReader();
      reader.onload = () => addFile(file.name, reader.result);
      reader.readAsText(file);
    }
  }

  dropArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropArea.classList.add('drag-over');
  });

  dropArea.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropArea.classList.remove('drag-over');
  });

  dropArea.addEventListener('drop', (e) => {
    e.preventDefault();
    dropArea.classList.remove('drag-over');
    if (e.dataTransfer.files.length) handleDroppedFiles(e.dataTransfer.files);
  });

  // Stop clicks on the hidden input from bubbling back to dropArea
  fileInput.addEventListener('click', (e) => e.stopPropagation());

  dropArea.addEventListener('click', (e) => {
    // Don't re-trigger if the click came from the label (it already opens the picker)
    if (e.target.closest('.file-label')) return;
    fileInput.click();
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleDroppedFiles(fileInput.files);
    fileInput.value = ''; // reset so same file can be re-added
  });

  // ── Settings overlay ──

  $('#settings-btn').onclick = () => {
    $('#settings-overlay').classList.remove('hidden');
    refreshModels();
    refreshStatus();
  };
  $('#settings-close').onclick = () => $('#settings-overlay').classList.add('hidden');
  $('#settings-overlay').onclick = (e) => {
    if (e.target === $('#settings-overlay')) $('#settings-overlay').classList.add('hidden');
  };

  // ── Model management ──

  async function refreshModels() {
    try {
      const d = await fetch('/api/models').then((r) => r.json());
      modelsData = d.models || [];

      // Populate main model dropdown
      const sel = $('#model-select');
      sel.innerHTML = '';
      for (const m of modelsData) {
        const o = document.createElement('option');
        o.value = m.filename;
        o.textContent = `${m.name} [${m.family}]`;
        if (m.filename === d.active) o.selected = true;
        sel.appendChild(o);
      }

      // Populate draft model dropdown
      const draftSel = $('#draft-select');
      draftSel.innerHTML = '<option value="">None (disabled)</option>';
      const candidates = d.draft_candidates || [];
      for (const fname of candidates.sort()) {
        const o = document.createElement('option');
        o.value = fname;
        o.textContent = fname.replace('.gguf', '');
        draftSel.appendChild(o);
      }
      // Sync current draft config
      if (d.draft) {
        draftSel.value = d.draft.model || '';
        $('#p-draft-max').value = d.draft.max || 10;
        $('#p-draft-ngl').value = d.draft.ngl ?? 99;
        updateDraftInfo(d.draft.model, d.draft.max);
      }

      // Sync active model name
      const activeModel = modelsData.find((m) => m.filename === d.active);
      if (activeModel) {
        modelName = activeModel.name;
      } else if (d.active) {
        modelName = d.active.split('.')[0];
      } else {
        modelName = '';
      }
      if (d.params) {
        $('#p-ngl').value = d.params.ngl;
        $('#p-ctx').value = d.params.ctx;
        $('#p-temp').value = d.params.temp;
        $('#p-topk').value = d.params.top_k;
        $('#p-topp').value = d.params.top_p;
        $('#p-rp').value = d.params.repeat_penalty;
        modelCtx = d.params.ctx || 4096;
      }
      updateBudget();
      updateBadge(d.llama);
      updateLlamaStatus(d.llama);
    } catch (e) {
      console.error('refreshModels:', e);
    }
  }

  function updateDraftInfo(draftModel, draftMax) {
    const el = $('#draft-info');
    if (!draftModel) {
      el.textContent = 'Speculative decoding disabled. No VRAM used for draft KV cache.';
    } else {
      el.textContent = `Draft: ${draftModel.replace('.gguf', '')}\n`
        + `Proposes up to ${draftMax || 10} tokens per step.\n`
        + `Note: draft model shares the main context window and allocates its own KV cache.`;
    }
  }

  // When user selects a different model in settings, fill param fields
  // with that model's configured values so they can see/adjust before loading.
  $('#model-select').addEventListener('change', () => {
    const selected = $('#model-select').value;
    const m = modelsData.find((x) => x.filename === selected);
    if (m) {
      $('#p-ngl').value = m.gpu_layers;
      $('#p-ctx').value = m.context_size;
      $('#p-temp').value = m.temperature;
      $('#p-topk').value = m.top_k;
      $('#p-topp').value = m.top_p;
      $('#p-rp').value = m.repeat_penalty;
    }
  });

  // Preview draft info when selection changes (before loading)
  $('#draft-select').addEventListener('change', () => {
    const v = $('#draft-select').value;
    updateDraftInfo(v, +$('#p-draft-max').value);
  });

  function updateBadge(llama) {
    const b = $('#model-badge');
    if (!llama || llama.status === 'stopped') {
      b.textContent = 'Stopped';
      b.className = 'badge badge-idle';
    } else if (llama.status === 'ready') {
      b.textContent = (llama.model || '').split('.')[0].slice(0, 20);
      b.className = 'badge badge-ready';
    } else if (llama.status === 'starting') {
      b.textContent = 'Loading...';
      b.className = 'badge badge-loading';
    } else {
      b.textContent = 'Error';
      b.className = 'badge badge-error';
    }
  }

  function updateLlamaStatus(llama) {
    if (!llama) return;
    let s = `Status: ${llama.status}`;
    if (llama.model) s += `\nModel: ${llama.model}`;
    if (llama.pid) s += `\nPID: ${llama.pid}`;
    if (llama.error) s += `\nError: ${llama.error}`;
    $('#llama-status').textContent = s;
  }

  $('#load-btn').onclick = async () => {
    const model = $('#model-select').value;
    if (!model) return;
    const draftModel = $('#draft-select').value; // empty string = disabled
    updateBadge({ status: 'starting' });
    const draftLabel = draftModel ? ` + draft ${draftModel.replace('.gguf', '')}` : '';
    $('#llama-status').textContent = `Loading model${draftLabel}... (may take 1-2 minutes)`;
    try {
      const d = await fetch('/api/load', {
        method: 'POST',
        body: JSON.stringify({
          model,
          ngl: +$('#p-ngl').value,
          ctx: +$('#p-ctx').value,
          flash_attn: true,
          temp: +$('#p-temp').value,
          top_k: +$('#p-topk').value,
          top_p: +$('#p-topp').value,
          repeat_penalty: +$('#p-rp').value,
          draft_model: draftModel,
          draft_max: +$('#p-draft-max').value || 10,
          gpu_layers_draft: +$('#p-draft-ngl').value,
        }),
      }).then((r) => r.json());
      updateBadge(d.llama || { status: d.error ? 'error' : 'stopped' });
      updateLlamaStatus(d.llama);
      if (d.error) $('#llama-status').textContent = d.error;
      // Sync ctx and model name from what was just loaded
      modelCtx = +$('#p-ctx').value || 4096;
      const sel = $('#model-select');
      modelName = sel.options[sel.selectedIndex]?.textContent || model.split('.')[0];
      updateDraftInfo(draftModel, +$('#p-draft-max').value);
      updateBudget();
    } catch (e) {
      updateBadge({ status: 'error' });
      $('#llama-status').textContent = String(e);
    }
  };

  $('#stop-btn').onclick = async () => {
    await fetch('/api/stop', { method: 'POST' });
    updateBadge({ status: 'stopped' });
    $('#llama-status').textContent = 'Stopped';
  };

  $('#save-params-btn').onclick = () =>
    fetch('/api/params', {
      method: 'POST',
      body: JSON.stringify({
        temp: +$('#p-temp').value,
        top_k: +$('#p-topk').value,
        top_p: +$('#p-topp').value,
        repeat_penalty: +$('#p-rp').value,
      }),
    });

  async function refreshStatus() {
    try {
      const d = await fetch('/api/status').then((r) => r.json());
      $('#usage-info').textContent =
        `Requests: ${d.requests}  Tokens: ${d.tokens_session}\nModel: ${d.model || 'none'}`;
      // Sync context size from status
      if (d.ctx) {
        modelCtx = d.ctx;
        updateBudget();
      }
    } catch (_) {}
  }

  // ── Streaming write ──

  let abortCtrl = null;

  $('#write-btn').onclick = doWrite;
  $('#write-desc').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      doWrite();
    }
  });

  async function doWrite() {
    const desc = $('#write-desc').value.trim();
    if (!desc) return;

    // Cancel any in-flight request
    if (abortCtrl) abortCtrl.abort();
    abortCtrl = new AbortController();

    const output = $('#output');
    const copyBtn = $('#copy-btn');
    const statsEl = $('#stats');
    const ctxInfo = $('#context-info');
    const isReview = currentMode === 'review';

    const loadingLabel = isReview ? 'Reviewing code...' : 'Writing code...';
    output.innerHTML = `<div class="loading"><div class="spinner"></div>${loadingLabel}</div>`;
    copyBtn.classList.add('hidden');
    statsEl.textContent = '';
    ctxInfo.classList.add('hidden');
    ctxInfo.innerHTML = '';

    // Build sorted files array
    const lang = $('#lang-select').value;
    const sorted = getSortedFiles(desc, lang);
    const filesPayload = sorted.map((f) => ({
      name: f.name,
      content: f.content,
      language: f.language,
    }));

    try {
      const res = await fetch('/api/write', {
        method: 'POST',
        signal: abortCtrl.signal,
        body: JSON.stringify({
          description: desc,
          language: lang,
          mode: currentMode,
          context: $('#write-ctx').value.trim() || undefined,
          files: filesPayload.length > 0 ? filesPayload : undefined,
        }),
      });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      let fullText = '';
      let started = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        // Process complete SSE messages (delimited by \n\n)
        let idx;
        while ((idx = buf.indexOf('\n\n')) !== -1) {
          const line = buf.slice(0, idx);
          buf = buf.slice(idx + 2);

          if (!line.startsWith('data: ')) continue;
          let data;
          try { data = JSON.parse(line.slice(6)); } catch { continue; }

          if (data.error) {
            output.innerHTML = `<div class="error-msg">${esc(data.error)}</div>`;
            return;
          }

          // Handle context_info event from server
          if (data.context_info) {
            const ci = data.context_info;
            let parts = [];
            if (ci.files_included && ci.files_included.length) {
              parts.push(`<span class="ctx-included">${ci.files_included.length} files</span>`);
            }
            if (ci.files_truncated && ci.files_truncated.length) {
              parts.push(`<span class="ctx-dropped">${ci.files_truncated.length} truncated</span>`);
            }
            if (ci.files_dropped && ci.files_dropped.length) {
              parts.push(`<span class="ctx-dropped">${ci.files_dropped.length} dropped</span>`);
            }
            if (ci.remaining_tokens != null) {
              parts.push(`${formatTokens(ci.remaining_tokens)} left for response`);
            }
            if (parts.length) {
              ctxInfo.innerHTML = parts.join(' · ');
              ctxInfo.classList.remove('hidden');
              const lines = [
                ...(ci.files_included || []).map((n) => '✓ ' + n),
                ...(ci.files_truncated || []).map((n) => '⚠ ' + n),
                ...(ci.files_dropped || []).map((n) => '✗ ' + n),
              ];
              ctxInfo.title = lines.join('\n');
            }
            continue;
          }

          if (data.token) {
            if (!started) {
              if (isReview) {
                output.innerHTML = '<div class="review-block streaming-cursor" id="stream-review"></div>';
              } else {
                output.innerHTML = '<pre class="code-block streaming-cursor"><code id="stream-code"></code></pre>';
              }
              started = true;
            }
            fullText += data.token;
            if (isReview) {
              // Live-render with basic markdown code fence support
              $('#stream-review').innerHTML = renderReview(fullText);
            } else {
              $('#stream-code').textContent = fullText;
            }
            output.scrollTop = output.scrollHeight;
          }

          if (data.done) {
            const cursor = output.querySelector('.streaming-cursor');
            if (cursor) cursor.classList.remove('streaming-cursor');

            // Final render pass for review mode
            if (isReview && fullText) {
              const el = $('#stream-review');
              if (el) el.innerHTML = renderReview(fullText);
            }

            const secs = ((data.elapsed_ms || 0) / 1000).toFixed(1);
            statsEl.textContent = `${data.tokens || 0} tok · ${secs}s`;

            if (fullText) {
              copyBtn.classList.remove('hidden');
              copyBtn.onclick = () => {
                navigator.clipboard.writeText(fullText);
                copyBtn.textContent = 'Copied!';
                setTimeout(() => (copyBtn.textContent = 'Copy'), 1500);
              };
            }
          }
        }
      }

      // If nothing streamed at all
      if (!started && !fullText) {
        output.innerHTML = '<div class="placeholder-msg">No output received</div>';
      }
    } catch (e) {
      if (e.name !== 'AbortError') {
        output.innerHTML = `<div class="error-msg">Error: ${esc(String(e))}</div>`;
      }
    } finally {
      abortCtrl = null;
    }
  }

  // Simple markdown-ish renderer for review output:
  // Converts ```...``` fenced code blocks and escapes the rest as HTML.
  function renderReview(text) {
    const parts = text.split(/(```[\s\S]*?```|```[\s\S]*$)/);
    let html = '';
    for (const part of parts) {
      if (part.startsWith('```')) {
        // Strip the opening ```lang and closing ```
        const inner = part.replace(/^```[^\n]*\n?/, '').replace(/\n?```$/, '');
        html += `<code class="review-code">${esc(inner)}</code>`;
      } else {
        html += esc(part);
      }
    }
    return html;
  }

  // ── Helpers ──

  function esc(s) {
    return (s || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // ── Init: sync model status + params ──

  (async () => {
    try {
      // refreshModels fetches /api/models which returns params, ctx, active model,
      // llama status — everything needed to sync the UI on page load
      await refreshModels();
    } catch (_) {}
  })();
})();
