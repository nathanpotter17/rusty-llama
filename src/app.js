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
  function estimateTokens(text) {
    return Math.ceil(text.length / 3.2);
  }

  function formatTokens(n) {
    if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
    return String(n);
  }

  // ── File state ──

  let contextFiles = []; // { id, name, content, language, tokens } — sent with prompt
  let ragFiles = [];     // { id, name, content, language, tokens } — queued for indexing
  let fileIdCounter = 0;
  let fileDest = 'context'; // 'context' or 'rag'
  let modelCtx = 4096;
  let modelName = '';
  let modelsData = [];
  let currentMode = 'write';
  let ragIndexed = 0;
  let ragEnabled = false;

  // ── Destination toggle ──

  document.querySelectorAll('.dest-btn').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      document.querySelectorAll('.dest-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      fileDest = btn.dataset.dest;
    });
  });

  // ── Mode toggle ──

  document.querySelectorAll('.mode-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.mode-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      currentMode = btn.dataset.mode;

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
    const entry = { id, name, content, language, tokens };
    if (fileDest === 'rag') {
      ragFiles.push(entry);
    } else {
      contextFiles.push(entry);
    }
    renderFileLists();
    updateBudget();
  }

  function removeContextFile(id) {
    contextFiles = contextFiles.filter((f) => f.id !== id);
    renderFileLists();
    updateBudget();
  }

  function removeRagFile(id) {
    ragFiles = ragFiles.filter((f) => f.id !== id);
    renderFileLists();
  }

  // ── Relevance scoring ──

  function scoreFile(file, description, targetLang) {
    let score = 0;
    if (file.language === targetLang) score += 10;
    const fnameBase = file.name.replace(/\.[^.]+$/, '').toLowerCase();
    const descLower = description.toLowerCase();
    if (fnameBase.length > 2 && descLower.includes(fnameBase)) score += 20;
    const words = descLower.split(/\s+/).filter((w) => w.length > 3);
    for (const w of words) {
      if (file.content.includes(w)) score += 2;
    }
    return score;
  }

  function getSortedFiles(description, targetLang) {
    return [...contextFiles]
      .map((f) => ({ ...f, _score: scoreFile(f, description, targetLang) }))
      .sort((a, b) => b._score - a._score);
  }

  // ── Render file lists ──

  function renderFileLists() {
    renderFileListInto('#context-file-list', contextFiles, removeContextFile);
    renderFileListInto('#rag-file-list', ragFiles, removeRagFile);

    // Show/hide sections
    const ctxSection = $('#context-file-section');
    const ragSection = $('#rag-file-section');
    if (contextFiles.length > 0) { ctxSection.classList.remove('hidden'); }
    else { ctxSection.classList.add('hidden'); }
    if (ragFiles.length > 0) { ragSection.classList.remove('hidden'); }
    else { ragSection.classList.add('hidden'); }
  }

  function renderFileListInto(selector, files, removeFn) {
    const list = $(selector);
    list.innerHTML = '';
    for (const f of files) {
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
    list.querySelectorAll('.file-remove').forEach((btn) => {
      btn.onclick = () => removeFn(+btn.dataset.id);
    });
  }

  // ── Budget bar ──

  function updateBudget() {
    const bar = $('#budget-bar');
    const fill = $('#budget-fill');
    const label = $('#budget-label');
    const detail = $('#budget-detail');

    const totalFileTokens = contextFiles.reduce((s, f) => s + f.tokens, 0);
    const ctxTokens = estimateTokens($('#write-ctx').value || '');
    const descTokens = estimateTokens($('#write-desc').value || '');
    const systemTokens = 80;

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

  $('#write-desc').addEventListener('input', updateBudget);
  $('#write-ctx').addEventListener('input', updateBudget);

  // ── Drag & drop ──

  const dropArea = $('#drop-area');
  const fileInput = $('#file-input');

  function handleDroppedFiles(fileList) {
    for (const file of fileList) {
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

  fileInput.addEventListener('click', (e) => e.stopPropagation());

  dropArea.addEventListener('click', (e) => {
    if (e.target.closest('.file-label')) return;
    fileInput.click();
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleDroppedFiles(fileInput.files);
    fileInput.value = '';
  });

  // ── RAG controls ──

  function updateRagBadge() {
    const badge = $('#rag-badge');
    if (!ragEnabled) {
      badge.textContent = 'RAG off';
      badge.className = 'badge badge-idle';
      badge.title = 'RAG disabled';
    } else if (!embedReady) {
      badge.textContent = 'Embed ✗';
      badge.className = 'badge badge-idle';
      badge.title = 'Embed server not running — start it in settings';
    } else if (ragIndexed > 0) {
      badge.textContent = `RAG ${ragIndexed}`;
      badge.className = 'badge badge-ready';
      badge.title = `${ragIndexed} chunks indexed · embed server ready`;
    } else {
      badge.textContent = 'RAG 0';
      badge.className = 'badge badge-ready';
      badge.title = 'Embed server ready · no files indexed yet';
    }
  }

  // Index RAG files, then auto-clear the RAG queue
  $('#rag-index-btn').onclick = async () => {
    if (ragFiles.length === 0) return;
    if (!embedReady) {
      $('#rag-index-status').textContent = 'Start embed server first (settings)';
      $('#rag-index-status').className = 'rag-index-status rag-error';
      return;
    }

    const btn = $('#rag-index-btn');
    const status = $('#rag-index-status');
    btn.disabled = true;
    status.textContent = 'Indexing...';
    status.className = 'rag-index-status rag-indexing';

    const filesPayload = ragFiles.map((f) => ({
      name: f.name,
      content: f.content,
      language: f.language,
    }));

    try {
      const res = await fetch('/api/rag/index', {
        method: 'POST',
        body: JSON.stringify({ files: filesPayload }),
      });
      const d = await res.json();

      if (d.error) {
        status.textContent = d.error;
        status.className = 'rag-index-status rag-error';
      } else {
        ragIndexed = d.chunks_indexed || 0;
        status.textContent = `${ragIndexed} chunks indexed`;
        status.className = 'rag-index-status rag-success';
        updateRagBadge();
        $('#rag-checkbox').checked = true;
        // Auto-clear RAG file queue — they're in the index now
        ragFiles = [];
        renderFileLists();
      }
    } catch (e) {
      status.textContent = String(e);
      status.className = 'rag-index-status rag-error';
    } finally {
      btn.disabled = false;
    }
  };

  // Clear RAG index
  $('#rag-clear-btn').onclick = async () => {
    try {
      await fetch('/api/rag/clear', { method: 'POST' });
      ragIndexed = 0;
      updateRagBadge();
      $('#rag-index-status').textContent = '';
      refreshRagSettings();
    } catch (_) {}
  };

  // ── Embed server management ──

  let embedReady = false;

  $('#embed-start-btn').onclick = async () => {
    $('#embed-start-btn').disabled = true;
    $('#embed-status').textContent = 'Starting embed server...';
    try {
      const d = await fetch('/api/embed/start', { method: 'POST' }).then((r) => r.json());
      if (d.error) {
        $('#embed-status').textContent = d.error;
      } else {
        pollEmbedReady();
      }
    } catch (e) {
      $('#embed-status').textContent = String(e);
    } finally {
      $('#embed-start-btn').disabled = false;
    }
  };

  $('#embed-stop-btn').onclick = async () => {
    await fetch('/api/embed/stop', { method: 'POST' });
    embedReady = false;
    updateRagBadge();
    refreshEmbedStatus();
  };

  function pollEmbedReady() {
    let elapsed = 0;
    const iv = setInterval(async () => {
      elapsed++;
      try {
        const d = await fetch('/api/embed/status').then((r) => r.json());
        updateEmbedStatusLine(d);
        if (d.status === 'ready') {
          clearInterval(iv);
          embedReady = true;
          updateRagBadge();
          $('#embed-start-btn').disabled = false;
        } else if (d.status === 'error' || d.status === 'stopped') {
          clearInterval(iv);
          embedReady = false;
          updateRagBadge();
          $('#embed-start-btn').disabled = false;
        }
      } catch (_) {}
      if (elapsed >= 60) {
        clearInterval(iv);
        $('#embed-start-btn').disabled = false;
      }
    }, 1000);
  }

  function updateEmbedStatusLine(d) {
    const el = $('#embed-status');
    let lines = [`Status: ${d.status || 'unknown'}`];
    if (d.model) lines.push(`Model: ${d.model}`);
    if (d.pid) lines.push(`PID: ${d.pid}`);
    if (d.port) lines.push(`Port: ${d.port}`);
    if (d.error) lines.push(`Error: ${d.error}`);
    el.textContent = lines.join('\n');
  }

  async function refreshEmbedStatus() {
    try {
      const d = await fetch('/api/embed/status').then((r) => r.json());
      embedReady = d.status === 'ready';
      updateEmbedStatusLine(d);
      updateRagBadge();
      // Populate prefix fields (only overwrite if user hasn't focused the input)
      const qp = $('#embed-query-prefix');
      const dp = $('#embed-doc-prefix');
      if (document.activeElement !== qp && d.query_prefix != null) {
        qp.value = d.query_prefix;
      }
      if (document.activeElement !== dp && d.doc_prefix != null) {
        dp.value = d.doc_prefix;
      }
    } catch (_) {}
  }

  // Save embed prefixes
  $('#save-prefixes-btn').onclick = async () => {
    const btn = $('#save-prefixes-btn');
    btn.textContent = 'Saving...';
    try {
      await fetch('/api/embed/prefixes', {
        method: 'POST',
        body: JSON.stringify({
          query_prefix: $('#embed-query-prefix').value,
          doc_prefix: $('#embed-doc-prefix').value,
        }),
      });
      btn.textContent = 'Applied ✓';
      setTimeout(() => (btn.textContent = 'Apply Prefixes'), 1500);
    } catch (e) {
      btn.textContent = 'Error';
      setTimeout(() => (btn.textContent = 'Apply Prefixes'), 2000);
    }
  };

  async function refreshRagSettings() {
    try {
      const d = await fetch('/api/rag/status').then((r) => r.json());
      ragIndexed = d.chunks || 0;
      ragEnabled = d.enabled || false;
      updateRagBadge();

      const el = $('#rag-settings-status');
      let lines = [`Status: ${ragEnabled ? 'enabled' : 'disabled'}`];
      lines.push(`Chunks indexed: ${ragIndexed}`);
      if (d.vector_dim) lines.push(`Vector dimension: ${d.vector_dim}`);
      if (d.files && d.files.length) lines.push(`Files: ${d.files.join(', ')}`);
      lines.push(`DB: ${d.db_path || 'N/A'}`);
      el.textContent = lines.join('\n');
    } catch (_) {}
  }

  // ── Settings overlay ──

  $('#settings-btn').onclick = () => {
    $('#settings-overlay').classList.remove('hidden');
    refreshModels();
    refreshStatus();
    refreshEmbedStatus();
    refreshRagSettings();
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

      const sel = $('#model-select');
      sel.innerHTML = '';
      for (const m of modelsData) {
        const o = document.createElement('option');
        o.value = m.filename;
        o.textContent = `${m.name} [${m.family}]`;
        if (m.filename === d.active) o.selected = true;
        sel.appendChild(o);
      }

      const draftSel = $('#draft-select');
      draftSel.innerHTML = '<option value="">None (disabled)</option>';
      const candidates = d.draft_candidates || [];
      for (const fname of candidates.sort()) {
        const o = document.createElement('option');
        o.value = fname;
        o.textContent = fname.replace('.gguf', '');
        draftSel.appendChild(o);
      }
      if (d.draft) {
        draftSel.value = d.draft.model || '';
        $('#p-draft-max').value = d.draft.max || 10;
        $('#p-draft-ngl').value = d.draft.ngl ?? 99;
        updateDraftInfo(d.draft.model, d.draft.max);
      }

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
      // Sync RAG status from models endpoint too
      if (d.rag) {
        ragIndexed = d.rag.chunks || 0;
        ragEnabled = d.rag.enabled || false;
      }
      // Sync embed server status
      if (d.embed) {
        embedReady = d.embed.status === 'ready';
      }
      updateRagBadge();
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
    const draftModel = $('#draft-select').value;
    updateBadge({ status: 'starting' });
    const draftLabel = draftModel ? ` + draft ${draftModel.replace('.gguf', '')}` : '';
    $('#llama-status').textContent = `Loading model${draftLabel}...`;
    $('#load-btn').disabled = true;

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

      if (d.error) {
        updateBadge({ status: 'error' });
        $('#llama-status').textContent = d.error;
        $('#load-btn').disabled = false;
        return;
      }

      modelCtx = +$('#p-ctx').value || 4096;
      const sel = $('#model-select');
      modelName = sel.options[sel.selectedIndex]?.textContent || model.split('.')[0];
      updateDraftInfo(draftModel, +$('#p-draft-max').value);
      updateBudget();

      pollUntilReady();
    } catch (e) {
      updateBadge({ status: 'error' });
      $('#llama-status').textContent = String(e);
      $('#load-btn').disabled = false;
    }
  };

  function pollUntilReady() {
    let elapsed = 0;
    const iv = setInterval(async () => {
      elapsed++;
      try {
        const d = await fetch('/api/status').then((r) => r.json());
        const status = d.llama?.status;
        updateBadge(d.llama);
        updateLlamaStatus(d.llama);

        if (status === 'ready') {
          clearInterval(iv);
          $('#load-btn').disabled = false;
          if (d.ctx) { modelCtx = d.ctx; updateBudget(); }
        } else if (status === 'error' || status === 'stopped') {
          clearInterval(iv);
          $('#load-btn').disabled = false;
        } else {
          $('#llama-status').textContent += `\nWaiting... (${elapsed}s)`;
        }
      } catch (_) {}

      if (elapsed >= 180) {
        clearInterval(iv);
        $('#load-btn').disabled = false;
        $('#llama-status').textContent += '\nPoll timeout — check server logs';
      }
    }, 1000);
  }

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

    if (abortCtrl) abortCtrl.abort();
    abortCtrl = new AbortController();

    const output = $('#output');
    const copyBtn = $('#copy-btn');
    const statsEl = $('#stats');
    const ctxInfo = $('#context-info');
    const isReview = currentMode === 'review';
    const useRag = $('#rag-checkbox').checked && ragIndexed > 0 && embedReady;

    const loadingLabel = isReview ? 'Reviewing code...' : (useRag ? 'Searching index & writing...' : 'Writing code...');
    output.innerHTML = `<div class="loading"><div class="spinner"></div>${loadingLabel}</div>`;
    copyBtn.classList.add('hidden');
    statsEl.textContent = '';
    ctxInfo.classList.add('hidden');
    ctxInfo.innerHTML = '';

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
          use_rag: useRag,
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

          // Handle RAG info event
          if (data.rag_info) {
            const ri = data.rag_info;
            if (ri.error) {
              console.warn('[rag]', ri.error);
            } else if (ri.chunks_retrieved) {
              const sources = (ri.sources || []).map((s) => s.source).join(', ');
              // Show brief RAG retrieval note in the loading area
              const loadEl = output.querySelector('.loading');
              if (loadEl) {
                loadEl.innerHTML = `<div class="spinner"></div>Retrieved ${ri.chunks_retrieved} chunks from index · generating...`;
              }
            }
            continue;
          }

          // Handle context_info event
          if (data.context_info) {
            const ci = data.context_info;
            let parts = [];
            if (ci.rag_chunks) {
              parts.push(`<span class="ctx-rag">${ci.rag_chunks} RAG</span>`);
            }
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
              if (ci.rag_chunks) lines.unshift(`⚡ ${ci.rag_chunks} RAG chunks retrieved`);
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
              $('#stream-review').innerHTML = renderReview(fullText);
            } else {
              $('#stream-code').textContent = fullText;
            }
            output.scrollTop = output.scrollHeight;
          }

          if (data.done) {
            const cursor = output.querySelector('.streaming-cursor');
            if (cursor) cursor.classList.remove('streaming-cursor');

            if (isReview && fullText) {
              const el = $('#stream-review');
              if (el) el.innerHTML = renderReview(fullText);
            }

            const secs = ((data.elapsed_ms || 0) / 1000).toFixed(1);
            let statsParts = [`${data.tokens || 0} tok`, `${secs}s`];
            if (data.rag_chunks) statsParts.push(`${data.rag_chunks} RAG`);
            statsEl.textContent = statsParts.join(' · ');

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

  function renderReview(text) {
    const parts = text.split(/(```[\s\S]*?```|```[\s\S]*$)/);
    let html = '';
    for (const part of parts) {
      if (part.startsWith('```')) {
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

  // ── Init ──

  (async () => {
    try {
      await refreshModels();
    } catch (_) {}
  })();
})();
