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

  // Mirrors main.rs `lang_domain` — these route to the text chunker, all
  // other languages to the code chunker. Keep the two lists in step.
  const TEXT_LANGS = new Set(['markdown', 'text', 'config', 'ini', 'env', 'gitignore']);

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
  let fileIdCounter = 0;
  let modelCtx = 4096;
  let modelName = '';
  let modelsData = [];
  let ragIndexed = 0;
  let ragCode = 0;
  let ragText = 0;
  let ragEnabled = false;
  let chatHistory = [];   // client-owned chat thread; server is stateless

  // One mode: chat with pinned files, RAG, and the Agent toggle covers
  // everything the old Write/Review modes did (rusty-streamer's approach,
  // codified here too). Dropped files always become pinned context; the
  // index is fed by PATH (file or directory), also the streamer convention.

  // ── Input mode ──
  //
  // Agent on  → the model walks the workspace itself with tools, so the only
  //             thing to configure is which directory that is.
  // Agent off → there is no tool loop, so retrieval is the only way in:
  //             files you add are auto-indexed and reached through search.
  // Exactly one panel is live at a time — showing both is what made this
  // pane read as two competing ways to do the same thing.
  function applyInputMode() {
    const toggle = $('#tools-toggle');
    const agent = toggle && !toggle.classList.contains('hidden')
      && $('#tools-checkbox').checked;
    $('#retrieval-panel').classList.toggle('hidden', agent);
    $('#workspace-section').classList.toggle('hidden', !agent);
    $('#rag-toggle').classList.toggle('hidden', agent);
    if (agent) refreshWorkspace();
    return agent;
  }

  $('#tools-checkbox').addEventListener('change', applyInputMode);

  // ── Server-side workspace ──

  async function refreshWorkspace() {
    try {
      const d = await (await fetch('/api/workspace/status')).json();
      if (d.path) {
        $('#workspace-path').value = d.path;
        // Two ways content gets in, and the line has to reflect both: files
        // the model chose to read, and a bulk walk that never touches the
        // read ledger. Reporting only the former read "nothing indexed yet"
        // with a thousand chunks in the store.
        $('#workspace-status').textContent = d.code_chunks
          ? (d.auto_indexed
              ? `${d.code_chunks} chunks · ${d.auto_indexed} read by the agent`
              : `${d.code_chunks} chunks indexed`)
          : 'ready — nothing indexed yet';
        $('#workspace-status').className = 'rag-index-status';
      }
    } catch { /* server not up yet */ }
  }
  refreshWorkspace();

  $('#workspace-set-btn').addEventListener('click', async () => {
    const path = $('#workspace-path').value.trim();
    const status = $('#workspace-status');
    if (!path) { status.textContent = 'enter a folder path'; return; }
    try {
      const d = await (await fetch('/api/workspace/set', {
        method: 'POST',
        body: JSON.stringify({ path }),
      })).json();
      if (d.error) {
        status.textContent = d.error;
        status.className = 'rag-index-status rag-error';
      } else {
        $('#workspace-path').value = d.path;
        status.textContent = 'workspace set — Sync to index';
        status.className = 'rag-index-status rag-success';
      }
    } catch (e) {
      status.textContent = String(e);
      status.className = 'rag-index-status rag-error';
    }
  });

  // ── Folder picker (navigates the server's filesystem) ──

  let wsbPath = '';

  async function wsbLoad(path) {
    const status = $('#workspace-status');
    try {
      const d = await (await fetch('/api/workspace/browse', {
        method: 'POST',
        body: JSON.stringify({ path: path || '' }),
      })).json();
      if (d.error) { status.textContent = d.error; status.className = 'rag-index-status rag-error'; return; }
      wsbPath = d.path;
      $('#wsb-path').textContent = d.path;
      $('#wsb-git').classList.toggle('hidden', !d.is_git);
      const sep = d.path.includes('\\') ? '\\' : '/';
      const join = (base, name) =>
        base.endsWith(sep) ? base + name : base + sep + name;
      const list = $('#wsb-list');
      list.innerHTML = '';
      const row = (label, target, cls) => {
        const el = document.createElement('div');
        el.className = 'wsb-entry' + (cls ? ' ' + cls : '');
        el.textContent = label;
        el.onclick = () => wsbLoad(target);
        list.appendChild(el);
      };
      if (d.parent) row('..', d.parent, 'wsb-up');
      for (const name of d.dirs) row(name, join(d.path, name));
    } catch (e) {
      status.textContent = String(e);
      status.className = 'rag-index-status rag-error';
    }
  }

  $('#workspace-browse-btn').addEventListener('click', () => {
    const panel = $('#workspace-browser');
    panel.classList.toggle('hidden');
    if (!panel.classList.contains('hidden')) {
      wsbLoad($('#workspace-path').value.trim());
    }
  });
  $('#wsb-cancel').addEventListener('click', () =>
    $('#workspace-browser').classList.add('hidden'));
  $('#wsb-select').addEventListener('click', () => {
    $('#workspace-path').value = wsbPath;
    $('#workspace-browser').classList.add('hidden');
    $('#workspace-set-btn').click();
  });

  // ── Bulk workspace index ──
  //
  // Pre-prompt only: it unloads the model so the GPU embedder has room, then
  // reloads it. The button stays disabled for the duration so a second click
  // cannot start a competing walk.
  let bulkPoll = null;

  $('#bulk-index-btn').onclick = async () => {
    const gpu = $('#bulk-gpu').checked;
    const path = $('#workspace-path').value.trim();
    if (!path) { setBulk('error', 'set a workspace first'); return; }
    const warn = gpu
      ? `Index the entire workspace on the GPU?\n\n${path}\n\nThe model is unloaded while this runs and reloaded when it finishes. On a large repo this can take several minutes.`
      : `Index the entire workspace on the CPU?\n\n${path}\n\nThis is ~13x slower than the GPU embedder but leaves the model loaded.`;
    if (!confirm(warn)) return;
    $('#bulk-index-btn').disabled = true;
    const d = await fetch('/api/workspace/bulk', {
      method: 'POST', body: JSON.stringify({ gpu }),
    }).then((r) => r.json());
    if (d.error) { setBulk('error', d.error); $('#bulk-index-btn').disabled = false; return; }
    if (gpu) updateBadge({ status: 'stopped' });
    pollBulk();
  };

  function setBulk(phase, msg, pct) {
    const st = $('#bulk-status');
    st.textContent = msg;
    st.className = 'rag-index-status ' +
      (phase === 'error' ? 'rag-error' : phase === 'done' ? 'rag-success' : 'rag-indexing');
    const bar = $('#bulk-bar');
    if (pct == null) { bar.classList.add('hidden'); return; }
    bar.classList.remove('hidden');
    $('#bulk-fill').style.width = `${pct.toFixed(1)}%`;
    $('#bulk-fill').className = `bulk-fill ${phase === 'error' ? 'bulk-error' : ''}`.trim();
  }

  function pollBulk() {
    clearInterval(bulkPoll);
    bulkPoll = setInterval(async () => {
      let d;
      try { d = await fetch('/api/workspace/bulk').then((r) => r.json()); }
      catch (_) { return; }
      const pct = d.files_total ? (d.files_done / d.files_total) * 100 : 0;
      if (d.phase === 'embedding') {
        setBulk(d.phase, `${d.files_done}/${d.files_total} files · ${d.chunks} chunks`, pct);
        $('#bulk-detail').textContent = d.message;
      } else if (d.running) {
        setBulk(d.phase, d.message, pct);
        $('#bulk-detail').textContent = '';
      }
      if (!d.running) {
        clearInterval(bulkPoll);
        bulkPoll = null;
        $('#bulk-index-btn').disabled = false;
        $('#bulk-detail').textContent = '';
        if (d.phase === 'error') setBulk('error', d.message, null);
        else if (d.phase === 'done') {
          setBulk('done', `${d.message}${d.skipped ? ` (${d.skipped} skipped)` : ''}`, null);
          $('#rag-checkbox').checked = true;
        }
        // The model was reloaded and the index grew — resync both badges.
        refreshModels();
        refreshWorkspace();
        await refreshRagCounts();
        updateRagBadge();
      }
    }, 1000);
  }

  function addFile(name, content) {
    const language = extToLang(name);
    const tokens = estimateTokens(content);
    const id = ++fileIdCounter;
    const file = { id, name, content, language, tokens, state: 'queued' };
    contextFiles.push(file);
    renderFileLists();
    enqueueIndex(file);
  }

  function removeContextFile(id) {
    contextFiles = contextFiles.filter((f) => f.id !== id);
    renderFileLists();
  }

  // ── Render file lists ──

  function renderFileLists() {
    renderFileListInto('#context-file-list', contextFiles, removeContextFile);
    const ctxSection = $('#context-file-section');
    if (contextFiles.length > 0) { ctxSection.classList.remove('hidden'); }
    else { ctxSection.classList.add('hidden'); }
  }

  function indexStateLabel(f) {
    switch (f.state) {
      case 'indexing': return 'indexing…';
      case 'indexed':  return f.chunks ? `${f.chunks} chunks` : 'indexed';
      case 'error':    return 'failed';
      default:         return 'queued';
    }
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
        <span class="file-index-state file-state-${f.state || 'queued'}"
              title="${esc(f.error || '')}">${indexStateLabel(f)}</span>
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

  // The old write-mode budget bar is gone: the server prices pinned files
  // with the real tokenizer and reports the outcome in context_info, which
  // is truth where the bar was a chars-per-token guess.

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
    // Retrieval searches both domains, so the badge counts both.
    const count = ragText + ragCode;
    if (!ragEnabled) {
      badge.textContent = 'RAG off';
      badge.className = 'badge badge-idle';
      badge.title = 'RAG disabled';
    } else if (count > 0) {
      badge.textContent = `RAG ${count}`;
      badge.className = 'badge badge-ready';
      badge.title = `${ragCode} code + ${ragText} text chunks indexed${embedReady ? ' · embed ready' : ' · embed starts on use'}`;
    } else {
      badge.textContent = 'RAG 0';
      badge.className = 'badge badge-idle';
      badge.title = 'Nothing indexed yet — embed server starts on first use';
    }
  }

  // Auto-index queue. In retrieval mode every file you add is embedded into
  // the RAG store as soon as it lands — no Index button to forget. Strictly
  // one at a time: the embed server runs a single slot, and a burst of
  // parallel /api/rag/index calls would queue inside it anyway while making
  // per-file progress impossible to report.
  const indexQueue = [];
  let indexRunning = false;

  function enqueueIndex(file) {
    indexQueue.push(file);
    if (!indexRunning) drainIndexQueue();
  }

  async function drainIndexQueue() {
    indexRunning = true;
    while (indexQueue.length) {
      const file = indexQueue[0];
      file.state = 'indexing';
      renderFileLists();
      setIndexStatus(
        embedReady
          ? `Indexing ${file.name}… (${indexQueue.length} queued)`
          : `Starting embed server to index ${file.name}…`,
        'rag-indexing',
      );
      try {
        const d = await fetch('/api/rag/index', {
          method: 'POST',
          body: JSON.stringify({
            files: [{ name: file.name, content: file.content, language: file.language }],
            domain: TEXT_LANGS.has(file.language) ? 'text' : 'code',
          }),
        }).then((r) => r.json());
        if (d.error) throw new Error(d.error);
        embedReady = true;               // the lazy start succeeded
        file.state = 'indexed';
        file.chunks = d.chunks_indexed || 0;
      } catch (e) {
        file.state = 'error';
        file.error = String(e.message || e);
        setIndexStatus(`${file.name}: ${file.error}`, 'rag-error');
      }
      indexQueue.shift();
      renderFileLists();
    }
    indexRunning = false;
    await refreshRagCounts();
    const failed = contextFiles.filter((f) => f.state === 'error').length;
    if (!failed) {
      setIndexStatus(`Indexed · ${ragCode} code / ${ragText} text chunks`, 'rag-success');
      $('#rag-checkbox').checked = true;
    }
    updateRagBadge();
  }

  function setIndexStatus(text, cls) {
    const el = $('#rag-index-status');
    if (!el) return;
    el.textContent = text;
    el.className = `rag-index-status ${cls || ''}`.trim();
  }

  async function refreshRagCounts() {
    try {
      const d = await fetch('/api/rag/status').then((r) => r.json());
      ragCode = d.chunks_code || 0;
      ragText = d.chunks_text || 0;
      ragIndexed = d.chunks || 0;
    } catch (_) {}
  }


  // Clear RAG index
  $('#rag-clear-btn').onclick = async () => {
    try {
      await fetch('/api/rag/clear', { method: 'POST' });
      ragIndexed = 0;
      ragCode = 0;
      ragText = 0;
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
      ragCode = d.chunks_code || 0;
      ragText = d.chunks_text || 0;
      ragEnabled = d.enabled || false;
      updateRagBadge();

      const el = $('#rag-settings-status');
      let lines = [`Status: ${ragEnabled ? 'enabled' : 'disabled'}`];
      lines.push(`Chunks: ${ragIndexed} (code ${ragCode}, text ${ragText})`);
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
      if (d.spec) {
        draftSel.value = d.spec.draft_model || '';
        $('#p-draft-max').value = d.spec.draft_n_max || 2;
        $('#p-draft-ngl').value = d.spec.gpu_layers_draft ?? 99;
        updateDraftInfo(d.spec.draft_model, d.spec.draft_n_max);
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
        $('#p-effort').value = d.params.reasoning_effort || '';
        renderEffortHint(d.params);
        modelCtx = d.params.ctx || 4096;
      }
      // Sync RAG status from models endpoint too
      if (d.rag) {
        ragIndexed = d.rag.chunks || 0;
        ragCode = d.rag.chunks_code || 0;
        ragText = d.rag.chunks_text || 0;
        ragEnabled = d.rag.enabled || false;
      }
      // Sync embed server status
      if (d.embed) {
        embedReady = d.embed.status === 'ready';
      }
      updateRagBadge();
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
        + `Proposes up to ${draftMax || 2} tokens per step.\n`
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
      $('#p-effort').value = m.reasoning_effort || '';
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
          temp: +$('#p-temp').value,
          top_k: +$('#p-topk').value,
          top_p: +$('#p-topp').value,
          repeat_penalty: +$('#p-rp').value,
          reasoning_effort: $('#p-effort').value,
          ...(draftModel ? {
            spec_type: 'draft-model',
            draft_model: draftModel,
            spec_draft_n_max: +$('#p-draft-max').value || 2,
            gpu_layers_draft: +$('#p-draft-ngl').value,
          } : {}),
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
          if (d.ctx) { modelCtx = d.ctx; }
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

  $('#save-params-btn').onclick = async () => {
    const d = await fetch('/api/params', {
      method: 'POST',
      body: JSON.stringify({
        temp: +$('#p-temp').value,
        top_k: +$('#p-topk').value,
        top_p: +$('#p-topp').value,
        repeat_penalty: +$('#p-rp').value,
        reasoning_effort: $('#p-effort').value,
      }),
    }).then((r) => r.json());
    if (d.error) {
      $('#effort-hint').textContent = d.error;
      return;
    }
    // The server clamps to the hardware tier's ceiling, so echo back what it
    // actually accepted rather than what was asked for.
    if (d.reasoning_effort !== undefined) $('#p-effort').value = d.reasoning_effort;
    refreshModels();
  };

  // Explains what the selected effort will actually do: which channel carries
  // the tag to the model, and whether this hardware tier capped it. Without
  // this, an ignored or clamped selection looks identical to a working one.
  function renderEffortHint(params) {
    const el = $('#effort-hint');
    if (!el) return;
    const eff = params.reasoning_effort || '';
    if (!eff) {
      el.textContent = 'Model default (Qwen 3.8 ships at xhigh thinking).';
      return;
    }
    // No base mode starts with "i", so the prefix is an unambiguous flag.
    const instruct = eff.startsWith('i');
    const parts = [
      `${eff} — ${instruct ? 'instruct (no thinking block)' : 'thinking'}`,
      params.effort_native
        ? 'sent as chat_template_kwargs.reasoning_effort'
        : "template ignores the kwarg — sent as a {REASON:} prompt marker",
    ];
    if (params.effort_ceiling && params.effort_ceiling !== 'spoon') {
      parts.push(`tier ceiling: ${params.effort_ceiling}`);
    }
    el.textContent = parts.join(' · ');
  }

  async function refreshStatus() {
    try {
      const d = await fetch('/api/status').then((r) => r.json());
      $('#usage-info').textContent =
        `Requests: ${d.requests}  Tokens: ${d.tokens_session}\nModel: ${d.model || 'none'}`;
      if (d.ctx) {
        modelCtx = d.ctx;
      }
      // The Agent toggle exists only when the server built a tool runtime.
      if (d.tools) {
        $('#tools-toggle').classList.toggle('hidden', !d.tools.enabled);
        applyInputMode();
      }
    } catch (_) {}
  }

  // ── Streaming write ──

  let abortCtrl = null;

  $('#write-btn').onclick = doChat;
  $('#write-desc').addEventListener('keydown', (e) => {
    if (e.key !== 'Enter') return;
    if (!e.shiftKey) { e.preventDefault(); doChat(); }   // Enter sends, Shift+Enter newline
  });

  $('#chat-new-btn').onclick = () => {
    if (abortCtrl) abortCtrl.abort();
    chatHistory = [];
    renderChat(null);
    $('#stats').textContent = '';
    $('#context-info').classList.add('hidden');
    resetCtxMeter();
  };

  // Tool status chips for the round in flight, cleared per turn.
  let liveToolChips = [];

  // One chip per tool_call on a stored assistant message.
  // Context window usage. The server prices the assembled prompt with the
  // real tokenizer and reports it per turn, so this is measured, not an
  // estimate — but it only arrives while a turn is in flight, so the last
  // reading is kept on screen between turns rather than blanking out.
  let lastCtxInfo = null;

  function renderCtxMeter(ci) {
    if (ci) lastCtxInfo = ci;
    const info = lastCtxInfo;
    const meter = $('#ctx-meter');
    if (!info || !info.model_ctx) { meter.classList.add('hidden'); return; }
    const used = info.input_tokens || 0;
    const total = info.model_ctx;
    const pct = Math.max(0, Math.min(100, (used / total) * 100));
    $('#ctx-meter-fill').style.width = `${pct.toFixed(1)}%`;
    // Bands, not a gradient: the number that matters is whether the next turn
    // still fits, and eviction starts biting well before the window is full.
    $('#ctx-meter-fill').className =
      `ctx-meter-fill ${pct >= 90 ? 'ctx-critical' : pct >= 70 ? 'ctx-warn' : ''}`.trim();
    $('#ctx-meter-label').textContent =
      `${formatTokens(used)} / ${formatTokens(total)}`;
    meter.title =
      `Context window: ${used.toLocaleString()} of ${total.toLocaleString()} tokens used `
      + `(${pct.toFixed(0)}%)\n`
      + `${formatTokens(info.remaining_tokens || 0)} left for the reply`
      + (info.rag_chunks ? `\n${info.rag_chunks} retrieved chunk(s) in this prompt` : '')
      + (info.pinned_files && info.pinned_files.length
          ? `\n${info.pinned_files.length} pinned file(s)` : '')
      + (info.turns_kept != null && info.turns_total != null && info.turns_kept < info.turns_total
          ? `\n${info.turns_total - info.turns_kept} older turn(s) trimmed to fit` : '');
    meter.classList.remove('hidden');

    // Mirror into the top bar, which is visible no matter which pane the
    // user is looking at — the pane-header copy scrolls out of reach as
    // soon as the conversation gets long, which is exactly when the number
    // starts to matter.
    const bar = $('#ctx-bar');
    if (bar) {
      $('#ctx-bar-fill').style.width = `${pct.toFixed(1)}%`;
      $('#ctx-bar-fill').className =
        `ctx-bar-fill ${pct >= 90 ? 'ctx-critical' : pct >= 70 ? 'ctx-warn' : ''}`.trim();
      $('#ctx-bar-text').textContent = `${formatTokens(used)}/${formatTokens(total)}`;
      bar.title = meter.title;
      bar.classList.remove('hidden');
    }
  }

  function renderCallChips(toolCalls) {
    let html = '';
    for (const tc of toolCalls || []) {
      const fn = tc.function || {};
      const arg = (fn.arguments || '').slice(0, 60);
      html += `<div class="tool-chip">🔧 ${esc(fn.name || 'tool')} <code>${esc(arg)}</code></div>`;
    }
    return html;
  }

  // Render the chat transcript from the RAW history objects. The rendering
  // reads tool_calls/tool messages for display, but the objects themselves
  // are what gets POSTed back — never strip-then-resend: history that hides
  // past tool rounds teaches the model that calling tools does nothing.
  // Pass a string to append a live streaming assistant bubble; null when idle.
  function renderChat(streaming) {
    const output = $('#output');
    $('#chat-new-btn').classList.toggle('hidden', chatHistory.length === 0 && streaming == null);
    if (chatHistory.length === 0 && streaming == null) {
      output.innerHTML = '<div class="placeholder-msg">Start a conversation — ask a question or paste text to discuss. Toggle RAG to ground answers in your indexed text.</div>';
      return;
    }
    let html = '<div class="chat-transcript">';
    for (const m of chatHistory) {
      if (m.role === 'tool') {
        html += `<div class="chat-msg chat-tool"><details class="tool-result"><summary>🔧 tool result</summary><pre>${esc(m.content || '')}</pre></details></div>`;
        continue;
      }
      let body = m.role === 'assistant' ? renderReview(m.content || '') : esc(m.content || '');
      if (m.role === 'assistant' && m.tool_calls) {
        body = renderCallChips(m.tool_calls) + body;
      }
      html += `<div class="chat-msg chat-${m.role}"><div class="chat-role">${m.role}</div><div class="chat-body">${body}</div></div>`;
    }
    if (streaming != null) {
      const chips = liveToolChips.map((c) =>
        `<div class="tool-chip ${c.ok ? 'tool-ok' : 'tool-fail'}">🔧 ${esc(c.summary)} → ${esc(c.status)}</div>`
      ).join('');
      html += `<div class="chat-msg chat-assistant"><div class="chat-role">assistant</div><div class="chat-body streaming-cursor">${chips}${renderReview(streaming)}</div></div>`;
    }
    html += '</div>';
    output.innerHTML = html;
    output.scrollTop = output.scrollHeight;
  }

  // Minimal SSE frame reader shared by the chat pipeline. onEvent may return
  // 'stop' to end the stream early.
  async function readSSE(res, onEvent) {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
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
        if (onEvent(data) === 'stop') return;
      }
    }
  }

  async function doChat() {
    const input = $('#write-desc');
    const text = input.value.trim();
    if (!text) return;

    if (abortCtrl) abortCtrl.abort();
    abortCtrl = new AbortController();

    chatHistory.push({ role: 'user', content: text });
    input.value = '';
    renderChat('');
    setGenerating(true);

    const statsEl = $('#stats');
    const ctxInfo = $('#context-info');
    statsEl.textContent = '';
    ctxInfo.classList.add('hidden');
    ctxInfo.innerHTML = '';
    renderCtxMeter(null);   // keep the previous reading visible while this turn assembles

    // Retrieval spans both domains server-side, so either count qualifies.
    const useRag = $('#rag-checkbox').checked && (ragText + ragCode) > 0;
    const useTools = !$('#tools-toggle').classList.contains('hidden')
      && $('#tools-checkbox').checked;
    let assistantText = '';
    let tokenCount = 0;
    // Set once the server's final `history` event stores the answer — the
    // synthesized push below is then a duplicate and must not fire.
    let historyFinal = false;
    liveToolChips = [];
    const genStart = Date.now();

    try {
      const res = await fetch('/api/write', {
        method: 'POST',
        signal: abortCtrl.signal,
        body: JSON.stringify({
          messages: chatHistory,
          use_rag: useRag,
          use_tools: useTools,
          // Context files ride as pinned code context in the (cached) system
          // prefix — they persist for the whole thread.
          files: contextFiles.map((f) => ({ name: f.name, content: f.content, language: f.language })),
        }),
      });

      await readSSE(res, (data) => {
        if (data.error) {
          renderChat(null);
          const out = $('#output');
          out.insertAdjacentHTML('beforeend', `<div class="error-msg">${esc(data.error)}</div>`);
          return 'stop';
        }
        if (data.rag_info) {
          if (data.rag_info.error) console.warn('[chat rag]', data.rag_info.error);
          return;
        }
        if (data.context_info) {
          const ci = data.context_info;
          renderCtxMeter(ci);
          const parts = [];
          if (ci.rag_chunks) parts.push(`<span class="ctx-rag">${ci.rag_chunks} RAG</span>`);
          if (ci.turns_kept != null && ci.turns_total != null && ci.turns_kept < ci.turns_total) {
            parts.push(`<span class="ctx-dropped">${ci.turns_total - ci.turns_kept} older turns trimmed</span>`);
          }
          if (ci.remaining_tokens != null) parts.push(`${formatTokens(ci.remaining_tokens)} left for reply`);
          if (parts.length) {
            ctxInfo.innerHTML = parts.join(' · ');
            ctxInfo.classList.remove('hidden');
          }
          return;
        }
        if (data.token) {
          assistantText += data.token;
          tokenCount++;
          renderChat(assistantText);
        }
        // Agentic events. `history` messages are stored VERBATIM and resent
        // whole next turn — the server's tool round-trip depends on it.
        if (data.history) {
          chatHistory.push(data.history);
          if (data.history.role === 'assistant') {
            if (data.history.tool_calls) {
              // Round boundary: this round's text is stored; the next round's
              // stream starts a fresh bubble.
              assistantText = '';
              renderChat('');
            } else {
              historyFinal = true;
            }
          }
        }
        if (data.tool) {
          liveToolChips.push(data.tool);
          renderChat(assistantText);
        }
        if (data.notice) {
          ctxInfo.innerHTML += `${ctxInfo.innerHTML ? ' · ' : ''}<span class="ctx-dropped">${esc(data.notice)}</span>`;
          ctxInfo.classList.remove('hidden');
        }
        if (data.done) {
          const secs = ((data.elapsed_ms || 0) / 1000).toFixed(1);
          const p = [`${data.tokens || 0} tok`, `${secs}s`];
          if (data.rag_chunks) p.push(`${data.rag_chunks} RAG`);
          if (data.tool_rounds) p.push(`${data.tool_rounds} tool round${data.tool_rounds === 1 ? '' : 's'}`);
          if (data.turns_kept) p.push(`${data.turns_kept} turns`);
          statsEl.textContent = p.join(' · ');
        }
      });

      // Abort fallback only: when the final `history` event stored the
      // answer, pushing the accumulated stream again would duplicate it.
      if (assistantText && !historyFinal) chatHistory.push({ role: 'assistant', content: assistantText });
      liveToolChips = [];
      renderChat(null);
    } catch (e) {
      if (e.name === 'AbortError') {
        if (assistantText && !historyFinal) chatHistory.push({ role: 'assistant', content: assistantText });
        liveToolChips = [];
        renderChat(null);
        const secs = ((Date.now() - genStart) / 1000).toFixed(1);
        statsEl.textContent = `${tokenCount} tok · ${secs}s · stopped`;
      } else {
        renderChat(null);
        $('#output').insertAdjacentHTML('beforeend', `<div class="error-msg">Error: ${esc(String(e))}</div>`);
      }
    } finally {
      abortCtrl = null;
      setGenerating(false);
    }
  }

  function resetCtxMeter() {
    lastCtxInfo = null;
    $('#ctx-meter').classList.add('hidden');
    $('#ctx-bar').classList.add('hidden');
  }

  function setGenerating(on) {
    const writeBtn = $('#write-btn');
    const abortBtn = $('#abort-btn');
    if (on) {
      writeBtn.classList.add('hidden');
      abortBtn.classList.remove('hidden');
    } else {
      writeBtn.classList.remove('hidden');
      abortBtn.classList.add('hidden');
    }
  }

  // Abort button handler
  $('#abort-btn').onclick = () => {
    if (abortCtrl) abortCtrl.abort();
  };

  // Escape key aborts generation
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && abortCtrl) {
      e.preventDefault();
      abortCtrl.abort();
    }
  });

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

  document.body.classList.add('mode-chat');
  renderChat(null);
  (async () => {
    try {
      await refreshModels();
      // Also picks up whether [tools] is enabled, which shows the Agent
      // toggle — refreshStatus otherwise only runs from the settings overlay.
      await refreshStatus();
    } catch (_) {}
  })();
})();
