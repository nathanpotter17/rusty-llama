(function () {
  const $ = (s) => document.querySelector(s);

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
      const sel = $('#model-select');
      sel.innerHTML = '';
      for (const m of d.models || []) {
        const o = document.createElement('option');
        o.value = m.filename;
        o.textContent = `${m.name} [${m.family}]`;
        if (m.filename === d.active) o.selected = true;
        sel.appendChild(o);
      }
      if (d.params) {
        $('#p-ngl').value = d.params.ngl;
        $('#p-ctx').value = d.params.ctx;
        $('#p-temp').value = d.params.temp;
        $('#p-topk').value = d.params.top_k;
        $('#p-topp').value = d.params.top_p;
        $('#p-rp').value = d.params.repeat_penalty;
      }
      updateBadge(d.llama);
      updateLlamaStatus(d.llama);
    } catch (e) {
      console.error('refreshModels:', e);
    }
  }

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
    updateBadge({ status: 'starting' });
    $('#llama-status').textContent = 'Loading model... (may take 1-2 minutes)';
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
        }),
      }).then((r) => r.json());
      updateBadge(d.llama || { status: d.error ? 'error' : 'stopped' });
      updateLlamaStatus(d.llama);
      if (d.error) $('#llama-status').textContent = d.error;
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

    output.innerHTML = '<div class="loading"><div class="spinner"></div>Writing code...</div>';
    copyBtn.classList.add('hidden');
    statsEl.textContent = '';

    try {
      const res = await fetch('/api/write', {
        method: 'POST',
        signal: abortCtrl.signal,
        body: JSON.stringify({
          description: desc,
          language: $('#lang-select').value,
          context: $('#write-ctx').value.trim() || undefined,
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

          if (data.token) {
            if (!started) {
              output.innerHTML = '<pre class="code-block streaming-cursor"><code id="stream-code"></code></pre>';
              started = true;
            }
            fullText += data.token;
            $('#stream-code').textContent = fullText;
            // Auto-scroll output
            output.scrollTop = output.scrollHeight;
          }

          if (data.done) {
            // Remove streaming cursor
            const pre = output.querySelector('.streaming-cursor');
            if (pre) pre.classList.remove('streaming-cursor');

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

  // ── Helpers ──

  function esc(s) {
    return (s || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // ── Init: check model status ──

  (async () => {
    try {
      const d = await fetch('/api/status').then((r) => r.json());
      updateBadge(d.llama);
    } catch (_) {}
  })();
})();
