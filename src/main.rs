use serde::{Deserialize, Serialize};
use std::{
    env, fs,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::Path,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

// ── Config ──────────────────────────────────────────────────

const MAX_BODY_BYTES: usize = 10 * 1024 * 1024; // 10 MB cap

#[derive(Deserialize)]
#[serde(default)]
struct FileConfig {
    server: ServerCfg,
    llama: LlamaCfg,
    defaults: DefaultsCfg,
    limits: LimitsCfg,
    #[serde(default)]
    models: Vec<ModelEntry>,
}

impl Default for FileConfig {
    fn default() -> Self {
        Self {
            server: ServerCfg::default(),
            llama: LlamaCfg::default(),
            defaults: DefaultsCfg::default(),
            limits: LimitsCfg::default(),
            models: Vec::new(),
        }
    }
}

#[derive(Deserialize, Clone)]
#[serde(default)]
struct ServerCfg {
    port: u16,
}
impl Default for ServerCfg {
    fn default() -> Self { Self { port: 8090 } }
}

#[derive(Deserialize, Clone)]
#[serde(default)]
struct LlamaCfg {
    binary: String,
    port: u16,
    parallel_slots: u32,
    startup_timeout: u64,
}
impl Default for LlamaCfg {
    fn default() -> Self {
        Self { binary: String::new(), port: 8079, parallel_slots: 1, startup_timeout: 120 }
    }
}

#[derive(Deserialize, Clone)]
#[serde(default)]
struct DefaultsCfg {
    model: String,
    models_dir: String,
    gpu_layers: i32,
    context_size: u32,
    flash_attention: bool,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    repeat_penalty: f32,
    cache_type_k: String,
    cache_type_v: String,
    // Speculative decoding
    draft_model: String,
    draft_max: u32,
    gpu_layers_draft: i32,
}
impl Default for DefaultsCfg {
    fn default() -> Self {
        Self {
            model: String::new(), models_dir: "models".into(), gpu_layers: -1,
            context_size: 0, flash_attention: true, temperature: 0.7,
            top_k: 40, top_p: 0.9, repeat_penalty: 1.1,
            cache_type_k: String::new(), cache_type_v: String::new(),
            draft_model: String::new(), draft_max: 0, gpu_layers_draft: 99,
        }
    }
}

#[derive(Deserialize, Clone)]
#[serde(default)]
struct LimitsCfg {
    session_tokens: u64,
    daily_tokens: u64,
}
impl Default for LimitsCfg {
    fn default() -> Self { Self { session_tokens: 0, daily_tokens: 0 } }
}

#[derive(Deserialize, Clone, Serialize)]
struct ModelEntry {
    filename: String,
    #[serde(default)]
    name: String,
    #[serde(default = "def_family")]
    family: String,
    #[serde(default = "def_ngl")]
    gpu_layers: i32,
    #[serde(default = "def_ctx")]
    context_size: u32,
    #[serde(default = "def_true")]
    flash_attention: bool,
    #[serde(default = "def_temp")]
    temperature: f32,
    #[serde(default = "def_topk")]
    top_k: u32,
    #[serde(default = "def_topp")]
    top_p: f32,
    #[serde(default = "def_rp")]
    repeat_penalty: f32,
}

fn def_family() -> String { "unknown".into() }
fn def_ngl() -> i32 { 15 }
fn def_ctx() -> u32 { 4096 }
fn def_true() -> bool { true }
fn def_temp() -> f32 { 0.7 }
fn def_topk() -> u32 { 40 }
fn def_topp() -> f32 { 0.9 }
fn def_rp() -> f32 { 1.1 }

// ── Runtime state ───────────────────────────────────────────

#[derive(Clone)]
struct RuntimeCfg {
    port: u16,
    llama_binary: String,
    llama_port: u16,
    parallel_slots: u32,
    startup_timeout: u64,
    session_limit: u64,
    daily_limit: u64,
    models_dir: String,
    // Active inference params
    active_model: String,
    ngl: i32,
    ctx: u32,
    flash_attn: bool,
    temp: f32,
    top_k: u32,
    top_p: f32,
    repeat_penalty: f32,
    // KV cache quantization (empty = llama-server default)
    cache_type_k: String,
    cache_type_v: String,
    // Speculative decoding (empty draft_model = disabled)
    draft_model: String,
    draft_max: u32,
    gpu_layers_draft: i32,
}

impl RuntimeCfg {
    fn endpoint(&self) -> String {
        format!("http://127.0.0.1:{}/v1/chat/completions", self.llama_port)
    }
    fn has_model(&self) -> bool { !self.active_model.is_empty() }
}

// ── Discovered model (on-disk .gguf matched against config) ─

#[derive(Clone, Serialize)]
struct Model {
    filename: String,
    path: String,
    name: String,
    family: String,
    gpu_layers: i32,
    context_size: u32,
    flash_attention: bool,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    repeat_penalty: f32,
}

fn discover_models(dir: &str, known: &[ModelEntry], defaults: &DefaultsCfg) -> Vec<Model> {
    let Ok(entries) = fs::read_dir(dir) else {
        eprintln!("  models dir '{dir}' not found");
        return Vec::new();
    };
    let mut models: Vec<Model> = entries
        .flatten()
        .filter_map(|e| {
            let fname = e.file_name().to_string_lossy().to_string();
            if !fname.ends_with(".gguf") { return None; }
            let path = e.path().to_string_lossy().to_string();
            Some(if let Some(k) = known.iter().find(|m| m.filename == fname) {
                Model {
                    filename: fname, path,
                    name: if k.name.is_empty() { pretty_name(&k.filename) } else { k.name.clone() },
                    family: k.family.clone(), gpu_layers: k.gpu_layers,
                    context_size: k.context_size, flash_attention: k.flash_attention,
                    temperature: k.temperature, top_k: k.top_k, top_p: k.top_p,
                    repeat_penalty: k.repeat_penalty,
                }
            } else {
                Model {
                    name: pretty_name(&fname), filename: fname, path, family: "unknown".into(),
                    gpu_layers: defaults.gpu_layers,
                    context_size: if defaults.context_size > 0 { defaults.context_size } else { 4096 },
                    flash_attention: defaults.flash_attention, temperature: defaults.temperature,
                    top_k: defaults.top_k, top_p: defaults.top_p, repeat_penalty: defaults.repeat_penalty,
                }
            })
        })
        .collect();
    models.sort_by(|a, b| a.filename.cmp(&b.filename));
    models
}

fn pretty_name(f: &str) -> String {
    f.trim_end_matches(".gguf").replace(['-', '_'], " ")
}

// ── Token estimation ────────────────────────────────────────
// Approximate: ~3.2 chars per token for code (conservative)

fn estimate_tokens(s: &str) -> u64 {
    (s.len() as f64 / 3.2).ceil() as u64
}

// ── Llama server management ─────────────────────────────────

#[derive(Clone, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum LlamaStatus { Stopped, Starting, Ready, Error(String) }

struct LlamaServer {
    child: Option<Child>,
    status: LlamaStatus,
    model: String,
    pid: Option<u32>,
}

impl LlamaServer {
    fn new() -> Self {
        Self { child: None, status: LlamaStatus::Stopped, model: String::new(), pid: None }
    }

    fn start(&mut self, cfg: &RuntimeCfg, model: &Model) -> Result<(), String> {
        self.stop();
        let ngl = if cfg.ngl < 0 { 99 } else { cfg.ngl };
        let fa = if cfg.flash_attn { "on" } else { "auto" };
        eprintln!("[llama] starting {} (ngl={ngl}, ctx={}, fa={fa})", model.name, cfg.ctx);

        let mut args = vec![
            "-m".into(), model.path.clone(),
            "--port".into(), cfg.llama_port.to_string(),
            "-ngl".into(), ngl.to_string(),
            "-c".into(), cfg.ctx.to_string(),
            "-np".into(), cfg.parallel_slots.to_string(),
            "--host".into(), "127.0.0.1".into(),
            "--flash-attn".into(), fa.into(),
        ];

        // KV cache quantization
        if !cfg.cache_type_k.is_empty() {
            eprintln!("[llama]   cache-type-k={}", cfg.cache_type_k);
            args.extend(["--cache-type-k".into(), cfg.cache_type_k.clone()]);
        }
        if !cfg.cache_type_v.is_empty() {
            eprintln!("[llama]   cache-type-v={}", cfg.cache_type_v);
            args.extend(["--cache-type-v".into(), cfg.cache_type_v.clone()]);
        }

        // Speculative decoding
        if !cfg.draft_model.is_empty() {
            let draft_path = format!("{}/{}", cfg.models_dir, cfg.draft_model);
            if Path::new(&draft_path).exists() {
                let draft_ngl = if cfg.gpu_layers_draft < 0 { 99 } else { cfg.gpu_layers_draft };
                eprintln!("[llama]   draft={} (ngl={draft_ngl}, max={})", cfg.draft_model, cfg.draft_max);
                args.extend([
                    "--model-draft".into(), draft_path,
                    "--gpu-layers-draft".into(), draft_ngl.to_string(),
                    "--draft-max".into(), cfg.draft_max.max(2).to_string(),
                ]);
            } else {
                eprintln!("[llama]   WARNING: draft model '{}' not found, skipping", draft_path);
            }
        }

        let child = Command::new(&cfg.llama_binary)
            .args(&args)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("spawn failed: {e}"))?;

        self.pid = Some(child.id());
        self.child = Some(child);
        self.status = LlamaStatus::Starting;
        self.model = model.filename.clone();
        Ok(())
    }

    fn wait_ready(&mut self, port: u16, timeout_secs: u64) -> bool {
        let url = format!("http://127.0.0.1:{port}/health");
        let deadline = Instant::now() + Duration::from_secs(timeout_secs);
        std::thread::sleep(Duration::from_millis(500));

        while Instant::now() < deadline {
            // Check if process died
            if let Some(ref mut c) = self.child {
                match c.try_wait() {
                    Ok(Some(st)) => {
                        self.status = LlamaStatus::Error(format!("exited: {st}"));
                        self.child = None;
                        return false;
                    }
                    Err(e) => {
                        self.status = LlamaStatus::Error(format!("wait: {e}"));
                        return false;
                    }
                    Ok(None) => {}
                }
            } else {
                self.status = LlamaStatus::Error("process gone".into());
                return false;
            }

            // Probe health
            if let Ok(out) = Command::new(curl_cmd())
                .args(["-s", "--max-time", "2", &url])
                .output()
            {
                let body = String::from_utf8_lossy(&out.stdout);
                if out.status.success() && (body.contains("ok") || body.contains("\"status\"")) {
                    eprintln!("[llama] ready");
                    self.status = LlamaStatus::Ready;
                    return true;
                }
            }
            std::thread::sleep(Duration::from_millis(800));
        }
        self.status = LlamaStatus::Error(format!("timeout ({timeout_secs}s)"));
        false
    }

    fn stop(&mut self) {
        if let Some(mut c) = self.child.take() {
            eprintln!("[llama] killing pid {:?}", self.pid);
            let _ = c.kill();
            let _ = c.wait();
        }
        self.status = LlamaStatus::Stopped;
        self.model.clear();
        self.pid = None;
    }

    fn is_ready(&self) -> bool { self.status == LlamaStatus::Ready }

    fn status_json(&self) -> serde_json::Value {
        serde_json::json!({
            "status": match &self.status {
                LlamaStatus::Stopped => "stopped",
                LlamaStatus::Starting => "starting",
                LlamaStatus::Ready => "ready",
                LlamaStatus::Error(_) => "error",
            },
            "model": self.model,
            "pid": self.pid,
            "error": match &self.status {
                LlamaStatus::Error(e) => Some(e.as_str()),
                _ => None,
            },
        })
    }
}

impl Drop for LlamaServer {
    fn drop(&mut self) { self.stop(); }
}

fn curl_cmd() -> &'static str {
    if cfg!(windows) { "curl.exe" } else { "curl" }
}

fn find_llama_binary(models_dir: &str) -> String {
    let candidates = [
        "llama-server", "./llama-server", "../llama-server",
        "./models/llama-server.exe", "/usr/local/bin/llama-server",
    ];
    for p in candidates {
        if Path::new(p).exists() {
            return Path::new(p).canonicalize()
                .map(|c| c.to_string_lossy().into()).unwrap_or_else(|_| p.into());
        }
    }
    // Check inside models dir
    let in_models = format!("{models_dir}/llama-server");
    if Path::new(&in_models).exists() {
        return Path::new(&in_models).canonicalize()
            .map(|c| c.to_string_lossy().into()).unwrap_or(in_models);
    }
    // Try PATH
    if let Ok(o) = Command::new("which").arg("llama-server").output() {
        if o.status.success() {
            let p = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if !p.is_empty() { return p; }
        }
    }
    "llama-server".into()
}

// ── Shared state ────────────────────────────────────────────

struct State {
    cfg: RuntimeCfg,
    models: Vec<Model>,
    llama: LlamaServer,
    tokens_session: u64,
    requests: u64,
}

type Shared = Arc<Mutex<State>>;

// ── API types ───────────────────────────────────────────────

#[derive(Deserialize)]
struct FileEntry {
    name: String,
    content: String,
    #[serde(default)]
    language: String,
}

#[derive(Deserialize)]
struct WriteReq {
    description: String,
    #[serde(default = "def_lang")]
    language: String,
    #[serde(default = "def_mode")]
    mode: String,
    #[serde(default)]
    context: String,
    #[serde(default)]
    files: Vec<FileEntry>,
}
fn def_lang() -> String { "python".into() }
fn def_mode() -> String { "write".into() }

#[derive(Deserialize)]
struct LoadReq {
    model: String,
    #[serde(default)]
    ngl: Option<i32>,
    #[serde(default)]
    ctx: Option<u32>,
    #[serde(default)]
    flash_attn: Option<bool>,
    #[serde(default)]
    temp: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    // Speculative decoding — empty string or absent = disable draft
    #[serde(default)]
    draft_model: Option<String>,
    #[serde(default)]
    draft_max: Option<u32>,
    #[serde(default)]
    gpu_layers_draft: Option<i32>,
}

#[derive(Deserialize)]
struct ParamsReq {
    #[serde(default)]
    temp: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
}

/// llama-server streaming chunk
#[derive(Deserialize)]
struct ChatChunk {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
}
#[derive(Deserialize)]
struct ChunkChoice {
    delta: ChunkDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}
#[derive(Deserialize)]
struct ChunkDelta {
    #[serde(default)]
    content: Option<String>,
}

// ── Main ────────────────────────────────────────────────────

fn main() {
    let config_path = env::args()
        .skip_while(|a| a != "--config").nth(1)
        .unwrap_or_else(|| "config.toml".into());

    let file_cfg: FileConfig = fs::read_to_string(&config_path)
        .ok()
        .and_then(|s| toml::from_str(&s).map_err(|e| eprintln!("config parse: {e}")).ok())
        .unwrap_or_default();

    let llama_binary = if file_cfg.llama.binary.is_empty() {
        find_llama_binary(&file_cfg.defaults.models_dir)
    } else {
        file_cfg.llama.binary.clone()
    };

    let models = discover_models(&file_cfg.defaults.models_dir, &file_cfg.models, &file_cfg.defaults);

    eprintln!("\n  CODEWRITER");
    eprintln!("  {} models in {}/", models.len(), file_cfg.defaults.models_dir);
    for m in &models {
        eprintln!("    {} [{}] ngl={} ctx={}", m.name, m.family, m.gpu_layers, m.context_size);
    }
    eprintln!("  llama-server: {llama_binary}");
    if !file_cfg.defaults.cache_type_k.is_empty() || !file_cfg.defaults.cache_type_v.is_empty() {
        eprintln!("  kv-cache: k={} v={}",
            if file_cfg.defaults.cache_type_k.is_empty() { "default" } else { &file_cfg.defaults.cache_type_k },
            if file_cfg.defaults.cache_type_v.is_empty() { "default" } else { &file_cfg.defaults.cache_type_v });
    }
    if !file_cfg.defaults.draft_model.is_empty() {
        eprintln!("  speculative: {} (draft-max={}, ngl={})",
            file_cfg.defaults.draft_model, file_cfg.defaults.draft_max, file_cfg.defaults.gpu_layers_draft);
    }

    let llama_ok = Command::new(&llama_binary)
        .arg("--help").stdout(Stdio::null()).stderr(Stdio::null())
        .status().map(|s| s.success()).unwrap_or(false);
    if !llama_ok {
        eprintln!("  WARNING: '{llama_binary}' not found or not executable");
    }

    let mut cfg = RuntimeCfg {
        port: file_cfg.server.port,
        llama_binary,
        llama_port: file_cfg.llama.port,
        parallel_slots: file_cfg.llama.parallel_slots,
        startup_timeout: file_cfg.llama.startup_timeout,
        session_limit: file_cfg.limits.session_tokens,
        daily_limit: file_cfg.limits.daily_tokens,
        models_dir: file_cfg.defaults.models_dir.clone(),
        active_model: String::new(),
        ngl: file_cfg.defaults.gpu_layers,
        ctx: if file_cfg.defaults.context_size > 0 { file_cfg.defaults.context_size } else { 4096 },
        flash_attn: file_cfg.defaults.flash_attention,
        temp: file_cfg.defaults.temperature,
        top_k: file_cfg.defaults.top_k,
        top_p: file_cfg.defaults.top_p,
        repeat_penalty: file_cfg.defaults.repeat_penalty,
        cache_type_k: file_cfg.defaults.cache_type_k.clone(),
        cache_type_v: file_cfg.defaults.cache_type_v.clone(),
        draft_model: file_cfg.defaults.draft_model.clone(),
        draft_max: file_cfg.defaults.draft_max,
        gpu_layers_draft: file_cfg.defaults.gpu_layers_draft,
    };

    let mut llama = LlamaServer::new();

    // Auto-load first/default model
    if !models.is_empty() && llama_ok {
        let target = if !file_cfg.defaults.model.is_empty() {
            models.iter().find(|m| m.filename == file_cfg.defaults.model)
        } else {
            Some(&models[0])
        };
        if let Some(m) = target {
            apply_model_params(&mut cfg, m);
            if llama.start(&cfg, m).is_ok() {
                llama.wait_ready(cfg.llama_port, cfg.startup_timeout);
            }
        }
    }

    let addr = format!("127.0.0.1:{}", cfg.port);
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("bind {addr}: {e}");
        std::process::exit(1);
    });
    eprintln!("  http://{addr}\n");

    let shared: Shared = Arc::new(Mutex::new(State {
        cfg, models, llama, tokens_session: 0, requests: 0,
    }));

    for stream in listener.incoming().flatten() {
        let st = Arc::clone(&shared);
        std::thread::spawn(move || serve(stream, &st));
    }
}

fn apply_model_params(cfg: &mut RuntimeCfg, m: &Model) {
    cfg.active_model = m.filename.clone();
    cfg.ngl = m.gpu_layers;
    cfg.ctx = if m.context_size > 0 { m.context_size } else { 4096 };
    cfg.flash_attn = m.flash_attention;
    cfg.temp = m.temperature;
    cfg.top_k = m.top_k;
    cfg.top_p = m.top_p;
    cfg.repeat_penalty = m.repeat_penalty;
}

// ── HTTP server ─────────────────────────────────────────────

fn serve(mut stream: TcpStream, st: &Shared) {
    // Longer timeout for large file uploads
    let _ = stream.set_read_timeout(Some(Duration::from_secs(120)));
    let mut reader = BufReader::new(&stream);

    // Parse request line
    let mut req_line = String::new();
    if reader.read_line(&mut req_line).is_err() { return; }
    let parts: Vec<&str> = req_line.trim().split_whitespace().collect();
    if parts.len() < 2 { return; }
    let (method, path) = (parts[0], parts[1]);

    // Read headers, extract content-length
    let mut content_len = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() || line.trim().is_empty() { break; }
        if let Some(rest) = line.to_lowercase().strip_prefix("content-length:") {
            content_len = rest.trim().parse().unwrap_or(0);
        }
    }

    // Enforce body size limit
    if content_len > MAX_BODY_BYTES {
        respond(&mut stream, 413, "text/plain", "payload too large");
        return;
    }

    // Read body
    let mut body_bytes = vec![0u8; content_len];
    if content_len > 0 { let _ = reader.read_exact(&mut body_bytes); }
    let body = String::from_utf8_lossy(&body_bytes).to_string();

    match (method, path) {
        ("GET", "/")           => respond(&mut stream, 200, "text/html", INDEX),
        ("GET", "/style.css")  => respond(&mut stream, 200, "text/css", STYLE),
        ("GET", "/script.js")  => respond(&mut stream, 200, "text/javascript", SCRIPT),
        ("GET", "/api/models") => respond_json(&mut stream, &handle_models(st)),
        ("GET", "/api/status") => respond_json(&mut stream, &handle_status(st)),
        ("POST", "/api/load")  => respond_json(&mut stream, &handle_load(st, &body)),
        ("POST", "/api/stop")  => respond_json(&mut stream, &handle_stop(st)),
        ("POST", "/api/params") => respond_json(&mut stream, &handle_params(st, &body)),
        ("POST", "/api/write") => handle_write_stream(&mut stream, st, &body),
        _ => respond(&mut stream, 404, "text/plain", "not found"),
    }
}

fn respond(s: &mut TcpStream, code: u16, ct: &str, body: &str) {
    let status = match code {
        200 => "OK",
        413 => "Payload Too Large",
        _ => "Not Found",
    };
    let _ = write!(
        s, "HTTP/1.1 {code} {status}\r\nContent-Type: {ct}\r\nContent-Length: {}\r\n\
            Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
}

fn respond_json(s: &mut TcpStream, val: &serde_json::Value) {
    let body = serde_json::to_string(val).unwrap_or_else(|_| "{}".into());
    respond(s, 200, "application/json", &body);
}

// ── Handlers ────────────────────────────────────────────────

fn handle_models(st: &Shared) -> serde_json::Value {
    let s = st.lock().unwrap();
    // Build draft candidate list: all .gguf files in models_dir
    let draft_candidates: Vec<String> = fs::read_dir(&s.cfg.models_dir)
        .into_iter().flatten().flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            if name.ends_with(".gguf") { Some(name) } else { None }
        })
        .collect();

    serde_json::json!({
        "models": s.models,
        "active": s.cfg.active_model,
        "draft_candidates": draft_candidates,
        "draft": {
            "model": if s.cfg.draft_model.is_empty() { None } else { Some(&s.cfg.draft_model) },
            "max": s.cfg.draft_max,
            "ngl": s.cfg.gpu_layers_draft,
        },
        "params": {
            "ngl": s.cfg.ngl, "ctx": s.cfg.ctx, "flash_attn": s.cfg.flash_attn,
            "temp": s.cfg.temp, "top_k": s.cfg.top_k, "top_p": s.cfg.top_p,
            "repeat_penalty": s.cfg.repeat_penalty,
            "cache_type_k": if s.cfg.cache_type_k.is_empty() { None } else { Some(&s.cfg.cache_type_k) },
            "cache_type_v": if s.cfg.cache_type_v.is_empty() { None } else { Some(&s.cfg.cache_type_v) },
        },
        "llama": s.llama.status_json(),
    })
}

fn handle_status(st: &Shared) -> serde_json::Value {
    let s = st.lock().unwrap();
    serde_json::json!({
        "tokens_session": s.tokens_session,
        "requests": s.requests,
        "model": s.cfg.active_model,
        "ctx": s.cfg.ctx,
        "has_model": s.cfg.has_model(),
        "llama": s.llama.status_json(),
    })
}

fn handle_load(st: &Shared, body: &str) -> serde_json::Value {
    let req: LoadReq = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(e) => return serde_json::json!({"error": e.to_string()}),
    };

    let (model, mut cfg) = {
        let s = st.lock().unwrap();
        let Some(m) = s.models.iter().find(|m| m.filename == req.model).cloned() else {
            return serde_json::json!({"error": format!("model '{}' not found", req.model)});
        };
        (m, s.cfg.clone())
    };

    apply_model_params(&mut cfg, &model);
    // Override with request-specific params
    if let Some(v) = req.ngl { cfg.ngl = v; }
    if let Some(v) = req.ctx { cfg.ctx = v.max(2048); }
    if let Some(v) = req.flash_attn { cfg.flash_attn = v; }
    if let Some(v) = req.temp { cfg.temp = v; }
    if let Some(v) = req.top_k { cfg.top_k = v; }
    if let Some(v) = req.top_p { cfg.top_p = v; }
    if let Some(v) = req.repeat_penalty { cfg.repeat_penalty = v; }
    // Draft model — empty string disables speculative decoding
    if let Some(v) = req.draft_model { cfg.draft_model = v; }
    if let Some(v) = req.draft_max { cfg.draft_max = v; }
    if let Some(v) = req.gpu_layers_draft { cfg.gpu_layers_draft = v; }

    // Stop current, start new (hold lock briefly)
    st.lock().unwrap().llama.stop();

    let mut llama = LlamaServer::new();
    if let Err(e) = llama.start(&cfg, &model) {
        st.lock().unwrap().cfg.active_model.clear();
        return serde_json::json!({"error": e});
    }

    let ok = llama.wait_ready(cfg.llama_port, cfg.startup_timeout);
    let status = llama.status_json();
    {
        let mut s = st.lock().unwrap();
        s.llama = llama;
        s.cfg = cfg;
    }
    serde_json::json!({"ok": ok, "llama": status})
}

fn handle_stop(st: &Shared) -> serde_json::Value {
    let mut s = st.lock().unwrap();
    s.llama.stop();
    s.cfg.active_model.clear();
    serde_json::json!({"ok": true})
}

fn handle_params(st: &Shared, body: &str) -> serde_json::Value {
    let req: ParamsReq = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(e) => return serde_json::json!({"error": e.to_string()}),
    };
    let mut s = st.lock().unwrap();
    if let Some(v) = req.temp { s.cfg.temp = v; }
    if let Some(v) = req.top_k { s.cfg.top_k = v; }
    if let Some(v) = req.top_p { s.cfg.top_p = v; }
    if let Some(v) = req.repeat_penalty { s.cfg.repeat_penalty = v; }
    serde_json::json!({
        "ok": true,
        "temp": s.cfg.temp, "top_k": s.cfg.top_k,
        "top_p": s.cfg.top_p, "repeat_penalty": s.cfg.repeat_penalty,
    })
}

// ── Relevance scoring for file ordering ─────────────────────

fn relevance_score(file: &FileEntry, description: &str, target_lang: &str) -> u32 {
    let mut score = 0u32;
    let file_lang = file.language.to_lowercase();
    let target = target_lang.to_lowercase();

    // Same language as target
    if file_lang == target { score += 10; }

    // Filename stem mentioned in description
    let fname_lower = file.name.to_lowercase();
    let stem = fname_lower.rsplit('/').next().unwrap_or(&fname_lower);
    let stem = stem.rsplit('.').last().unwrap_or(stem);
    let desc_lower = description.to_lowercase();
    if stem.len() > 2 && desc_lower.contains(stem) {
        score += 20;
    }

    // Content words from description found in file
    for word in desc_lower.split_whitespace() {
        if word.len() > 3 && file.content.contains(word) {
            score += 2;
        }
    }

    score
}

// ── Token-budget-aware context assembly ─────────────────────

struct ContextResult {
    context_block: String,
    files_included: Vec<String>,
    files_truncated: Vec<String>,
    files_dropped: Vec<String>,
    total_input_tokens: u64,
    model_ctx: u64,
}

/// Minimum tokens to leave for the model to respond.
const MIN_OUTPUT_TOKENS: u64 = 256;

fn assemble_context(
    files: &[FileEntry],
    extra_ctx: &str,
    description: &str,
    target_lang: &str,
    model_ctx: u32,
    system_text: &str,
) -> ContextResult {
    let system_tok = estimate_tokens(system_text);
    let desc_tok = estimate_tokens(description);
    let fixed_input = system_tok + desc_tok;
    // How many tokens files/context can use: everything except the fixed input and a minimum for output
    let file_budget = (model_ctx as u64).saturating_sub(fixed_input + MIN_OUTPUT_TOKENS);

    let mut result = ContextResult {
        context_block: String::new(),
        files_included: Vec::new(),
        files_truncated: Vec::new(),
        files_dropped: Vec::new(),
        total_input_tokens: fixed_input,
        model_ctx: model_ctx as u64,
    };

    // Sort files by relevance score (descending)
    let mut scored: Vec<(usize, u32)> = files.iter().enumerate()
        .map(|(i, f)| (i, relevance_score(f, description, target_lang)))
        .collect();
    scored.sort_by(|a, b| b.1.cmp(&a.1));

    let mut files_used: u64 = 0;

    // Fill file_budget with files in relevance order
    for (idx, _score) in &scored {
        let f = &files[*idx];
        let lang_tag = if f.language.is_empty() { "text" } else { &f.language };
        let block = format!("\n--- {} ---\n```{}\n{}\n```\n", f.name, lang_tag, f.content);
        let cost = estimate_tokens(&block);

        if files_used + cost <= file_budget {
            // Fits entirely
            result.context_block.push_str(&block);
            files_used += cost;
            result.files_included.push(f.name.clone());
        } else if files_used < file_budget {
            // Partial fit — truncate content to fill remaining space
            let remaining_tokens = file_budget - files_used;
            let overhead: u64 = 15;
            if remaining_tokens > overhead {
                let content_tokens = remaining_tokens - overhead;
                let max_chars = (content_tokens as f64 * 3.2) as usize;
                let truncated = if max_chars < f.content.len() {
                    let slice = &f.content[..max_chars.min(f.content.len())];
                    if let Some(nl) = slice.rfind('\n') {
                        &f.content[..nl + 1]
                    } else {
                        slice
                    }
                } else {
                    &f.content
                };
                let block = format!(
                    "\n--- {} (truncated) ---\n```{}\n{}\n```\n",
                    f.name, lang_tag, truncated
                );
                let actual_cost = estimate_tokens(&block);
                result.context_block.push_str(&block);
                files_used += actual_cost;
                result.files_truncated.push(f.name.clone());
            } else {
                result.files_dropped.push(f.name.clone());
            }
        } else {
            result.files_dropped.push(f.name.clone());
        }
    }

    // Append extra text context if it fits
    if !extra_ctx.is_empty() {
        let block = format!("\n--- additional context ---\n```\n{}\n```\n", extra_ctx);
        let cost = estimate_tokens(&block);
        if files_used + cost <= file_budget {
            result.context_block.push_str(&block);
            files_used += cost;
        } else if files_used < file_budget {
            let remaining_tokens = file_budget - files_used;
            let overhead: u64 = 15;
            if remaining_tokens > overhead {
                let content_tokens = remaining_tokens - overhead;
                let max_chars = (content_tokens as f64 * 3.2) as usize;
                let truncated = if max_chars < extra_ctx.len() {
                    let slice = &extra_ctx[..max_chars.min(extra_ctx.len())];
                    if let Some(nl) = slice.rfind('\n') {
                        &extra_ctx[..nl + 1]
                    } else {
                        slice
                    }
                } else {
                    extra_ctx
                };
                let block = format!(
                    "\n--- additional context (truncated) ---\n```\n{}\n```\n",
                    truncated
                );
                let actual_cost = estimate_tokens(&block);
                result.context_block.push_str(&block);
                files_used += actual_cost;
            }
        }
    }

    result.total_input_tokens = fixed_input + files_used;

    result
}

// ── Streaming write ─────────────────────────────────────────

fn handle_write_stream(stream: &mut TcpStream, st: &Shared, body: &str) {
    let req: WriteReq = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(e) => { send_sse_error(stream, &e.to_string()); return; }
    };
    if req.description.is_empty() {
        send_sse_error(stream, "No description provided");
        return;
    }

    let cfg = {
        let s = st.lock().unwrap();
        if !s.cfg.has_model() || !s.llama.is_ready() {
            let msg = if !s.cfg.has_model() { "No model loaded" } else { "Model not ready" };
            drop(s);
            send_sse_error(stream, msg);
            return;
        }
        s.cfg.clone()
    };

    // Send SSE headers immediately
    let _ = write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
         Cache-Control: no-cache\r\nConnection: keep-alive\r\n\
         Access-Control-Allow-Origin: *\r\n\r\n"
    );
    let _ = stream.flush();

    let is_review = req.mode == "review";

    let system = if is_review {
        format!(
            "You are an expert {} code reviewer and software engineer.\n\
             Analyze the provided code thoroughly. Discuss:\n\
             - Correctness and potential bugs\n\
             - Performance issues and optimization opportunities\n\
             - Security concerns\n\
             - Code style and readability\n\
             - Concrete suggestions for improvement\n\n\
             Be specific — reference functions, types, and line-level details.\n\
             Use markdown code fences (```) when showing code snippets.\n\
             Focus on actionable feedback, not generic advice.",
            req.language
        )
    } else {
        format!(
            "You are an expert {} programmer. Write clean, efficient, well-documented code.\n\
             Output ONLY the code with clear comments. No markdown fences, no prose outside code.",
            req.language
        )
    };

    // Assemble context — packs as much input as possible, leaving MIN_OUTPUT_TOKENS for response
    let ctx_result = assemble_context(
        &req.files,
        &req.context,
        &req.description,
        &req.language,
        cfg.ctx,
        &system,
    );

    // Send context_info event so the frontend knows what fit
    let remaining = ctx_result.model_ctx.saturating_sub(ctx_result.total_input_tokens);
    if !req.files.is_empty() || !req.context.is_empty() {
        send_sse(stream, &serde_json::json!({
            "context_info": {
                "model_ctx": ctx_result.model_ctx,
                "input_tokens": ctx_result.total_input_tokens,
                "remaining_tokens": remaining,
                "files_included": ctx_result.files_included,
                "files_truncated": ctx_result.files_truncated,
                "files_dropped": ctx_result.files_dropped,
            }
        }));
    }

    let user = if is_review {
        if ctx_result.context_block.is_empty() {
            format!("{}", req.description)
        } else {
            format!(
                "{}\n\nCode to review:{}\n",
                req.description, ctx_result.context_block
            )
        }
    } else {
        if ctx_result.context_block.is_empty() {
            format!("Write {} code for: {}", req.language, req.description)
        } else {
            format!(
                "Write {} code for: {}\n\nExisting code context:{}\n",
                req.language, req.description, ctx_result.context_block
            )
        }
    };

    // max_tokens = whatever context space is left after actual input
    let actual_input = estimate_tokens(&system) + estimate_tokens(&user);
    let max_tokens = (cfg.ctx as u64).saturating_sub(actual_input);

    if max_tokens < MIN_OUTPUT_TOKENS {
        send_sse(stream, &serde_json::json!({
            "error": format!(
                "Context full — input uses ~{} of {} tokens, only {} left for output. \
                 Remove files or increase context size.",
                actual_input, cfg.ctx, max_tokens
            )
        }));
        return;
    }

    let llama_req = serde_json::json!({
        "model": "local",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": cfg.temp,
        "top_p": cfg.top_p,
        "repeat_penalty": cfg.repeat_penalty,
        "stream": true,
    });

    let tmp = format!("/tmp/cw_{}.json", std::process::id());
    if fs::write(&tmp, llama_req.to_string()).is_err() {
        send_sse(stream, &serde_json::json!({"error": "failed to write temp file"}));
        return;
    }

    let child = Command::new(curl_cmd())
        .args([
            "-s", "--no-buffer", "-X", "POST",
            &cfg.endpoint(),
            "-H", "content-type: application/json",
            "--max-time", "300",
            "-d", &format!("@{tmp}"),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn();

    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            send_sse(stream, &serde_json::json!({"error": format!("curl: {e}")}));
            let _ = fs::remove_file(&tmp);
            return;
        }
    };

    let stdout = child.stdout.take().unwrap();
    let reader = BufReader::new(stdout);
    let t0 = Instant::now();
    let mut token_count = 0u64;

    for line in reader.lines().flatten() {
        let Some(data) = line.strip_prefix("data: ") else { continue };
        if data == "[DONE]" { break; }

        if let Ok(chunk) = serde_json::from_str::<ChatChunk>(data) {
            if let Some(content) = chunk.choices.first().and_then(|c| c.delta.content.as_deref()) {
                if !content.is_empty() {
                    token_count += 1;
                    send_sse(stream, &serde_json::json!({"token": content}));
                }
            }
        }
    }

    let elapsed_ms = t0.elapsed().as_millis() as u64;
    send_sse(stream, &serde_json::json!({
        "done": true, "tokens": token_count, "elapsed_ms": elapsed_ms,
    }));

    let _ = fs::remove_file(&tmp);
    let _ = child.wait();

    // Update usage
    let mut s = st.lock().unwrap();
    s.tokens_session += token_count;
    s.requests += 1;
}

fn send_sse(stream: &mut TcpStream, val: &serde_json::Value) {
    let data = serde_json::to_string(val).unwrap_or_default();
    let _ = write!(stream, "data: {data}\n\n");
    let _ = stream.flush();
}

fn send_sse_error(stream: &mut TcpStream, msg: &str) {
    let _ = write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
         Cache-Control: no-cache\r\nConnection: keep-alive\r\n\
         Access-Control-Allow-Origin: *\r\n\r\n"
    );
    send_sse(stream, &serde_json::json!({"error": msg}));
}

// ── Embedded assets ─────────────────────────────────────────

const INDEX: &str = include_str!("index.html");
const STYLE: &str = include_str!("style.css");
const SCRIPT: &str = include_str!("app.js");
