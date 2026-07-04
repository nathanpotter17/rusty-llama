use serde::{Deserialize, Serialize};
use std::{
    collections::{BinaryHeap, HashSet},
    env, fs,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::Path,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};



// ── Config ──────────────────────────────────────────────────

const MAX_BODY_BYTES: usize = 10 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(default)]
struct FileConfig {
    server: ServerCfg,
    llama: LlamaCfg,
    defaults: DefaultsCfg,
    limits: LimitsCfg,
    embed: EmbedCfg,
    rag: RagCfg,
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
            embed: EmbedCfg::default(),
            rag: RagCfg::default(),
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
    draft_model: String,
    spec_type: String,
    spec_draft_n_max: u32,
    gpu_layers_draft: i32,
}
impl Default for DefaultsCfg {
    fn default() -> Self {
        Self {
            model: String::new(), models_dir: "models".into(), gpu_layers: -1,
            context_size: 0, flash_attention: true, temperature: 0.7,
            top_k: 40, top_p: 0.9, repeat_penalty: 1.1,
            cache_type_k: String::new(), cache_type_v: String::new(),
            draft_model: String::new(), spec_type: String::new(), spec_draft_n_max: 2, gpu_layers_draft: 99,
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

/// Dedicated embedding server — runs a second llama-server process
/// on a separate port with --embedding enabled.
#[derive(Deserialize, Clone)]
#[serde(default)]
struct EmbedCfg {
    enabled: bool,
    model: String,         // .gguf filename inside models_dir
    port: u16,
    gpu_layers: i32,
    context_size: u32,
    parallel_slots: u32,
    startup_timeout: u64,
    pooling: String,       // "mean", "cls", "last", or "" (server default)
    query_prefix: String,  // prepended to search queries before embedding
    doc_prefix: String,    // prepended to documents/chunks before embedding
}
impl Default for EmbedCfg {
    fn default() -> Self {
        Self {
            enabled: false,
            model: String::new(),
            port: 8078,
            gpu_layers: 99,
            context_size: 2048,
            parallel_slots: 2,
            startup_timeout: 60,
            pooling: String::new(),
            query_prefix: String::new(),
            doc_prefix: String::new(),
        }
    }
}

#[derive(Deserialize, Clone)]
#[serde(default)]
struct RagCfg {
    enabled: bool,
    db_path: String,
    chunk_size: usize,
    chunk_overlap: usize,
    search_results: usize,
    // Retrieval quality
    min_similarity: f32,         // skip chunks below this cosine similarity
    hybrid_weight_vector: f32,   // weight for vector similarity in hybrid scoring
    hybrid_weight_bm25: f32,     // weight for BM25 keyword score in hybrid scoring
    // External chunker tool (empty = use internal chunker only)
    chunker_tool: String,
    // HNSW graph parameters
    hnsw_m: usize,               // max connections per node per layer (M0 = 2*M for layer 0)
    hnsw_ef_construction: usize, // beam width during index build
    hnsw_ef_search: usize,       // beam width during query (higher = more accurate, slower)
}
impl Default for RagCfg {
    fn default() -> Self {
        Self {
            enabled: true,
            db_path: "data/rag_index.json".into(),
            chunk_size: 60,
            chunk_overlap: 10,
            search_results: 5,
            min_similarity: 0.25,
            hybrid_weight_vector: 0.7,
            hybrid_weight_bm25: 0.3,
            chunker_tool: "tools/chunker.py".into(),
            hnsw_m: 16,
            hnsw_ef_construction: 150,
            hnsw_ef_search: 64,
        }
    }
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
    // Speculative decoding (per-model — spec capability is a model property).
    #[serde(default)]
    spec_type: String,               // "" = off | "draft-mtp" | "draft-model" | "eagle" | ...
    #[serde(default = "def_spec_nmax")]
    spec_draft_n_max: u32,           // --spec-draft-n-max
    #[serde(default)]
    draft_model: String,             // used only when spec_type == "draft-model"
    #[serde(default = "def_ngl_draft")]
    gpu_layers_draft: i32,           // draft-model offload
}

fn def_family() -> String { "unknown".into() }
fn def_ngl() -> i32 { 15 }
fn def_ctx() -> u32 { 4096 }
fn def_true() -> bool { true }
fn def_temp() -> f32 { 0.7 }
fn def_topk() -> u32 { 40 }
fn def_topp() -> f32 { 0.9 }
fn def_rp() -> f32 { 1.1 }
fn def_spec_nmax() -> u32 { 2 }
fn def_ngl_draft() -> i32 { 99 }

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
    active_model: String,
    ngl: i32,
    ctx: u32,
    flash_attn: bool,
    temp: f32,
    top_k: u32,
    top_p: f32,
    repeat_penalty: f32,
    cache_type_k: String,
    cache_type_v: String,
    draft_model: String,
    spec_type: String,
    spec_draft_n_max: u32,
    gpu_layers_draft: i32,
    threads: usize,           // generation threads, derived from SystemInfo
    auto_ctx: u32,            // VRAM-derived context default (used when ctx unset)
    // Embed server config (cloned from EmbedCfg)
    embed: EmbedCfg,
}

impl RuntimeCfg {
    fn endpoint(&self) -> String {
        format!("http://127.0.0.1:{}/v1/chat/completions", self.llama_port)
    }
    fn embedding_endpoint(&self) -> String {
        if self.embed.enabled && !self.embed.model.is_empty() {
            format!("http://127.0.0.1:{}/v1/embeddings", self.embed.port)
        } else {
            // Fallback: use main model (requires --embedding on main server)
            format!("http://127.0.0.1:{}/v1/embeddings", self.llama_port)
        }
    }
    fn has_model(&self) -> bool { !self.active_model.is_empty() }
}

// ── Discovered model ────────────────────────────────────────

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
    spec_type: String,
    spec_draft_n_max: u32,
    draft_model: String,
    gpu_layers_draft: i32,
}

fn discover_models(dir: &str, known: &[ModelEntry], defaults: &DefaultsCfg, exclude: &[&str]) -> Vec<Model> {
    let Ok(entries) = fs::read_dir(dir) else {
        eprintln!("  models dir '{dir}' not found");
        return Vec::new();
    };
    let mut models: Vec<Model> = entries
        .flatten()
        .filter_map(|e| {
            let fname = e.file_name().to_string_lossy().to_string();
            if !fname.ends_with(".gguf") { return None; }
            if exclude.iter().any(|ex| ex.eq_ignore_ascii_case(&fname)) { return None; }
            let path = e.path().to_string_lossy().to_string();
            Some(if let Some(k) = known.iter().find(|m| m.filename == fname) {
                Model {
                    filename: fname, path,
                    name: if k.name.is_empty() { pretty_name(&k.filename) } else { k.name.clone() },
                    family: k.family.clone(), gpu_layers: k.gpu_layers,
                    context_size: k.context_size, flash_attention: k.flash_attention,
                    temperature: k.temperature, top_k: k.top_k, top_p: k.top_p,
                    repeat_penalty: k.repeat_penalty,
                    spec_type: k.spec_type.clone(), spec_draft_n_max: k.spec_draft_n_max,
                    draft_model: k.draft_model.clone(), gpu_layers_draft: k.gpu_layers_draft,
                }
            } else {
                Model {
                    name: pretty_name(&fname), filename: fname, path, family: "unknown".into(),
                    gpu_layers: defaults.gpu_layers,
                    context_size: if defaults.context_size > 0 { defaults.context_size } else { 4096 },
                    flash_attention: defaults.flash_attention, temperature: defaults.temperature,
                    top_k: defaults.top_k, top_p: defaults.top_p, repeat_penalty: defaults.repeat_penalty,
                    spec_type: defaults.spec_type.clone(), spec_draft_n_max: defaults.spec_draft_n_max,
                    draft_model: defaults.draft_model.clone(), gpu_layers_draft: defaults.gpu_layers_draft,
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

fn estimate_tokens(s: &str) -> u64 {
    estimate_tokens_lang(s, "")
}

/// Per-language token estimation — code-heavy languages pack more tokens
/// per character due to operators, short identifiers, and punctuation.
fn estimate_tokens_lang(s: &str, lang: &str) -> u64 {
    let ratio = match lang {
        "rust" | "c" | "c++" | "cpp" | "java" | "csharp" => 2.6,
        "go" | "swift" | "kotlin" | "zig" => 2.8,
        "javascript" | "typescript" => 2.9,
        "python" | "ruby" | "lua" => 3.4,
        "html" | "xml" | "css" | "scss" => 3.0,
        "sql" | "graphql" => 3.0,
        "bash" | "sh" => 3.0,
        "markdown" | "md" | "text" => 3.8,
        _ => 3.2,
    };
    (s.len() as f64 / ratio).ceil() as u64
}

// ── RAG: Text chunking ─────────────────────────────────────

/// Simple line-window chunker — used as fallback when the external
/// chunker tool is unavailable or fails.
fn chunk_code_file_simple(name: &str, content: &str, chunk_size: usize, overlap: usize) -> Vec<(String, String, String, String)> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() { return Vec::new(); }
    let mut chunks = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let end = std::cmp::min(i + chunk_size, lines.len());
        let chunk = lines[i..end].join("\n");
        let label = format!("{}:{}-{}", name, i + 1, end);
        // Add metadata header for better embedding quality
        let enriched = enrich_chunk_metadata(name, &chunk, i + 1, end);
        chunks.push((label, enriched, "block".to_string(), name.to_string()));
        if end == lines.len() { break; }
        i += chunk_size.saturating_sub(overlap);
    }
    chunks
}

// Text-domain (chat) embedding prefixes. The configured [embed] prefixes are
// code-retrieval instructions; prose retrieval needs its own framing. These
// are domain defaults, not user config.
const TEXT_QUERY_PREFIX: &str = "Instruct: Retrieve passages relevant to the question\nQuery: ";
const TEXT_DOC_PREFIX: &str = "";

// Prose window sizing (in words). Smaller, focused windows retrieve better
// than code-sized blocks for natural-language Q&A.
const TEXT_CHUNK_WORDS: usize = 180;
const TEXT_OVERLAP_WORDS: usize = 30;

/// Prose-aware chunker for the "text" domain: packs whitespace-delimited
/// words into overlapping windows on paragraph-friendly boundaries. No code
/// construct detection or metadata header. Returns (source, text, kind, file).
fn chunk_text_file(name: &str, content: &str) -> Vec<(String, String, String, String)> {
    let words: Vec<&str> = content.split_whitespace().collect();
    if words.is_empty() { return Vec::new(); }
    let step = TEXT_CHUNK_WORDS.saturating_sub(TEXT_OVERLAP_WORDS).max(1);
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut idx = 0;
    while start < words.len() {
        let end = (start + TEXT_CHUNK_WORDS).min(words.len());
        let text = words[start..end].join(" ");
        let label = format!("{name}#{idx}");
        chunks.push((label, text, "text".to_string(), name.to_string()));
        if end == words.len() { break; }
        start += step;
        idx += 1;
    }
    chunks
}

/// Detect the display language name from a filename extension.
fn lang_display_name(filename: &str) -> &'static str {
    let ext = filename.rsplit('.').next().unwrap_or("");
    match ext {
        "rs" => "Rust", "py" => "Python", "js" | "jsx" | "mjs" => "JavaScript",
        "ts" | "tsx" => "TypeScript", "go" => "Go", "c" | "h" => "C",
        "cpp" | "cc" | "cxx" | "hpp" => "C++", "java" => "Java",
        "cs" => "C#", "rb" => "Ruby", "php" => "PHP",
        "swift" => "Swift", "kt" | "kts" => "Kotlin", "zig" => "Zig",
        "lua" => "Lua", "sh" | "bash" => "Bash", "sql" => "SQL",
        "html" | "htm" => "HTML", "css" | "scss" => "CSS",
        "json" => "JSON", "yaml" | "yml" => "YAML", "toml" => "TOML",
        "md" | "markdown" => "Markdown", _ => "Text",
    }
}

/// Detect what constructs a chunk contains for metadata enrichment.
fn detect_chunk_contents(text: &str) -> String {
    let mut tags: Vec<&str> = Vec::new();
    if text.contains("fn ") || text.contains("def ") || text.contains("function ") || text.contains("func ") {
        tags.push("functions");
    }
    if text.contains("struct ") || text.contains("class ") || text.contains("interface ") {
        tags.push("types");
    }
    if text.contains("enum ") { tags.push("enums"); }
    if text.contains("impl ") { tags.push("impl"); }
    if text.contains("trait ") { tags.push("traits"); }
    let has_test = text.contains("#[test]") || text.contains("#[cfg(test)]")
        || text.contains("def test_") || text.contains("describe(")
        || text.contains("@Test") || text.contains("@test");
    if has_test { tags.push("tests"); }
    if text.contains("use ") || text.contains("import ") || text.contains("require(") || text.contains("#include") {
        tags.push("imports");
    }
    if text.contains("Error") || text.contains("Result<") || text.contains("unwrap(")
        || text.contains("expect(") || text.contains("panic!") || text.contains("try ")
        || text.contains("catch ") || text.contains("except ") {
        tags.push("error_handling");
    }
    if tags.is_empty() { "code".into() } else { tags.join(", ") }
}

/// Add metadata header to a chunk for better embedding quality.
fn enrich_chunk_metadata(filename: &str, text: &str, start_line: usize, end_line: usize) -> String {
    let lang = lang_display_name(filename);
    let contents = detect_chunk_contents(text);
    format!("File: {} | Language: {} | Lines: {}-{} | Contains: {}\n{}",
        filename, lang, start_line, end_line, contents, text)
}

/// Try to run the external chunker tool (Python script) for syntax-aware chunking.
/// Returns None if the tool is not available or fails.
fn try_external_chunker(
    tool_path: &str, files: &[FileEntry], chunk_size: usize, overlap: usize,
) -> Option<Vec<(String, String, String, String)>> {
    if tool_path.is_empty() { return None; }

    // Check tool exists
    let path = Path::new(tool_path);
    if !path.exists() {
        eprintln!("[rag] chunker tool not found: {tool_path}");
        return None;
    }

    // Serialize files to JSON
    let input: Vec<serde_json::Value> = files.iter().map(|f| {
        serde_json::json!({
            "name": f.name,
            "content": f.content,
            "language": f.language,
        })
    }).collect();
    let json_input = match serde_json::to_string(&input) {
        Ok(s) => s,
        Err(e) => { eprintln!("[rag] chunker serialize error: {e}"); return None; }
    };

    // Spawn the chunker process
    let t0 = Instant::now();
    let mut child = match Command::new("python3")
        .arg(tool_path)
        .arg("--chunk-size").arg(chunk_size.to_string())
        .arg("--overlap").arg(overlap.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[rag] chunker spawn error: {e}");
            return None;
        }
    };

    // Write input to stdin
    if let Some(mut stdin) = child.stdin.take() {
        if stdin.write_all(json_input.as_bytes()).is_err() {
            eprintln!("[rag] chunker stdin write error");
            return None;
        }
    }

    // Read output
    let output = match child.wait_with_output() {
        Ok(o) => o,
        Err(e) => { eprintln!("[rag] chunker wait error: {e}"); return None; }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("[rag] chunker exited with error: {}", stderr.trim());
        return None;
    }

    // Parse output JSON: [{"source": "...", "text": "...", "kind": "...", "file": "..."}, ...]
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: Vec<serde_json::Value> = match serde_json::from_str(&stdout) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[rag] chunker output parse error: {e}");
            return None;
        }
    };

    let mut result: Vec<(String, String, String, String)> = Vec::with_capacity(parsed.len());
    for item in &parsed {
        let source = item["source"].as_str().unwrap_or("unknown").to_string();
        let text   = item["text"].as_str().unwrap_or("").to_string();
        let kind   = item["kind"].as_str().unwrap_or("block").to_string();
        let file   = item["file"].as_str().unwrap_or("").to_string();
        if !text.trim().is_empty() {
            result.push((source, text, kind, file));
        }
    }

    eprintln!("[rag] external chunker produced {} chunks in {:.1}ms",
        result.len(), t0.elapsed().as_secs_f64() * 1000.0);
    Some(result)
}

// ── RAG: Embedding via dedicated embed server ───────────────

/// Parse "http://host:port/path" into components.
fn parse_endpoint(endpoint: &str) -> Result<(&str, u16, String), String> {
    let without_scheme = endpoint.strip_prefix("http://")
        .ok_or_else(|| format!("bad endpoint: {endpoint}"))?;
    let (host_port, path) = match without_scheme.find('/') {
        Some(i) => (&without_scheme[..i], &without_scheme[i..]),
        None => (without_scheme, "/"),
    };
    let (host, port_str) = host_port.split_once(':')
        .ok_or_else(|| format!("no port in endpoint: {endpoint}"))?;
    let port: u16 = port_str.parse()
        .map_err(|_| format!("bad port: {port_str}"))?;
    Ok((host, port, path.to_string()))
}

fn get_embedding(endpoint: &str, text: &str, prefix: &str) -> Result<Vec<f32>, String> {
    let input = if prefix.is_empty() {
        text.to_string()
    } else {
        format!("{prefix}{text}")
    };
    let (host, port, path) = parse_endpoint(endpoint)?;
    let req_body = serde_json::json!({
        "input": input,
        "model": "local"
    });
    let body_str = req_body.to_string();
    let resp_body = http_post_json(host, port, &path, &body_str, 60)?;
    let resp: serde_json::Value = serde_json::from_str(&resp_body)
        .map_err(|e| format!("parse: {e} — body: {}", &resp_body[..resp_body.len().min(200)]))?;
    parse_single_embedding(&resp)
}

/// Send all texts in a single batched request: { "input": [...], "model": "local" }.
/// Falls back to sequential if the server doesn't support batch input.
fn get_embeddings_batch(endpoint: &str, texts: &[String], prefix: &str) -> Result<Vec<Vec<f32>>, String> {
    if texts.is_empty() { return Ok(Vec::new()); }
    if texts.len() == 1 {
        return get_embedding(endpoint, &texts[0], prefix).map(|v| vec![v]);
    }

    let prefixed: Vec<String> = if prefix.is_empty() {
        texts.to_vec()
    } else {
        texts.iter().map(|t| format!("{prefix}{t}")).collect()
    };

    let (host, port, path) = parse_endpoint(endpoint)?;

    // Try batched request first
    let req_body = serde_json::json!({
        "input": prefixed,
        "model": "local"
    });
    let body_str = req_body.to_string();
    let resp_body = http_post_json(host, port, &path, &body_str, 120)?;
    let resp: serde_json::Value = serde_json::from_str(&resp_body)
        .map_err(|e| format!("parse: {e} — body: {}", &resp_body[..resp_body.len().min(200)]))?;

    // OpenAI-compatible batch: { "data": [{ "embedding": [...] }, ...] }
    if let Some(data) = resp["data"].as_array() {
        if data.len() == texts.len() {
            let mut results = Vec::with_capacity(data.len());
            for (i, item) in data.iter().enumerate() {
                if let Some(arr) = item["embedding"].as_array() {
                    results.push(arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect());
                } else {
                    return Err(format!("missing embedding at index {i}"));
                }
            }
            return Ok(results);
        }
    }

    // Batch not supported — fall back to sequential (still using direct HTTP, not curl)
    eprintln!("[embed] batch response didn't match, falling back to sequential");
    let mut results = Vec::with_capacity(texts.len());
    for (i, text) in texts.iter().enumerate() {
        match get_embedding(endpoint, text, prefix) {
            Ok(v) => results.push(v),
            Err(e) => return Err(format!("embedding #{i} failed: {e}")),
        }
    }
    Ok(results)
}

fn parse_single_embedding(resp: &serde_json::Value) -> Result<Vec<f32>, String> {
    // OpenAI-compatible: { "data": [{ "embedding": [...] }] }
    if let Some(arr) = resp["data"][0]["embedding"].as_array() {
        return Ok(arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect());
    }
    // Legacy: { "embedding": [...] }
    if let Some(arr) = resp["embedding"].as_array() {
        return Ok(arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect());
    }
    let s = resp.to_string();
    Err(format!("no embedding in response: {}", &s[..s.len().min(300)]))
}

// ── RAG: In-memory vector store with HNSW index ────────────

#[derive(Clone)]
struct VecChunk {
    text: String,
    source: String,
    vector: Vec<f32>,
    kind: String,    // "block", "block_part", "file_summary", "gap", "text"
    file: String,    // originating filename, e.g. "main.rs"
    domain: String,  // retrieval corpus: "code" | "text"
}

/// Cosine distance: 1.0 − cosine_similarity.  Lower = more similar.
/// HNSW uses "lower is closer" convention throughout.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() { return 1.0; }
    let mut dot = 0.0f32;
    let mut mag_a = 0.0f32;
    let mut mag_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        mag_a += a[i] * a[i];
        mag_b += b[i] * b[i];
    }
    let denom = mag_a.sqrt() * mag_b.sqrt();
    if denom < 1e-10 { return 1.0; }
    1.0 - (dot / denom)
}

/// Convert cosine distance back to similarity for the public API.
#[inline]
fn distance_to_similarity(d: f32) -> f32 { 1.0 - d }

/// Cosine similarity between two vectors.  Returns 0..1 (1 = identical).
#[inline]
fn cosine_similarity_vecs(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_distance(a, b)
}

// ── HNSW Graph ──────────────────────────────────────────────
//
// Hierarchical Navigable Small World graph for approximate nearest
// neighbor search.  O(log n) query time with high recall.
//
// References:
//   Malkov & Yashunin, "Efficient and robust approximate nearest
//   neighbor search using Hierarchical Navigable Small World graphs"
//   (2018), arXiv:1603.09320v4.

/// Per-node metadata: the layers it lives on and its neighbor lists.
struct HnswNode {
    /// Maximum layer this node exists on (0-indexed, layer 0 is bottom).
    level: usize,
    /// Neighbors per layer: neighbors[layer] = vec of node indices.
    /// Layer 0 allows up to M0 = 2*M connections; higher layers allow M.
    neighbors: Vec<Vec<usize>>,
}

struct HnswGraph {
    nodes: Vec<HnswNode>,
    entry_point: Option<usize>,
    max_level: usize,
    m: usize,               // max connections per layer (layer 0 gets 2*m)
    m0: usize,               // = 2 * m
    ef_construction: usize,
    ml: f64,                 // level multiplier = 1 / ln(m)
    rng_state: u64,          // xorshift64 state
}

/// (distance, node_id) — ordered by distance ascending for min-extraction.
#[derive(Clone, Copy, PartialEq)]
struct DistNode {
    dist: f32,
    id: usize,
}

impl Eq for DistNode {}
impl PartialOrd for DistNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}
impl Ord for DistNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Natural order: lower dist = "less" → min-heap extracts closest first.
        self.dist.partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl HnswGraph {
    fn new(m: usize, ef_construction: usize) -> Self {
        let m = m.max(4);
        Self {
            nodes: Vec::new(),
            entry_point: None,
            max_level: 0,
            m,
            m0: m * 2,
            ef_construction,
            ml: 1.0 / (m as f64).ln(),
            rng_state: 0xDEAD_BEEF_CAFE_1337,
        }
    }

    /// Xorshift64 PRNG — fast, no deps, good enough for layer assignment.
    fn rand_f64(&mut self) -> f64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        (x as f64) / (u64::MAX as f64)
    }

    fn random_level(&mut self) -> usize {
        let r = self.rand_f64().max(1e-15);
        (-r.ln() * self.ml).floor() as usize
    }

    fn max_neighbors(&self, layer: usize) -> usize {
        if layer == 0 { self.m0 } else { self.m }
    }

    /// Core HNSW layer search: beam search starting from `entry_points`,
    /// returning the `ef` closest nodes to `query` on `layer`.
    ///
    /// Generic over V so callers can pass &[Vec<f32>] (build) or &[&[f32]] (search)
    /// without cloning vector data.
    ///
    /// Returns a max-heap (furthest on top) of up to `ef` results.
    fn search_layer<V: AsRef<[f32]>>(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
        vectors: &[V],
    ) -> BinaryHeap<DistNode> {
        let mut visited = HashSet::with_capacity(ef * 2);

        // candidates: min-heap (closest on top) — use Reverse
        let mut candidates: BinaryHeap<std::cmp::Reverse<DistNode>> = BinaryHeap::new();
        // results: max-heap (furthest on top) — natural order
        let mut results: BinaryHeap<DistNode> = BinaryHeap::new();

        for &ep in entry_points {
            if !visited.insert(ep) { continue; }
            let d = cosine_distance(query, vectors[ep].as_ref());
            candidates.push(std::cmp::Reverse(DistNode { dist: d, id: ep }));
            results.push(DistNode { dist: d, id: ep });
        }

        while let Some(std::cmp::Reverse(closest)) = candidates.pop() {
            // If the closest candidate is further than the furthest result, stop
            let furthest_dist = results.peek().map(|n| n.dist).unwrap_or(f32::MAX);
            if closest.dist > furthest_dist {
                break;
            }

            // Explore neighbors of this candidate on the given layer
            let node = &self.nodes[closest.id];
            if layer < node.neighbors.len() {
                for &neighbor_id in &node.neighbors[layer] {
                    if !visited.insert(neighbor_id) { continue; }
                    let d = cosine_distance(query, vectors[neighbor_id].as_ref());
                    let furthest_dist = results.peek().map(|n| n.dist).unwrap_or(f32::MAX);

                    if d < furthest_dist || results.len() < ef {
                        candidates.push(std::cmp::Reverse(DistNode { dist: d, id: neighbor_id }));
                        results.push(DistNode { dist: d, id: neighbor_id });
                        if results.len() > ef {
                            results.pop(); // evict furthest
                        }
                    }
                }
            }
        }

        results
    }

    /// Select the best M neighbors from candidates using the simple heuristic:
    /// just take the M closest.  (The "heuristic neighbor selection" from the
    /// paper is more complex but benchmarks show diminishing returns for our
    /// dimensionality range.)
    fn select_neighbors(candidates: &BinaryHeap<DistNode>, m: usize) -> Vec<usize> {
        let mut sorted: Vec<DistNode> = candidates.iter().copied().collect();
        sorted.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
        sorted.iter().take(m).map(|n| n.id).collect()
    }

    /// Prune a node's neighbor list on a given layer to at most `max_m`,
    /// keeping the closest neighbors by distance.
    fn prune(&mut self, node_id: usize, layer: usize, vectors: &[Vec<f32>]) {
        let max_m = self.max_neighbors(layer);
        let neighbors = &self.nodes[node_id].neighbors[layer];
        if neighbors.len() <= max_m { return; }

        let mut scored: Vec<(f32, usize)> = neighbors.iter()
            .map(|&nid| (cosine_distance(&vectors[node_id], &vectors[nid]), nid))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max_m);

        self.nodes[node_id].neighbors[layer] = scored.iter().map(|&(_, id)| id).collect();
    }

    /// Insert a single node into the graph.  `node_id` must already be
    /// appended to `self.nodes` (with empty neighbors) before calling.
    fn insert(&mut self, node_id: usize, vectors: &[Vec<f32>]) {
        let level = self.nodes[node_id].level;

        // First node — just set as entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(node_id);
            self.max_level = level;
            return;
        }

        let ep = self.entry_point.unwrap();

        // Phase 1: greedy descent from top layer down to level+1
        let mut current_ep = ep;
        let start_layer = self.max_level;
        if start_layer > level {
            for lc in ((level + 1)..=start_layer).rev() {
                let results = self.search_layer(
                    &vectors[node_id], &[current_ep], 1, lc, vectors,
                );
                if let Some(nearest) = results.into_iter()
                    .min_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal))
                {
                    current_ep = nearest.id;
                }
            }
        }

        // Phase 2: insert at layers min(level, max_level) down to 0
        let insert_top = level.min(self.max_level);
        let mut entry_points = vec![current_ep];

        for lc in (0..=insert_top).rev() {
            let results = self.search_layer(
                &vectors[node_id], &entry_points, self.ef_construction, lc, vectors,
            );
            let max_m = self.max_neighbors(lc);
            let selected = Self::select_neighbors(&results, max_m);

            // Connect node_id → selected
            self.nodes[node_id].neighbors[lc] = selected.clone();

            // Bidirectional: connect each selected → node_id, then prune
            for &sid in &selected {
                self.nodes[sid].neighbors[lc].push(node_id);
                if self.nodes[sid].neighbors[lc].len() > max_m {
                    self.prune(sid, lc, vectors);
                }
            }

            // Entry points for next layer down = the search results
            entry_points = results.into_iter().map(|n| n.id).collect();
        }

        // Promote entry point if this node is on a higher level
        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(node_id);
        }
    }

    /// K-nearest-neighbor search.  Returns up to `k` results sorted by
    /// ascending distance (most similar first).
    fn knn_search<V: AsRef<[f32]>>(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        vectors: &[V],
    ) -> Vec<DistNode> {
        let Some(ep) = self.entry_point else { return Vec::new(); };

        // Phase 1: greedy descent from top to layer 1
        let mut current_ep = ep;
        if self.max_level > 0 {
            for lc in (1..=self.max_level).rev() {
                let results = self.search_layer(query, &[current_ep], 1, lc, vectors);
                if let Some(nearest) = results.into_iter()
                    .min_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal))
                {
                    current_ep = nearest.id;
                }
            }
        }

        // Phase 2: full beam search on layer 0
        let ef = ef_search.max(k);
        let results = self.search_layer(query, &[current_ep], ef, 0, vectors);

        // Extract top-k sorted by distance
        let mut sorted: Vec<DistNode> = results.into_iter().collect();
        sorted.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(k);
        sorted
    }

    /// Build the entire graph from scratch for all vectors.
    fn build_all(&mut self, vectors: &[Vec<f32>]) {
        let n = vectors.len();
        self.nodes.clear();
        self.nodes.reserve(n);
        self.entry_point = None;
        self.max_level = 0;

        // Pre-assign levels and create empty nodes
        for _ in 0..n {
            let level = self.random_level();
            let neighbors = (0..=level).map(|_| Vec::new()).collect();
            self.nodes.push(HnswNode { level, neighbors });
        }

        // Insert nodes one by one
        for i in 0..n {
            self.insert(i, vectors);
        }

        eprintln!(
            "[hnsw] built graph: {} nodes, max_level={}, M={}, ef_c={}",
            n, self.max_level, self.m, self.ef_construction
        );
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.entry_point = None;
        self.max_level = 0;
    }

    fn is_empty(&self) -> bool { self.nodes.is_empty() }

    fn len(&self) -> usize { self.nodes.len() }

    /// Total edges in the graph (for diagnostics).
    fn edge_count(&self) -> usize {
        self.nodes.iter()
            .flat_map(|n| n.neighbors.iter())
            .map(|nbrs| nbrs.len())
            .sum()
    }

    // ── Serialization ───────────────────────────────────────

    /// Serialize graph to binary:
    ///   [u32 node_count] [u32 max_level] [u32 entry_point (u32::MAX if none)] [u32 m]
    ///   for each node:
    ///     [u32 level]
    ///     for layer in 0..=level:
    ///       [u32 neighbor_count] [u32 * neighbor_count]
    fn save_to(&self, buf: &mut Vec<u8>) {
        let count = self.nodes.len() as u32;
        let ep = self.entry_point.map(|e| e as u32).unwrap_or(u32::MAX);
        buf.extend_from_slice(&count.to_le_bytes());
        buf.extend_from_slice(&(self.max_level as u32).to_le_bytes());
        buf.extend_from_slice(&ep.to_le_bytes());
        buf.extend_from_slice(&(self.m as u32).to_le_bytes());

        for node in &self.nodes {
            buf.extend_from_slice(&(node.level as u32).to_le_bytes());
            for layer_neighbors in &node.neighbors {
                buf.extend_from_slice(&(layer_neighbors.len() as u32).to_le_bytes());
                for &nid in layer_neighbors {
                    buf.extend_from_slice(&(nid as u32).to_le_bytes());
                }
            }
        }
    }

    /// Deserialize graph from binary.  Returns bytes consumed.
    fn load_from(&mut self, data: &[u8]) -> Result<usize, String> {
        if data.len() < 16 { return Err("hnsw header too short".into()); }
        let count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let max_level = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        let ep_raw = u32::from_le_bytes(data[8..12].try_into().unwrap());
        let m = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;

        self.max_level = max_level;
        self.entry_point = if ep_raw == u32::MAX { None } else { Some(ep_raw as usize) };
        self.m = m;
        self.m0 = m * 2;
        self.ml = 1.0 / (m as f64).ln();

        let mut pos = 16;
        self.nodes.clear();
        self.nodes.reserve(count);

        for i in 0..count {
            if pos + 4 > data.len() { return Err(format!("hnsw truncated at node {i} (level)")); }
            let level = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
            pos += 4;

            let mut neighbors = Vec::with_capacity(level + 1);
            for lc in 0..=level {
                if pos + 4 > data.len() { return Err(format!("hnsw truncated at node {i} layer {lc}")); }
                let nn = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
                pos += 4;
                let bytes_needed = nn * 4;
                if pos + bytes_needed > data.len() {
                    return Err(format!("hnsw truncated at node {i} layer {lc} neighbors"));
                }
                let nbrs: Vec<usize> = (0..nn)
                    .map(|j| u32::from_le_bytes(data[pos + j*4..pos + j*4 + 4].try_into().unwrap()) as usize)
                    .collect();
                pos += bytes_needed;
                neighbors.push(nbrs);
            }
            self.nodes.push(HnswNode { level, neighbors });
        }

        eprintln!(
            "[hnsw] loaded graph: {} nodes, max_level={}, M={}, edges={}",
            self.nodes.len(), self.max_level, self.m, self.edge_count()
        );
        Ok(pos)
    }
}

// ── RAG Store (chunks + HNSW) ───────────────────────────────

struct RagStore {
    chunks: Vec<VecChunk>,
    graph: HnswGraph,
    cfg: RagCfg,
    indexed_files: Vec<String>,
}

impl RagStore {
    fn new(cfg: RagCfg) -> Self {
        let graph = HnswGraph::new(cfg.hnsw_m, cfg.hnsw_ef_construction);
        let mut store = Self { chunks: Vec::new(), graph, cfg, indexed_files: Vec::new() };
        if let Err(e) = store.load() {
            eprintln!("[rag] load: {e} (starting empty)");
        }
        store
    }

    /// Store pre-computed embeddings and build the HNSW graph.
    fn store_embeddings(
        &mut self,
        chunks: Vec<String>,
        sources: Vec<String>,
        vectors: Vec<Vec<f32>>,
        file_names: Vec<String>,
        kinds: Vec<String>,
        files: Vec<String>,
        domain: &str,
    ) -> Result<usize, String> {
        if vectors.is_empty() { return Err("no embeddings".into()); }
        let dim = vectors[0].len();
        if dim == 0 { return Err("embedding dimension is 0".into()); }
        if let Some(existing) = self.vector_dim() {
            if existing != dim {
                return Err(format!(
                    "embedding dim {dim} != indexed dim {existing} — clear the index before switching embed models"
                ));
            }
        }

        // Domain-scoped replace: drop only this domain's chunks, keep the
        // other corpus intact, then append the freshly embedded ones. The
        // HNSW graph is rebuilt over the union so node ids stay aligned.
        self.chunks.retain(|c| c.domain != domain);
        self.chunks.reserve(chunks.len());
        let added = chunks.len();
        let iter = chunks.into_iter()
            .zip(sources)
            .zip(vectors)
            .zip(kinds)
            .zip(files);
        for ((((text, source), vector), kind), file) in iter {
            self.chunks.push(VecChunk { text, source, vector, kind, file, domain: domain.to_string() });
        }

        // Merge indexed_files (union across domains).
        for f in file_names {
            if !self.indexed_files.contains(&f) { self.indexed_files.push(f); }
        }

        let t0 = Instant::now();
        self.rebuild_graph();
        eprintln!("[rag] HNSW built in {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

        if let Err(e) = self.save() {
            eprintln!("[rag] save warning: {e}");
        }
        eprintln!("[rag] indexed {added} '{domain}' chunks (dim={dim}, total {} chunks, persisted to {})",
            self.chunks.len(), self.cfg.db_path);
        Ok(added)
    }

    /// Code-aware BM25 keyword score.
    /// Splits on whitespace AND code punctuation, uses a code-specific
    /// stopword list, and accepts short tokens (code has 1-2 char identifiers).
    fn bm25_score(&self, query: &str, chunk_idx: usize) -> f32 {
        const STOP: &[&str] = &[
            // English
            "the", "and", "for", "this", "that", "with", "from", "are", "was",
            "not", "but", "can", "will", "has", "had", "its", "all", "any",
            // Rust keywords too common to discriminate
            "let", "mut", "pub", "use", "mod", "ref", "str", "self",
            "true", "false", "crate", "super", "where",
            // Common types
            "i32", "u32", "i64", "u64", "f32", "f64", "usize", "bool",
            "fn", "impl", "struct", "enum", "type", "const",
            // Common in all languages
            "var", "val", "new", "return", "if", "else", "while",
            "for", "in", "to", "of", "is", "it", "be", "as", "do",
        ];

        let query_terms: Vec<String> = query
            .split(|c: char| c.is_whitespace() || c == ':' || c == '.' || c == '(' || c == ')')
            .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric() && c != '_').to_lowercase())
            .filter(|t| !t.is_empty() && !STOP.contains(&t.as_str()))
            .collect();
        if query_terms.is_empty() { return 0.0; }

        let text = self.chunks[chunk_idx].text.to_lowercase();
        let doc_len = text.split_whitespace().count() as f32;
        let avg_len = if self.chunks.is_empty() { doc_len } else {
            self.chunks.iter()
                .map(|c| c.text.split_whitespace().count())
                .sum::<usize>() as f32 / self.chunks.len() as f32
        };

        let k1: f32 = 1.2;
        let b: f32 = 0.75;
        let n = self.chunks.len() as f32;
        let mut score: f32 = 0.0;

        for term in &query_terms {
            let tf = text.matches(term.as_str()).count() as f32;
            if tf == 0.0 { continue; }
            let df = self.chunks.iter()
                .filter(|c| c.text.to_lowercase().contains(term.as_str()))
                .count() as f32;
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            score += idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * doc_len / avg_len));
        }
        score
    }

    /// Search using HNSW graph with hybrid BM25+vector scoring,
    /// MMR diversity reranking, and hierarchical chunk expansion.
    fn search_local(
        &self,
        query_vec: &[f32],
        limit: usize,
        query_hint: &str,
        domain: &str,
    ) -> Vec<(String, String, f32)> {
        if self.chunks.is_empty() { return Vec::new(); }

        // ── Stage 1: Retrieve candidates for hybrid re-ranking + MMR ──
        // Widen retrieval when domain-scoped so the target corpus isn't
        // starved by nearer neighbours from the other domain.
        let candidate_limit = limit * 6;
        let use_hybrid = self.cfg.hybrid_weight_bm25 > 0.0 && !query_hint.is_empty();

        let raw: Vec<(usize, f32)> = if !self.graph.is_empty()
            && self.graph.len() == self.chunks.len()
        {
            // HNSW search path — O(log n)
            let vec_refs: Vec<&[f32]> = self.chunks.iter()
                .map(|c| c.vector.as_slice()).collect();
            let results = self.graph.knn_search(
                query_vec, candidate_limit, self.cfg.hnsw_ef_search, &vec_refs,
            );
            results.iter()
                .map(|dn| (dn.id, distance_to_similarity(dn.dist)))
                .collect()
        } else {
            // Fallback: brute-force
            eprintln!("[rag] WARN: HNSW graph missing/stale, falling back to brute force");
            let mut scored: Vec<(usize, f32)> = self.chunks.iter().enumerate()
                .map(|(i, c)| (i, cosine_distance(query_vec, &c.vector)))
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.iter()
                .take(candidate_limit)
                .map(|(i, d)| (*i, distance_to_similarity(*d)))
                .collect()
        };

        // Scope to the requested retrieval domain.
        let candidates: Vec<(usize, f32)> = raw.into_iter()
            .filter(|(id, _)| self.chunks[*id].domain == domain)
            .take(limit * 3)
            .collect();
        if candidates.is_empty() { return Vec::new(); }

        // ── Stage 2: Hybrid BM25 re-scoring ──────────────────────────────
        let mut hybrid_scored: Vec<(usize, f32)> = if use_hybrid {
            let wv = self.cfg.hybrid_weight_vector;
            let wb = self.cfg.hybrid_weight_bm25;
            candidates.iter().map(|(id, vec_sim)| {
                let bm25_raw = self.bm25_score(query_hint, *id);
                let bm25_norm = bm25_raw / (bm25_raw + 1.0);
                let combined = wv * vec_sim + wb * bm25_norm;
                (*id, combined)
            }).collect()
        } else {
            candidates
        };

        hybrid_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // ── Stage 3: MMR diversity reranking ─────────────────────────────
        // score_mmr = λ * relevance − (1−λ) * max_sim_to_already_selected
        let lambda: f32 = 0.7;
        let min_sim = self.cfg.min_similarity;

        let mut selected: Vec<(usize, f32)> = Vec::with_capacity(limit);
        let mut remaining: Vec<(usize, f32)> = hybrid_scored
            .into_iter()
            .filter(|(_, sim)| *sim >= min_sim)
            .collect();

        // First pick: highest relevance
        if let Some(first) = remaining.first().copied() {
            selected.push(first);
            remaining.remove(0);
        }

        // Subsequent picks: balance relevance vs. diversity
        while selected.len() < limit && !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_mmr = f32::MIN;

            for (i, (chunk_id, relevance)) in remaining.iter().enumerate() {
                let max_sim_to_sel = selected.iter()
                    .map(|(sel_id, _)| {
                        cosine_similarity_vecs(
                            &self.chunks[*chunk_id].vector,
                            &self.chunks[*sel_id].vector,
                        )
                    })
                    .fold(f32::MIN, f32::max);

                let mmr = lambda * relevance - (1.0 - lambda) * max_sim_to_sel;
                if mmr > best_mmr {
                    best_mmr = mmr;
                    best_idx = i;
                }
            }

            selected.push(remaining.remove(best_idx));
        }

        // ── Stage 4: Hierarchical expansion ──────────────────────────────
        // If a file_summary chunk was selected, replace it with top child
        // chunks from the same file (they carry the actual code).
        let mut final_results: Vec<(String, String, f32)> = Vec::new();

        for (id, sim) in &selected {
            let chunk = &self.chunks[*id];

            if chunk.kind == "file_summary" && !chunk.file.is_empty() {
                let already: HashSet<usize> =
                    selected.iter().map(|(sid, _)| *sid).collect();

                let mut file_chunks: Vec<(usize, f32)> = self.chunks.iter()
                    .enumerate()
                    .filter(|(i, c)| {
                        c.file == chunk.file
                            && c.kind != "file_summary"
                            && !already.contains(i)
                    })
                    .map(|(i, c)| {
                        let vsim = distance_to_similarity(
                            cosine_distance(query_vec, &c.vector),
                        );
                        (i, vsim)
                    })
                    .collect();

                file_chunks.sort_by(|a, b| b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal));

                let expand_count = 3.min(limit.saturating_sub(final_results.len()));
                for (cid, csim) in file_chunks.into_iter().take(expand_count) {
                    if csim >= min_sim {
                        let cc = &self.chunks[cid];
                        final_results.push((cc.source.clone(), cc.text.clone(), csim));
                    }
                }
            } else {
                final_results.push((chunk.source.clone(), chunk.text.clone(), *sim));
            }
        }

        final_results.truncate(limit);
        final_results
    }

    fn clear(&mut self) -> Result<(), String> {
        self.chunks.clear();
        self.graph.clear();
        self.indexed_files.clear();
        let _ = fs::remove_file(&self.cfg.db_path);
        eprintln!("[rag] index cleared");
        Ok(())
    }

    /// Clear only one retrieval domain, preserving the other corpus.
    fn clear_domain(&mut self, domain: &str) -> Result<(), String> {
        let before = self.chunks.len();
        self.chunks.retain(|c| c.domain != domain);
        let removed = before - self.chunks.len();
        if self.chunks.is_empty() {
            return self.clear();
        }
        self.indexed_files.retain(|f| self.chunks.iter().any(|c| &c.file == f));
        self.rebuild_graph();
        if let Err(e) = self.save() { eprintln!("[rag] save warning: {e}"); }
        eprintln!("[rag] cleared {removed} '{domain}' chunks ({} remain)", self.chunks.len());
        Ok(())
    }

    /// Chunk count for a single domain.
    fn domain_count(&self, domain: &str) -> usize {
        self.chunks.iter().filter(|c| c.domain == domain).count()
    }

    /// Persist index in binary format (v3):
    ///   [u8 x 4  "RAG3" magic]
    ///   [u32 chunk_count] [u32 vector_dim]
    ///   for each chunk:
    ///     [u32 source_len] [source_bytes]
    ///     [u32 text_len] [text_bytes]
    ///     [u32 kind_len] [kind_bytes]
    ///     [u32 file_len] [file_bytes]
    ///     [u32 domain_len] [domain_bytes]
    ///     [f32 * dim]
    ///   [HNSW graph bytes]
    fn save(&self) -> Result<(), String> {
        if let Some(parent) = Path::new(&self.cfg.db_path).parent() {
            let _ = fs::create_dir_all(parent);
        }
        let dim = self.chunks.first().map(|c| c.vector.len()).unwrap_or(0) as u32;
        let count = self.chunks.len() as u32;

        let mut buf = Vec::with_capacity(12 + self.chunks.len() * (20 + 200 + dim as usize * 4));
        buf.extend_from_slice(b"RAG3");
        buf.extend_from_slice(&count.to_le_bytes());
        buf.extend_from_slice(&dim.to_le_bytes());

        let put = |buf: &mut Vec<u8>, s: &[u8]| {
            buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
            buf.extend_from_slice(s);
        };
        for c in &self.chunks {
            put(&mut buf, c.source.as_bytes());
            put(&mut buf, c.text.as_bytes());
            put(&mut buf, c.kind.as_bytes());
            put(&mut buf, c.file.as_bytes());
            put(&mut buf, c.domain.as_bytes());
            for &v in &c.vector {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }

        self.graph.save_to(&mut buf);

        fs::write(&self.cfg.db_path, &buf)
            .map_err(|e| format!("write {}: {e}", self.cfg.db_path))
    }

    /// Load persisted index from disk (v3 format only).
    fn load(&mut self) -> Result<(), String> {
        let path = Path::new(&self.cfg.db_path);
        if !path.exists() { return Ok(()); }
        let data = fs::read(path)
            .map_err(|e| format!("read {}: {e}", self.cfg.db_path))?;
        if data.len() < 12 { return Ok(()); }

        // Require RAG3 magic — older formats are deleted, not migrated.
        if &data[0..4] != b"RAG3" {
            eprintln!("[rag] stale index at {} — delete and re-index", self.cfg.db_path);
            return Ok(());
        }

        let count = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let mut pos = 12;

        self.chunks.clear();
        self.chunks.reserve(count);

        // Length-prefixed string reader.
        let read_str = |data: &[u8], pos: &mut usize, i: usize, what: &str| -> Result<String, String> {
            if *pos + 4 > data.len() { return Err(format!("truncated at chunk {i} ({what} len)")); }
            let n = u32::from_le_bytes(data[*pos..*pos+4].try_into().unwrap()) as usize;
            *pos += 4;
            if *pos + n > data.len() { return Err(format!("truncated at chunk {i} ({what})")); }
            let s = String::from_utf8_lossy(&data[*pos..*pos+n]).to_string();
            *pos += n;
            Ok(s)
        };

        for i in 0..count {
            let source = read_str(&data, &mut pos, i, "source")?;
            let text   = read_str(&data, &mut pos, i, "text")?;
            let kind   = read_str(&data, &mut pos, i, "kind")?;
            let file   = read_str(&data, &mut pos, i, "file")?;
            let domain = read_str(&data, &mut pos, i, "domain")?;

            let vec_bytes = dim * 4;
            if pos + vec_bytes > data.len() { return Err(format!("truncated at chunk {i} (vector)")); }
            let vector: Vec<f32> = (0..dim)
                .map(|j| {
                    let o = pos + j * 4;
                    f32::from_le_bytes(data[o..o+4].try_into().unwrap())
                })
                .collect();
            pos += vec_bytes;

            self.chunks.push(VecChunk { text, source, vector, kind, file, domain });
        }

        // Load HNSW graph
        if pos < data.len() {
            self.graph.load_from(&data[pos..])?;
            if self.graph.len() != self.chunks.len() {
                eprintln!("[rag] graph/chunk count mismatch, rebuilding");
                self.rebuild_graph();
            }
        } else {
            self.rebuild_graph();
        }

        // Reconstruct indexed_files from the file field (domain-agnostic).
        let mut files: Vec<String> = self.chunks.iter()
            .map(|c| if c.file.is_empty() {
                c.source.split(':').next().unwrap_or("").to_string()
            } else {
                c.file.clone()
            })
            .collect();
        files.sort();
        files.dedup();
        self.indexed_files = files;
        eprintln!("[rag] loaded {} chunks from {}", self.chunks.len(), self.cfg.db_path);
        Ok(())
    }

    /// Rebuild the HNSW graph from the current chunk vectors.
    fn rebuild_graph(&mut self) {
        if self.chunks.is_empty() {
            self.graph.clear();
            return;
        }
        let t0 = Instant::now();
        let vecs: Vec<Vec<f32>> = self.chunks.iter().map(|c| c.vector.clone()).collect();
        self.graph = HnswGraph::new(self.cfg.hnsw_m, self.cfg.hnsw_ef_construction);
        self.graph.build_all(&vecs);
        eprintln!("[rag] graph rebuilt in {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    fn vector_dim(&self) -> Option<usize> {
        self.chunks.first().map(|c| c.vector.len())
    }

    fn status_json(&self) -> serde_json::Value {
        serde_json::json!({
            "enabled": self.cfg.enabled,
            "chunks": self.chunks.len(),
            "chunks_code": self.domain_count("code"),
            "chunks_text": self.domain_count("text"),
            "vector_dim": self.vector_dim(),
            "files": self.indexed_files,
            "db_path": self.cfg.db_path,
            "min_similarity": self.cfg.min_similarity,
            "hybrid": {
                "vector_weight": self.cfg.hybrid_weight_vector,
                "bm25_weight": self.cfg.hybrid_weight_bm25,
            },
            "chunker_tool": self.cfg.chunker_tool,
            "hnsw": {
                "nodes": self.graph.len(),
                "max_level": self.graph.max_level,
                "edges": self.graph.edge_count(),
                "m": self.graph.m,
                "ef_search": self.cfg.hnsw_ef_search,
            },
        })
    }
}

// ── Managed llama-server process (generation + embedding) ───
//
// One process-manager type for both roles. `kind` selects the log
// prefix ("llama" | "embed"); role-specific launch flags are built by
// the free `llama_args` / `embed_args` functions. The old split
// EmbedServer / LlamaServer types (near-identical wait/stop/poll logic)
// are deleted — this is the single production implementation.

#[derive(Clone, Debug, PartialEq)]
enum ServerStatus { Stopped, Starting, Ready, Error(String) }

/// Result of a single non-blocking readiness check.
enum PollOutcome { Pending, Ready, Dead(String) }

struct ManagedServer {
    kind: &'static str,
    child: Option<Child>,
    status: ServerStatus,
    model: String,
    pid: Option<u32>,
    port: u16,
}

impl ManagedServer {
    fn new(kind: &'static str, port: u16) -> Self {
        Self { kind, child: None, status: ServerStatus::Stopped, model: String::new(), pid: None, port }
    }

    /// Spawn `binary args`, replacing any existing child. stderr is
    /// inherited (not piped-and-unread — an unread pipe can deadlock the
    /// child once its buffer fills).
    fn spawn(&mut self, binary: &str, args: &[String], model: &str, port: u16) -> Result<(), String> {
        self.stop();
        let child = Command::new(binary)
            .args(args)
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("spawn {}: {e}", self.kind))?;
        self.pid = Some(child.id());
        self.child = Some(child);
        self.status = ServerStatus::Starting;
        self.model = model.to_string();
        self.port = port;
        Ok(())
    }

    /// One non-blocking readiness probe: reap the child if it died,
    /// otherwise health-check the port. Does not sleep or mutate status.
    fn poll_once(&mut self) -> PollOutcome {
        match self.child.as_mut().map(|c| c.try_wait()) {
            None => return PollOutcome::Dead("process gone".into()),
            Some(Ok(Some(code))) => { self.child = None; return PollOutcome::Dead(format!("exited: {code}")); }
            Some(Err(e)) => return PollOutcome::Dead(format!("wait: {e}")),
            Some(Ok(None)) => {}
        }
        if check_health(self.port) { PollOutcome::Ready } else { PollOutcome::Pending }
    }

    /// Block up to `timeout_secs` for readiness. Used on the boot path
    /// where blocking is acceptable. Sets `status` to Ready/Error.
    fn wait_ready(&mut self, timeout_secs: u64) -> bool {
        let deadline = Instant::now() + Duration::from_secs(timeout_secs);
        std::thread::sleep(Duration::from_millis(500));
        while Instant::now() < deadline {
            match self.poll_once() {
                PollOutcome::Ready => {
                    self.status = ServerStatus::Ready;
                    eprintln!("[{}] ready (pid {:?})", self.kind, self.pid);
                    return true;
                }
                PollOutcome::Dead(e) => { self.status = ServerStatus::Error(e); return false; }
                PollOutcome::Pending => std::thread::sleep(Duration::from_millis(700)),
            }
        }
        self.status = ServerStatus::Error(format!("timeout ({timeout_secs}s)"));
        false
    }

    fn stop(&mut self) {
        if let Some(mut c) = self.child.take() {
            eprintln!("[{}] killing pid {:?}", self.kind, self.pid);
            let _ = c.kill();
            let _ = c.wait();
        }
        self.status = ServerStatus::Stopped;
        self.model.clear();
        self.pid = None;
    }

    fn is_ready(&self) -> bool { self.status == ServerStatus::Ready }

    /// True while a start is in flight or already serving — used to avoid
    /// a duplicate spawn when two requests race to lazy-load.
    fn is_active(&self) -> bool {
        matches!(self.status, ServerStatus::Ready | ServerStatus::Starting)
    }

    fn status_json(&self) -> serde_json::Value {
        serde_json::json!({
            "status": match &self.status {
                ServerStatus::Stopped => "stopped",
                ServerStatus::Starting => "starting",
                ServerStatus::Ready => "ready",
                ServerStatus::Error(_) => "error",
            },
            "model": self.model,
            "pid": self.pid,
            "port": self.port,
            "error": match &self.status {
                ServerStatus::Error(e) => Some(e.as_str()),
                _ => None,
            },
        })
    }
}

impl Drop for ManagedServer {
    fn drop(&mut self) { self.stop(); }
}

/// Launch flags for the main generation server.
fn llama_args(cfg: &RuntimeCfg, model: &Model) -> Vec<String> {
    let ngl = if cfg.ngl < 0 { 99 } else { cfg.ngl };
    let fa = if cfg.flash_attn { "on" } else { "auto" };
    let mut args = vec![
        "-m".into(), model.path.clone(),
        "--port".into(), cfg.llama_port.to_string(),
        "-ngl".into(), ngl.to_string(),
        "-c".into(), cfg.ctx.to_string(),
        "-np".into(), cfg.parallel_slots.to_string(),
        "--threads".into(), cfg.threads.to_string(),
        "--host".into(), "127.0.0.1".into(),
        "--flash-attn".into(), fa.into(),
    ];
    // No --embedding here: embeddings run in a dedicated ManagedServer so
    // the main model keeps maximum KV cache for generation.
    if !cfg.cache_type_k.is_empty() { args.extend(["--cache-type-k".into(), cfg.cache_type_k.clone()]); }
    if !cfg.cache_type_v.is_empty() { args.extend(["--cache-type-v".into(), cfg.cache_type_v.clone()]); }

    // Speculative decoding via llama.cpp's unified --spec-type interface.
    // Self-speculation types (draft-mtp, eagle, medusa, ...) use the model's
    // own heads — no draft checkpoint. Only draft-model type pairs a separate
    // draft model. Flags are emitted only when the config is complete, so an
    // invalid draft-model setup never reaches the server as a broken launch.
    if !cfg.spec_type.is_empty() {
        let draft_flags: Option<Vec<String>> = if cfg.spec_type == "draft-model" {
            let draft_path = format!("{}/{}", cfg.models_dir, cfg.draft_model);
            if cfg.draft_model.is_empty() {
                eprintln!("[llama]   WARNING: spec_type='draft-model' needs draft_model — speculation disabled");
                None
            } else if !Path::new(&draft_path).exists() {
                eprintln!("[llama]   WARNING: draft model '{draft_path}' not found — speculation disabled");
                None
            } else {
                let draft_ngl = if cfg.gpu_layers_draft < 0 { 99 } else { cfg.gpu_layers_draft };
                Some(vec![
                    "--model-draft".into(), draft_path,
                    "--gpu-layers-draft".into(), draft_ngl.to_string(),
                ])
            }
        } else {
            Some(Vec::new()) // self-speculation: no draft checkpoint required
        };

        if let Some(extra) = draft_flags {
            eprintln!("[llama]   spec={} (n_max={})", cfg.spec_type, cfg.spec_draft_n_max);
            args.extend(["--spec-type".into(), cfg.spec_type.clone()]);
            if cfg.spec_draft_n_max > 0 {
                args.extend(["--spec-draft-n-max".into(), cfg.spec_draft_n_max.to_string()]);
            }
            args.extend(extra);
        }
    }
    args
}

/// Launch flags for the dedicated embedding server.
fn embed_args(model_path: &str, cfg: &EmbedCfg) -> Vec<String> {
    let ngl = if cfg.gpu_layers < 0 { 99 } else { cfg.gpu_layers };
    let ctx = cfg.context_size.to_string();
    let mut args = vec![
        "-m".into(), model_path.to_string(),
        "--port".into(), cfg.port.to_string(),
        "-ngl".into(), ngl.to_string(),
        "-c".into(), ctx.clone(),
        "-ub".into(), ctx,
        "-np".into(), cfg.parallel_slots.to_string(),
        "--host".into(), "127.0.0.1".into(),
        "--embedding".into(),
    ];
    if !cfg.pooling.is_empty() { args.extend(["--pooling".into(), cfg.pooling.clone()]); }
    args
}

/// Selects which managed server a background poller operates on.
#[derive(Clone, Copy)]
enum Which { Llama, Embed }

/// Background readiness poll — updates `State` under short locks so other
/// requests aren't blocked while a server warms up. Shared by /api/load
/// and /api/embed/start.
fn spawn_ready_poll(st: &Shared, which: Which, timeout_secs: u64) {
    let bg = Arc::clone(st);
    std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(timeout_secs);
        std::thread::sleep(Duration::from_millis(500));
        loop {
            if Instant::now() >= deadline {
                bg.lock().unwrap().server_mut(which).status =
                    ServerStatus::Error(format!("timeout ({timeout_secs}s)"));
                return;
            }
            let outcome = bg.lock().unwrap().server_mut(which).poll_once();
            match outcome {
                PollOutcome::Ready => {
                    let mut s = bg.lock().unwrap();
                    let srv = s.server_mut(which);
                    srv.status = ServerStatus::Ready;
                    eprintln!("[{}] ready (pid {:?})", srv.kind, srv.pid);
                    return;
                }
                PollOutcome::Dead(e) => {
                    bg.lock().unwrap().server_mut(which).status = ServerStatus::Error(e);
                    return;
                }
                PollOutcome::Pending => std::thread::sleep(Duration::from_millis(700)),
            }
        }
    });
}

/// Lazy-start the embed server on first RAG use and block until ready.
/// Idempotent and race-safe: if another request already started it, this
/// only waits. Never holds the state lock across sleeps / health checks.
fn ensure_embed_ready(st: &Shared) -> Result<(), String> {
    // Fast path + config validation.
    {
        let s = st.lock().unwrap();
        if s.embed.is_ready() { return Ok(()); }
        if !s.cfg.embed.enabled || s.cfg.embed.model.is_empty() {
            return Err("embed server not configured — set [embed] model in config.toml".into());
        }
    }

    let (binary, model_path, model_name, embed_cfg, timeout) = {
        let s = st.lock().unwrap();
        let path = format!("{}/{}", s.cfg.models_dir, s.cfg.embed.model);
        (s.cfg.llama_binary.clone(), path, s.cfg.embed.model.clone(),
         s.cfg.embed.clone(), s.cfg.embed.startup_timeout)
    };
    if !Path::new(&model_path).exists() {
        return Err(format!("embed model '{model_path}' not found"));
    }

    // Spawn under a brief lock — but only if nobody else owns the start.
    {
        let mut s = st.lock().unwrap();
        if !s.embed.is_active() {
            let ngl = if embed_cfg.gpu_layers < 0 { 99 } else { embed_cfg.gpu_layers };
            eprintln!("[embed] lazy-starting {} (ngl={ngl}, ctx={}, port={})",
                model_name, embed_cfg.context_size, embed_cfg.port);
            let args = embed_args(&model_path, &embed_cfg);
            s.embed.spawn(&binary, &args, &model_name, embed_cfg.port)?;
        }
    }

    // Wait for readiness with short status-update locks.
    let deadline = Instant::now() + Duration::from_secs(timeout);
    std::thread::sleep(Duration::from_millis(400));
    loop {
        if Instant::now() >= deadline {
            st.lock().unwrap().embed.status = ServerStatus::Error(format!("timeout ({timeout}s)"));
            return Err(format!("embed server timeout ({timeout}s)"));
        }
        let outcome = st.lock().unwrap().embed.poll_once();
        match outcome {
            PollOutcome::Ready => {
                let mut s = st.lock().unwrap();
                s.embed.status = ServerStatus::Ready;
                eprintln!("[embed] ready (pid {:?})", s.embed.pid);
                return Ok(());
            }
            PollOutcome::Dead(e) => {
                st.lock().unwrap().embed.status = ServerStatus::Error(e.clone());
                return Err(format!("embed server failed: {e}"));
            }
            PollOutcome::Pending => std::thread::sleep(Duration::from_millis(500)),
        }
    }
}

fn curl_cmd() -> &'static str {
    if cfg!(windows) { "curl.exe" } else { "curl" }
}

// ── Direct HTTP client (replaces curl for non-streaming) ────

/// Simple HTTP GET via raw TcpStream. Returns response body or error.
fn http_get(host: &str, port: u16, path: &str, timeout_secs: u64) -> Result<String, String> {
    let addr = format!("{host}:{port}");
    let mut stream = TcpStream::connect(&addr)
        .map_err(|e| format!("connect {addr}: {e}"))?;
    stream.set_read_timeout(Some(Duration::from_secs(timeout_secs))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(5))).ok();

    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(req.as_bytes()).map_err(|e| format!("write: {e}"))?;

    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).map_err(|e| format!("read: {e}"))?;
    let raw = String::from_utf8_lossy(&buf);
    extract_http_body(&raw)
}

/// Simple HTTP POST with JSON body via raw TcpStream. Returns response body.
fn http_post_json(host: &str, port: u16, path: &str, body: &str, timeout_secs: u64) -> Result<String, String> {
    let addr = format!("{host}:{port}");
    let mut stream = TcpStream::connect(&addr)
        .map_err(|e| format!("connect {addr}: {e}"))?;
    stream.set_read_timeout(Some(Duration::from_secs(timeout_secs))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(10))).ok();

    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(req.as_bytes()).map_err(|e| format!("write: {e}"))?;

    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).map_err(|e| format!("read: {e}"))?;
    let raw = String::from_utf8_lossy(&buf);
    extract_http_body(&raw)
}

/// Extract body from raw HTTP response, handling both Content-Length and chunked TE.
fn extract_http_body(raw: &str) -> Result<String, String> {
    let Some(split) = raw.find("\r\n\r\n") else {
        return Err("malformed HTTP response (no header boundary)".into());
    };
    let headers = &raw[..split].to_lowercase();
    let body = &raw[split + 4..];

    if headers.contains("transfer-encoding: chunked") {
        // Decode chunked transfer encoding
        let mut decoded = String::new();
        let mut pos = 0;
        let bytes = body.as_bytes();
        loop {
            // Skip whitespace / newlines between chunks
            while pos < bytes.len() && (bytes[pos] == b'\r' || bytes[pos] == b'\n') {
                pos += 1;
            }
            if pos >= bytes.len() { break; }
            // Read chunk size (hex)
            let size_end = body[pos..].find("\r\n").unwrap_or(body.len() - pos);
            let size_str = &body[pos..pos + size_end];
            let chunk_size = usize::from_str_radix(size_str.trim(), 16).unwrap_or(0);
            if chunk_size == 0 { break; }
            pos += size_end + 2; // skip size line + \r\n
            if pos + chunk_size <= body.len() {
                decoded.push_str(&body[pos..pos + chunk_size]);
            }
            pos += chunk_size;
        }
        Ok(decoded)
    } else {
        Ok(body.to_string())
    }
}

/// Health-check a local server: returns true if /health responds with "ok" or "status".
fn check_health(port: u16) -> bool {
    match http_get("127.0.0.1", port, "/health", 3) {
        Ok(body) => body.contains("ok") || body.contains("\"status\""),
        Err(_) => false,
    }
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
    let in_models = format!("{models_dir}/llama-server");
    if Path::new(&in_models).exists() {
        return Path::new(&in_models).canonicalize()
            .map(|c| c.to_string_lossy().into()).unwrap_or(in_models);
    }
    if let Ok(o) = Command::new("which").arg("llama-server").output() {
        if o.status.success() {
            let p = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if !p.is_empty() { return p; }
        }
    }
    "llama-server".into()
}

// ── System info probe ───────────────────────────────────────
//
// Read the host once at boot so context window + thread count are set
// from real hardware instead of blind constants. VRAM is the governing
// constraint for a GPU-offloaded model, so it drives the context ceiling
// used when no explicit ctx is configured.

struct GpuInfo { name: String, total_mib: u64, free_mib: u64 }

struct SystemInfo {
    cpu_threads: usize,
    ram_total_mib: u64,
    ram_free_mib: u64,
    gpus: Vec<GpuInfo>,
}

impl SystemInfo {
    fn probe() -> Self {
        let cpu_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let (ram_total_mib, ram_free_mib) = probe_ram();
        Self { cpu_threads, ram_total_mib, ram_free_mib, gpus: probe_gpus() }
    }

    /// Largest free-VRAM pool across detected GPUs (None => CPU-only host).
    fn free_vram_mib(&self) -> Option<u64> { self.gpus.iter().map(|g| g.free_mib).max() }

    /// Generation thread count: half the logical CPUs (accounts for SMT),
    /// clamped so we neither starve the HTTP server nor oversubscribe.
    fn gen_threads(&self) -> usize { (self.cpu_threads / 2).clamp(4, 16) }

    fn print(&self) {
        eprintln!("  system: {} logical CPUs, RAM {} / {} MiB free",
            self.cpu_threads, self.ram_free_mib, self.ram_total_mib);
        if self.gpus.is_empty() {
            eprintln!("  gpu: none detected (nvidia-smi unavailable) — CPU inference");
        } else {
            for g in &self.gpus {
                eprintln!("  gpu: {} — {} / {} MiB free", g.name, g.free_mib, g.total_mib);
            }
        }
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "cpu_threads": self.cpu_threads,
            "gen_threads": self.gen_threads(),
            "ram_total_mib": self.ram_total_mib,
            "ram_free_mib": self.ram_free_mib,
            "gpus": self.gpus.iter().map(|g| serde_json::json!({
                "name": g.name, "total_mib": g.total_mib, "free_mib": g.free_mib,
            })).collect::<Vec<_>>(),
        })
    }
}

/// Context ceiling derived from free VRAM. Heuristic, tuned for a 4-7B
/// Q4/Q5 model with quantized (q8_0) KV cache. Only applied when no
/// explicit ctx is configured — an explicit per-model ctx is always
/// honored, never silently clamped.
fn vram_ctx_ceiling(free_vram_mib: Option<u64>) -> u32 {
    match free_vram_mib {
        Some(v) if v >= 20000 => 65536,
        Some(v) if v >= 12000 => 32768,
        Some(v) if v >=  8000 => 16384,
        Some(v) if v >=  5000 => 8192,
        Some(v) if v >=  3000 => 4096,
        Some(_)               => 2048,
        None                  => 8192, // CPU / unknown host: modest default
    }
}

fn probe_gpus() -> Vec<GpuInfo> {
    let Ok(out) = Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"])
        .stderr(Stdio::null())
        .output()
    else { return Vec::new(); };
    if !out.status.success() { return Vec::new(); }
    String::from_utf8_lossy(&out.stdout).lines().filter_map(|line| {
        let mut it = line.split(',').map(|s| s.trim());
        Some(GpuInfo {
            name: it.next()?.to_string(),
            total_mib: it.next()?.parse().ok()?,
            free_mib: it.next()?.parse().ok()?,
        })
    }).collect()
}

#[cfg(target_os = "linux")]
fn probe_ram() -> (u64, u64) {
    let Ok(txt) = fs::read_to_string("/proc/meminfo") else { return (0, 0); };
    let field = |key: &str| txt.lines()
        .find_map(|l| l.strip_prefix(key))
        .and_then(|v| v.split_whitespace().next())
        .and_then(|n| n.parse::<u64>().ok())
        .unwrap_or(0);
    (field("MemTotal:") / 1024, field("MemAvailable:") / 1024)
}

#[cfg(windows)]
fn probe_ram() -> (u64, u64) {
    // Win32_OperatingSystem reports KiB.
    let Ok(out) = Command::new("powershell")
        .args([
            "-NoProfile", "-Command",
            "$m=Get-CimInstance Win32_OperatingSystem; \
             \"$($m.TotalVisibleMemorySize) $($m.FreePhysicalMemory)\"",
        ])
        .stderr(Stdio::null())
        .output()
    else { return (0, 0); };
    let s = String::from_utf8_lossy(&out.stdout);
    let mut it = s.split_whitespace().filter_map(|n| n.parse::<u64>().ok());
    (it.next().unwrap_or(0) / 1024, it.next().unwrap_or(0) / 1024)
}

#[cfg(not(any(target_os = "linux", windows)))]
fn probe_ram() -> (u64, u64) { (0, 0) }

// ── Shared state ────────────────────────────────────────────

struct State {
    cfg: RuntimeCfg,
    models: Vec<Model>,
    llama: ManagedServer,
    embed: ManagedServer,
    rag: RagStore,
    sys_info: serde_json::Value,
    tokens_session: u64,
    requests: u64,
}

impl State {
    fn server_mut(&mut self, which: Which) -> &mut ManagedServer {
        match which {
            Which::Llama => &mut self.llama,
            Which::Embed => &mut self.embed,
        }
    }
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

#[derive(Deserialize, Serialize, Clone)]
struct ChatMsg {
    role: String,     // "user" | "assistant" (system is server-generated)
    content: String,
}

#[derive(Deserialize)]
struct WriteReq {
    #[serde(default)]
    description: String,
    #[serde(default = "def_lang")]
    language: String,
    #[serde(default = "def_mode")]
    mode: String,
    #[serde(default)]
    files: Vec<FileEntry>,
    #[serde(default)]
    use_rag: bool,
    // Chat mode: full client-held thread (server is stateless).
    #[serde(default)]
    messages: Vec<ChatMsg>,
}
fn def_lang() -> String { "python".into() }
fn def_mode() -> String { "write".into() }

#[derive(Deserialize)]
struct LoadReq {
    model: String,
    #[serde(default)] ngl: Option<i32>,
    #[serde(default)] ctx: Option<u32>,
    #[serde(default)] flash_attn: Option<bool>,
    #[serde(default)] temp: Option<f32>,
    #[serde(default)] top_k: Option<u32>,
    #[serde(default)] top_p: Option<f32>,
    #[serde(default)] repeat_penalty: Option<f32>,
    #[serde(default)] draft_model: Option<String>,
    #[serde(default)] spec_type: Option<String>,
    #[serde(default)] spec_draft_n_max: Option<u32>,
    #[serde(default)] gpu_layers_draft: Option<i32>,
}

#[derive(Deserialize)]
struct ParamsReq {
    #[serde(default)] temp: Option<f32>,
    #[serde(default)] top_k: Option<u32>,
    #[serde(default)] top_p: Option<f32>,
    #[serde(default)] repeat_penalty: Option<f32>,
}

#[derive(Deserialize)]
struct RagIndexReq {
    #[serde(default)]
    files: Vec<FileEntry>,
    #[serde(default = "def_domain")]
    domain: String,   // "code" | "text"
}

#[derive(Deserialize)]
struct RagSearchReq {
    query: String,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default = "def_domain")]
    domain: String,
}

#[derive(Deserialize)]
struct RagClearReq {
    #[serde(default)]
    domain: Option<String>,   // None = clear all domains
}

fn def_domain() -> String { "code".into() }

#[derive(Deserialize)]
struct EmbedPrefixReq {
    #[serde(default)] query_prefix: Option<String>,
    #[serde(default)] doc_prefix: Option<String>,
}

#[derive(Deserialize)]
struct ChatChunk {
    #[serde(default)] choices: Vec<ChunkChoice>,
}
#[derive(Deserialize)]
struct ChunkChoice {
    delta: ChunkDelta,
    #[serde(default)] finish_reason: Option<String>,
}
#[derive(Deserialize)]
struct ChunkDelta {
    #[serde(default)] content: Option<String>,
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

    // Exclude the embed model from generation model discovery
    let embed_model = &file_cfg.embed.model;
    let exclude: Vec<&str> = if embed_model.is_empty() { vec![] } else { vec![embed_model.as_str()] };
    let models = discover_models(&file_cfg.defaults.models_dir, &file_cfg.models, &file_cfg.defaults, &exclude);

    eprintln!("\n  CODEWRITER + RAG");
    eprintln!("  {} models in {}/", models.len(), file_cfg.defaults.models_dir);
    for m in &models {
        eprintln!("    {} [{}] ngl={} ctx={}", m.name, m.family, m.gpu_layers, m.context_size);
    }
    eprintln!("  llama-server: {llama_binary}");
    if file_cfg.embed.enabled && !file_cfg.embed.model.is_empty() {
        eprintln!("  embed-server: {} (port={}, ngl={}, ctx={})",
            file_cfg.embed.model, file_cfg.embed.port,
            file_cfg.embed.gpu_layers, file_cfg.embed.context_size);
    } else {
        eprintln!("  embed-server: disabled");
    }
    if file_cfg.rag.enabled {
        eprintln!("  rag: enabled (db={}, chunk={}/{}, hnsw M={} ef_c={} ef_s={})",
            file_cfg.rag.db_path, file_cfg.rag.chunk_size, file_cfg.rag.chunk_overlap,
            file_cfg.rag.hnsw_m, file_cfg.rag.hnsw_ef_construction, file_cfg.rag.hnsw_ef_search);
        eprintln!("       min_sim={:.2}, hybrid vec={:.1}/bm25={:.1}, chunker={}",
            file_cfg.rag.min_similarity,
            file_cfg.rag.hybrid_weight_vector, file_cfg.rag.hybrid_weight_bm25,
            if file_cfg.rag.chunker_tool.is_empty() { "internal" }
            else { &file_cfg.rag.chunker_tool });
    }

    let llama_ok = Command::new(&llama_binary)
        .arg("--help").stdout(Stdio::null()).stderr(Stdio::null())
        .status().map(|s| s.success()).unwrap_or(false);
    if !llama_ok {
        eprintln!("  WARNING: '{llama_binary}' not found or not executable");
    }

    // Read the host once; drives thread count and the auto context default.
    let sys = SystemInfo::probe();
    sys.print();
    let auto_ctx = vram_ctx_ceiling(sys.free_vram_mib());
    if file_cfg.defaults.context_size == 0 {
        eprintln!("  auto ctx: {auto_ctx} (from free VRAM); threads: {}", sys.gen_threads());
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
        ctx: if file_cfg.defaults.context_size > 0 { file_cfg.defaults.context_size } else { auto_ctx },
        flash_attn: file_cfg.defaults.flash_attention,
        temp: file_cfg.defaults.temperature,
        top_k: file_cfg.defaults.top_k,
        top_p: file_cfg.defaults.top_p,
        repeat_penalty: file_cfg.defaults.repeat_penalty,
        cache_type_k: file_cfg.defaults.cache_type_k.clone(),
        cache_type_v: file_cfg.defaults.cache_type_v.clone(),
        draft_model: file_cfg.defaults.draft_model.clone(),
        spec_type: file_cfg.defaults.spec_type.clone(),
        spec_draft_n_max: file_cfg.defaults.spec_draft_n_max,
        gpu_layers_draft: file_cfg.defaults.gpu_layers_draft,
        threads: sys.gen_threads(),
        auto_ctx,
        embed: file_cfg.embed.clone(),
    };

    let mut llama = ManagedServer::new("llama", cfg.llama_port);
    let embed = ManagedServer::new("embed", file_cfg.embed.port);

    // Auto-load main model. The embed server is NOT started here — it is
    // lazy-loaded on first RAG use (indexing or a retrieval-backed request)
    // so a review-only session never pays its VRAM/startup cost.
    if !models.is_empty() && llama_ok {
        let target = if !file_cfg.defaults.model.is_empty() {
            models.iter().find(|m| m.filename == file_cfg.defaults.model)
        } else {
            Some(&models[0])
        };
        if let Some(m) = target {
            apply_model_params(&mut cfg, m);
            eprintln!("[llama] starting {} (ngl={}, ctx={}, fa={})",
                m.name, if cfg.ngl < 0 { 99 } else { cfg.ngl }, cfg.ctx,
                if cfg.flash_attn { "on" } else { "auto" });
            if llama.spawn(&cfg.llama_binary, &llama_args(&cfg, m), &m.filename, cfg.llama_port).is_ok() {
                llama.wait_ready(cfg.startup_timeout);
            }
        }
    }
    if file_cfg.embed.enabled && !file_cfg.embed.model.is_empty() {
        eprintln!("  embed-server: lazy (starts on first RAG use)");
    }

    let rag = RagStore::new(file_cfg.rag);
    let sys_info = sys.to_json();

    let addr = format!("127.0.0.1:{}", cfg.port);
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("bind {addr}: {e}");
        std::process::exit(1);
    });
    eprintln!("  http://{addr}\n");

    let shared: Shared = Arc::new(Mutex::new(State {
        cfg, models, llama, embed, rag, sys_info, tokens_session: 0, requests: 0,
    }));

    for stream in listener.incoming().flatten() {
        let st = Arc::clone(&shared);
        std::thread::spawn(move || serve(stream, &st));
    }
}

fn apply_model_params(cfg: &mut RuntimeCfg, m: &Model) {
    cfg.active_model = m.filename.clone();
    cfg.ngl = m.gpu_layers;
    cfg.ctx = if m.context_size > 0 { m.context_size } else { cfg.auto_ctx };
    cfg.flash_attn = m.flash_attention;
    cfg.temp = m.temperature;
    cfg.top_k = m.top_k;
    cfg.top_p = m.top_p;
    cfg.repeat_penalty = m.repeat_penalty;
    cfg.spec_type = m.spec_type.clone();
    cfg.spec_draft_n_max = m.spec_draft_n_max;
    cfg.draft_model = m.draft_model.clone();
    cfg.gpu_layers_draft = m.gpu_layers_draft;
}

// ── HTTP server ─────────────────────────────────────────────

fn serve(mut stream: TcpStream, st: &Shared) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(120)));
    let mut reader = BufReader::new(&stream);

    let mut req_line = String::new();
    if reader.read_line(&mut req_line).is_err() { return; }
    let parts: Vec<&str> = req_line.trim().split_whitespace().collect();
    if parts.len() < 2 { return; }
    let (method, path) = (parts[0], parts[1]);

    let mut content_len = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() || line.trim().is_empty() { break; }
        if let Some(rest) = line.to_lowercase().strip_prefix("content-length:") {
            content_len = rest.trim().parse().unwrap_or(0);
        }
    }

    if content_len > MAX_BODY_BYTES {
        respond(&mut stream, 413, "text/plain", "payload too large");
        return;
    }

    let mut body_bytes = vec![0u8; content_len];
    if content_len > 0 { let _ = reader.read_exact(&mut body_bytes); }
    let body = String::from_utf8_lossy(&body_bytes).to_string();

    match (method, path) {
        ("GET", "/")                 => respond(&mut stream, 200, "text/html", INDEX),
        ("GET", "/style.css")        => respond(&mut stream, 200, "text/css", STYLE),
        ("GET", "/script.js")        => respond(&mut stream, 200, "text/javascript", SCRIPT),
        ("GET", "/api/models")       => respond_json(&mut stream, &handle_models(st)),
        ("GET", "/api/status")       => respond_json(&mut stream, &handle_status(st)),
        ("POST", "/api/load")        => respond_json(&mut stream, &handle_load(st, &body)),
        ("POST", "/api/stop")        => respond_json(&mut stream, &handle_stop(st)),
        ("POST", "/api/params")      => respond_json(&mut stream, &handle_params(st, &body)),
        ("POST", "/api/write")       => handle_write_stream(&mut stream, st, &body),
        // Embed server management
        ("GET", "/api/embed/status")  => respond_json(&mut stream, &handle_embed_status(st)),
        ("POST", "/api/embed/start")  => respond_json(&mut stream, &handle_embed_start(st)),
        ("POST", "/api/embed/stop")   => respond_json(&mut stream, &handle_embed_stop(st)),
        ("POST", "/api/embed/prefixes") => respond_json(&mut stream, &handle_embed_prefixes(st, &body)),
        // RAG endpoints
        ("GET", "/api/rag/status")   => respond_json(&mut stream, &handle_rag_status(st)),
        ("POST", "/api/rag/index")   => respond_json(&mut stream, &handle_rag_index(st, &body)),
        ("POST", "/api/rag/search")  => respond_json(&mut stream, &handle_rag_search(st, &body)),
        ("POST", "/api/rag/clear")   => respond_json(&mut stream, &handle_rag_clear(st, &body)),
        _ => respond(&mut stream, 404, "text/plain", "not found"),
    }
}

fn respond(s: &mut TcpStream, code: u16, ct: &str, body: &str) {
    let status = match code {
        200 => "OK", 413 => "Payload Too Large", _ => "Not Found",
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
        "spec": {
            "type": if s.cfg.spec_type.is_empty() { None } else { Some(&s.cfg.spec_type) },
            "draft_n_max": s.cfg.spec_draft_n_max,
            "draft_model": if s.cfg.draft_model.is_empty() { None } else { Some(&s.cfg.draft_model) },
            "gpu_layers_draft": s.cfg.gpu_layers_draft,
        },
        "params": {
            "ngl": s.cfg.ngl, "ctx": s.cfg.ctx, "flash_attn": s.cfg.flash_attn,
            "temp": s.cfg.temp, "top_k": s.cfg.top_k, "top_p": s.cfg.top_p,
            "repeat_penalty": s.cfg.repeat_penalty,
        },
        "embed": s.embed.status_json(),
        "rag": s.rag.status_json(),
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
        "embed": s.embed.status_json(),
        "rag": s.rag.status_json(),
        "system": s.sys_info,
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
    if let Some(v) = req.ngl { cfg.ngl = v; }
    if let Some(v) = req.ctx { cfg.ctx = v.max(2048); }
    if let Some(v) = req.flash_attn { cfg.flash_attn = v; }
    if let Some(v) = req.temp { cfg.temp = v; }
    if let Some(v) = req.top_k { cfg.top_k = v; }
    if let Some(v) = req.top_p { cfg.top_p = v; }
    if let Some(v) = req.repeat_penalty { cfg.repeat_penalty = v; }
    if let Some(v) = req.draft_model { cfg.draft_model = v; }
    if let Some(v) = req.spec_type { cfg.spec_type = v; }
    if let Some(v) = req.spec_draft_n_max { cfg.spec_draft_n_max = v; }
    if let Some(v) = req.gpu_layers_draft { cfg.gpu_layers_draft = v; }

    eprintln!("[llama] starting {} (ngl={}, ctx={}, fa={})",
        model.name, if cfg.ngl < 0 { 99 } else { cfg.ngl }, cfg.ctx,
        if cfg.flash_attn { "on" } else { "auto" });

    // Stop old model + spawn new one under one lock (embed server untouched).
    let status = {
        let mut s = st.lock().unwrap();
        s.llama.stop();
        if let Err(e) = s.llama.spawn(&cfg.llama_binary, &llama_args(&cfg, &model), &model.filename, cfg.llama_port) {
            s.cfg.active_model.clear();
            return serde_json::json!({"error": e});
        }
        s.cfg = cfg.clone();
        s.llama.status_json()
    };

    spawn_ready_poll(st, Which::Llama, cfg.startup_timeout);
    serde_json::json!({"ok": true, "loading": true, "llama": status})
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
    serde_json::json!({"ok": true})
}

// ── Embed server handlers ───────────────────────────────────

fn handle_embed_status(st: &Shared) -> serde_json::Value {
    let s = st.lock().unwrap();
    let mut status = s.embed.status_json();
    status["query_prefix"] = serde_json::json!(s.cfg.embed.query_prefix);
    status["doc_prefix"] = serde_json::json!(s.cfg.embed.doc_prefix);
    status["pooling"] = serde_json::json!(s.cfg.embed.pooling);
    status
}

fn handle_embed_start(st: &Shared) -> serde_json::Value {
    let (binary, models_dir, embed_cfg) = {
        let s = st.lock().unwrap();
        (s.cfg.llama_binary.clone(), s.cfg.models_dir.clone(), s.cfg.embed.clone())
    };

    if !embed_cfg.enabled || embed_cfg.model.is_empty() {
        return serde_json::json!({"error": "embed server not configured — set [embed] model in config.toml"});
    }

    let model_path = format!("{}/{}", models_dir, embed_cfg.model);
    if !Path::new(&model_path).exists() {
        return serde_json::json!({"error": format!("embed model '{}' not found", model_path)});
    }

    let ngl = if embed_cfg.gpu_layers < 0 { 99 } else { embed_cfg.gpu_layers };
    eprintln!("[embed] starting {} (ngl={ngl}, ctx={}, port={})",
        embed_cfg.model, embed_cfg.context_size, embed_cfg.port);
    {
        let mut s = st.lock().unwrap();
        s.embed.stop();
        let args = embed_args(&model_path, &embed_cfg);
        if let Err(e) = s.embed.spawn(&binary, &args, &embed_cfg.model, embed_cfg.port) {
            return serde_json::json!({"error": e});
        }
    }

    spawn_ready_poll(st, Which::Embed, embed_cfg.startup_timeout);
    serde_json::json!({"ok": true, "loading": true})
}

fn handle_embed_stop(st: &Shared) -> serde_json::Value {
    let mut s = st.lock().unwrap();
    s.embed.stop();
    serde_json::json!({"ok": true})
}

fn handle_embed_prefixes(st: &Shared, body: &str) -> serde_json::Value {
    let req: EmbedPrefixReq = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(e) => return serde_json::json!({"error": e.to_string()}),
    };
    let mut s = st.lock().unwrap();
    if let Some(v) = req.query_prefix { s.cfg.embed.query_prefix = v; }
    if let Some(v) = req.doc_prefix { s.cfg.embed.doc_prefix = v; }
    serde_json::json!({
        "ok": true,
        "query_prefix": s.cfg.embed.query_prefix,
        "doc_prefix": s.cfg.embed.doc_prefix,
    })
}

// ── RAG Handlers ────────────────────────────────────────────

fn handle_rag_status(st: &Shared) -> serde_json::Value {
    let s = st.lock().unwrap();
    let mut status = s.rag.status_json();
    status["embed_ready"] = serde_json::json!(s.embed.is_ready());
    status
}

fn handle_rag_index(st: &Shared, body: &str) -> serde_json::Value {
    let req: RagIndexReq = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(e) => return serde_json::json!({"error": e.to_string()}),
    };

    if req.files.is_empty() {
        return serde_json::json!({"error": "no files to index"});
    }

    let domain = if req.domain == "text" { "text" } else { "code" };

    // Lazy-start the embed server on first index (blocks until ready).
    if let Err(e) = ensure_embed_ready(st) {
        return serde_json::json!({"error": e});
    }

    // Phase 1: lock briefly to read config
    let (endpoint, code_doc_prefix, chunk_size, chunk_overlap, chunker_tool) = {
        let s = st.lock().unwrap();
        (s.cfg.embedding_endpoint(), s.cfg.embed.doc_prefix.clone(),
         s.rag.cfg.chunk_size, s.rag.cfg.chunk_overlap, s.rag.cfg.chunker_tool.clone())
    };
    // Lock released here

    // Phase 1b: chunk files outside the lock.
    // Text domain uses the prose chunker; code domain tries the external
    // syntax-aware chunker, falling back to the internal line-window one.
    let (all_chunks, all_sources, all_kinds, all_files) = if domain == "text" {
        let mut chunks = Vec::new();
        let mut sources = Vec::new();
        let mut kinds = Vec::new();
        let mut files = Vec::new();
        for f in &req.files {
            for (label, text, kind, file) in chunk_text_file(&f.name, &f.content) {
                sources.push(label);
                chunks.push(text);
                kinds.push(kind);
                files.push(file);
            }
        }
        (chunks, sources, kinds, files)
    } else if let Some(ext_chunks) =
        try_external_chunker(&chunker_tool, &req.files, chunk_size, chunk_overlap)
    {
        let mut chunks = Vec::with_capacity(ext_chunks.len());
        let mut sources = Vec::with_capacity(ext_chunks.len());
        let mut kinds = Vec::with_capacity(ext_chunks.len());
        let mut files = Vec::with_capacity(ext_chunks.len());
        for (source, text, kind, file) in ext_chunks {
            sources.push(source);
            chunks.push(text);
            kinds.push(kind);
            files.push(file);
        }
        (chunks, sources, kinds, files)
    } else {
        eprintln!("[rag] using internal fallback chunker");
        let mut chunks: Vec<String> = Vec::new();
        let mut sources: Vec<String> = Vec::new();
        let mut kinds: Vec<String> = Vec::new();
        let mut files: Vec<String> = Vec::new();
        for f in &req.files {
            for (label, text, kind, file) in chunk_code_file_simple(&f.name, &f.content, chunk_size, chunk_overlap) {
                sources.push(label);
                chunks.push(text);
                kinds.push(kind);
                files.push(file);
            }
        }
        (chunks, sources, kinds, files)
    };

    if all_chunks.is_empty() {
        return serde_json::json!({"error": "no chunks produced from files"});
    }

    // Domain-appropriate document embedding prefix.
    let doc_prefix = if domain == "text" { TEXT_DOC_PREFIX.to_string() } else { code_doc_prefix };

    // Phase 2: embedding call (network I/O, no lock held)
    eprintln!("[rag] embedding {} '{domain}' chunks from {} files...", all_chunks.len(), req.files.len());
    let t0 = Instant::now();

    let vectors = match get_embeddings_batch(&endpoint, &all_chunks, &doc_prefix) {
        Ok(v) => v,
        Err(e) => return serde_json::json!({"error": e}),
    };

    eprintln!("[rag] {} embeddings in {:.1}s", vectors.len(), t0.elapsed().as_secs_f64());

    // Phase 3: lock briefly to store results
    let file_names: Vec<String> = req.files.iter().map(|f| f.name.clone()).collect();
    let mut s = st.lock().unwrap();
    match s.rag.store_embeddings(all_chunks, all_sources, vectors, file_names, all_kinds, all_files, domain) {
        Ok(count) => serde_json::json!({
            "ok": true,
            "domain": domain,
            "chunks_indexed": count,
            "files": req.files.iter().map(|f| &f.name).collect::<Vec<_>>(),
        }),
        Err(e) => serde_json::json!({"error": e}),
    }
}

fn handle_rag_search(st: &Shared, body: &str) -> serde_json::Value {
    let req: RagSearchReq = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(e) => return serde_json::json!({"error": e.to_string()}),
    };
    let domain = if req.domain == "text" { "text" } else { "code" };

    // Phase 1: lock briefly to get config
    let (endpoint, code_query_prefix, search_limit, has_chunks) = {
        let s = st.lock().unwrap();
        (s.cfg.embedding_endpoint(), s.cfg.embed.query_prefix.clone(),
         s.rag.cfg.search_results, s.rag.domain_count(domain) > 0)
    };

    if !has_chunks {
        return serde_json::json!({"ok": true, "results": []});
    }

    // Domain-appropriate query prefix.
    let query_prefix = if domain == "text" { TEXT_QUERY_PREFIX.to_string() } else { code_query_prefix };

    // Phase 2: embed query (no lock)
    let limit = req.limit.unwrap_or(search_limit);
    let query_vec = match get_embedding(&endpoint, &req.query, &query_prefix) {
        Ok(v) => v,
        Err(e) => return serde_json::json!({"error": e}),
    };

    // Phase 3: lock briefly for similarity search (CPU only, fast)
    let s = st.lock().unwrap();
    let hits = s.rag.search_local(&query_vec, limit, &req.query, domain);
    let results: Vec<serde_json::Value> = hits.iter().map(|(src, text, dist)| {
        serde_json::json!({"source": src, "text": text, "distance": dist})
    }).collect();
    serde_json::json!({"ok": true, "results": results})
}

fn handle_rag_clear(st: &Shared, body: &str) -> serde_json::Value {
    let req: RagClearReq = serde_json::from_str(body).unwrap_or(RagClearReq { domain: None });
    let mut s = st.lock().unwrap();
    let result = match req.domain.as_deref() {
        Some("text") => s.rag.clear_domain("text"),
        Some("code") => s.rag.clear_domain("code"),
        _ => s.rag.clear(),
    };
    match result {
        Ok(()) => serde_json::json!({"ok": true}),
        Err(e) => serde_json::json!({"error": e}),
    }
}

// ── Relevance scoring ───────────────────────────────────────

fn relevance_score(file: &FileEntry, description: &str, target_lang: &str) -> u32 {
    let mut score = 0u32;
    let file_lang = file.language.to_lowercase();
    let target = target_lang.to_lowercase();
    if file_lang == target { score += 10; }
    let fname_lower = file.name.to_lowercase();
    let stem = fname_lower.rsplit('/').next().unwrap_or(&fname_lower);
    let stem = stem.rsplit('.').last().unwrap_or(stem);
    let desc_lower = description.to_lowercase();
    if stem.len() > 2 && desc_lower.contains(stem) { score += 20; }
    for word in desc_lower.split_whitespace() {
        if word.len() > 3 && file.content.contains(word) { score += 2; }
    }
    score
}

// ── Context assembly ────────────────────────────────────────

struct ContextResult {
    context_block: String,
    files_included: Vec<String>,
    files_truncated: Vec<String>,
    files_dropped: Vec<String>,
    total_input_tokens: u64,
    model_ctx: u64,
}

const MIN_OUTPUT_TOKENS: u64 = 256;

fn assemble_context(
    files: &[FileEntry], rag_context: &str,
    description: &str, target_lang: &str, model_ctx: u32, system_text: &str,
) -> ContextResult {
    let system_tok = estimate_tokens(system_text);
    let desc_tok = estimate_tokens(description);
    let rag_tok = estimate_tokens(rag_context);
    let fixed_input = system_tok + desc_tok + rag_tok;
    let file_budget = (model_ctx as u64).saturating_sub(fixed_input + MIN_OUTPUT_TOKENS);

    let mut result = ContextResult {
        context_block: String::new(),
        files_included: Vec::new(), files_truncated: Vec::new(), files_dropped: Vec::new(),
        total_input_tokens: fixed_input, model_ctx: model_ctx as u64,
    };

    if !rag_context.is_empty() {
        result.context_block.push_str(rag_context);
    }

    let mut scored: Vec<(usize, u32)> = files.iter().enumerate()
        .map(|(i, f)| (i, relevance_score(f, description, target_lang))).collect();
    scored.sort_by(|a, b| b.1.cmp(&a.1));

    let mut files_used: u64 = 0;

    for (idx, _) in &scored {
        let f = &files[*idx];
        let lang_tag = if f.language.is_empty() { "text" } else { &f.language };
        let block = format!("\n--- {} ---\n```{}\n{}\n```\n", f.name, lang_tag, f.content);
        let cost = estimate_tokens_lang(&block, lang_tag);

        if files_used + cost <= file_budget {
            result.context_block.push_str(&block);
            files_used += cost;
            result.files_included.push(f.name.clone());
        } else if files_used < file_budget {
            let remaining = file_budget - files_used;
            if remaining > 15 {
                // Use per-language ratio for truncation char budget
                let ratio = match lang_tag {
                    "rust" | "c" | "c++" | "java" | "csharp" => 2.6,
                    "go" | "swift" | "kotlin" | "zig" => 2.8,
                    "javascript" | "typescript" => 2.9,
                    "python" | "ruby" | "lua" => 3.4,
                    _ => 3.2,
                };
                let max_chars = ((remaining - 15) as f64 * ratio) as usize;
                let truncated = if max_chars < f.content.len() {
                    let slice = &f.content[..max_chars.min(f.content.len())];
                    slice.rfind('\n').map(|nl| &f.content[..nl + 1]).unwrap_or(slice)
                } else { &f.content };
                let block = format!("\n--- {} (truncated) ---\n```{}\n{}\n```\n", f.name, lang_tag, truncated);
                files_used += estimate_tokens_lang(&block, lang_tag);
                result.context_block.push_str(&block);
                result.files_truncated.push(f.name.clone());
            } else {
                result.files_dropped.push(f.name.clone());
            }
        } else {
            result.files_dropped.push(f.name.clone());
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

    let is_chat = req.mode == "chat";
    let has_input = if is_chat {
        req.messages.iter().any(|m| m.role == "user" && !m.content.trim().is_empty())
    } else {
        !req.description.is_empty()
    };
    if !has_input {
        send_sse_error(stream, if is_chat { "No message provided" } else { "No description provided" });
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

    let _ = write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
         Cache-Control: no-cache\r\nConnection: keep-alive\r\n\
         Access-Control-Allow-Origin: *\r\n\r\n"
    );
    let _ = stream.flush();

    if is_chat {
        handle_chat_stream(stream, st, &req, &cfg);
        return;
    }

    let is_review = req.mode == "review";

    // ════════════════════════════════════════════════════════════
    // Precomputed token budget — exact values, no estimation.
    // Built once, used throughout the pipeline.
    // ════════════════════════════════════════════════════════════
    struct TokenBudget {
        pub model_ctx: u64,
        pub system_tokens: u64,
        pub desc_tokens: u64,
        pub output_reserve: u64,
        pub context_budget: u64,    // everything available for RAG + files
        pub rag_tokens: u64,        // filled during retrieval
    }

    // Step 1: Build system prompt base (rag_note appended later if chunks found)
    let system_base = if is_review {
        format!(
            "You are a senior {} engineer in a pair-programming conversation.\n\
             The user will ask questions, request explanations, or discuss code \
             they've provided. Respond naturally — like a knowledgeable colleague, \
             not a report generator.\n\n\
             Guidelines:\n\
             - Answer the actual question. Don't run a generic review checklist \
               unless they specifically ask for a review.\n\
             - When you reference specific code, show the relevant snippet in a \
               fenced code block so the user can see exactly what you're talking \
               about. Pull from the provided code context — don't paraphrase \
               field names or signatures from memory.\n\
             - Organize around the concepts the user asked about, not around \
               categories like \"correctness\" or \"security\".\n\
             - Be concrete and specific. \"This Vec<Option<CachedShadowTile>> \
               tracks per-light cache state\" is useful. \"Ensure proper memory \
               management\" is not.\n\
             - If the question is broad (\"explain this struct\"), walk through \
               the logical groups/sections and explain the design — what each \
               cluster of fields does, how they relate, why they're structured \
               that way.",
            req.language
        )
    } else {
        format!(
            "You are an expert {} programmer. Write clean, efficient, well-documented code.\n\
             Output ONLY the code with clear comments. No markdown fences, no prose outside code.",
            req.language
        )
    };

    let rag_note = if is_review {
        "\nRelevant code from the project has been retrieved and included below. \
         Reference it directly — quote specific fields, types, and function \
         signatures when they're relevant to the discussion."
    } else {
        "\nRelevant code context has been retrieved from the project index. \
         Use it to ensure consistency with the existing codebase."
    };

    // Step 2: Compute exact token costs — fixed overhead only.
    // RAG and files share one pool: everything left after system + desc + output reserve.
    let system_tokens = estimate_tokens(&system_base) + estimate_tokens(rag_note);
    let desc_tokens = estimate_tokens(&req.description);
    let context_budget = (cfg.ctx as u64).saturating_sub(
        system_tokens + desc_tokens + MIN_OUTPUT_TOKENS
    );

    let mut budget = TokenBudget {
        model_ctx: cfg.ctx as u64,
        system_tokens,
        desc_tokens,
        output_reserve: MIN_OUTPUT_TOKENS,
        context_budget,
        rag_tokens: 0,
    };

    // ── RAG retrieval (budget-adaptive) ──────────────────────
    let mut rag_context = String::new();
    let mut rag_chunks_used = 0usize;
    if req.use_rag {
        // Only pay embed startup if there's actually something to retrieve.
        let (have_index, endpoint) = {
            let s = st.lock().unwrap();
            (!s.rag.chunks.is_empty() && s.rag.cfg.enabled, cfg.embedding_endpoint())
        };

        // Lazy-start the embed server; on failure, skip RAG and still generate.
        let should_search = have_index && match ensure_embed_ready(st) {
            Ok(()) => true,
            Err(e) => {
                eprintln!("[rag] embed unavailable, generating without retrieval: {e}");
                send_sse(stream, &serde_json::json!({"rag_info": {"error": e}}));
                false
            }
        };

        if should_search {
            const RAG_CANDIDATE_POOL: usize = 30;

            match get_embedding(&endpoint, &req.description, &cfg.embed.query_prefix) {
                Ok(query_vec) => {
                    let s = st.lock().unwrap();
                    let hits = s.rag.search_local(
                        &query_vec, RAG_CANDIDATE_POOL, &req.description, "code",
                    );
                    drop(s);

                    if !hits.is_empty() {
                        let header = "\n--- retrieved context (RAG) ---\n";
                        rag_context.push_str(header);
                        let mut rag_used: u64 = estimate_tokens(header);
                        let mut selected: Vec<(&String, &String, &f32)> = Vec::new();

                        for (source, text, dist) in &hits {
                            let chunk_block = format!(
                                "# {} (score: {:.4})\n{}\n\n", source, dist, text
                            );
                            let cost = estimate_tokens_lang(&chunk_block, &req.language);

                            if rag_used + cost > context_budget {
                                if selected.is_empty() {
                                    rag_context.push_str(&chunk_block);
                                    rag_used += cost;
                                    selected.push((source, text, dist));
                                }
                                break;
                            }

                            rag_context.push_str(&chunk_block);
                            rag_used += cost;
                            selected.push((source, text, dist));
                        }

                        budget.rag_tokens = rag_used;
                        rag_chunks_used = selected.len();
                        eprintln!(
                            "[rag] {} chunks, {} tok (context budget {} tok)",
                            rag_chunks_used, rag_used, context_budget
                        );
                        send_sse(stream, &serde_json::json!({
                            "rag_info": {
                                "chunks_retrieved": rag_chunks_used,
                                "rag_tokens": rag_used,
                                "context_budget": context_budget,
                                "sources": selected.iter().map(|(s, _, d)| {
                                    serde_json::json!({"source": s, "score": d})
                                }).collect::<Vec<_>>(),
                            }
                        }));
                    }
                }
                Err(e) => {
                    eprintln!("[rag] search error: {e}");
                    send_sse(stream, &serde_json::json!({"rag_info": {"error": e}}));
                }
            }
        }
    }

    // Step 3: Finalize system prompt — add rag_note only if chunks found
    let system = if rag_chunks_used > 0 {
        format!("{}{}", system_base, rag_note)
    } else {
        // Reclaim the rag_note tokens we reserved
        budget.system_tokens = estimate_tokens(&system_base);
        system_base
    };

    let ctx_result = assemble_context(
        &req.files, &rag_context,
        &req.description, &req.language, cfg.ctx, &system,
    );

    let remaining = ctx_result.model_ctx.saturating_sub(ctx_result.total_input_tokens);
    if !req.files.is_empty() || rag_chunks_used > 0 {
        let mut info = serde_json::json!({
            "context_info": {
                "model_ctx": budget.model_ctx,
                "system_tokens": budget.system_tokens,
                "desc_tokens": budget.desc_tokens,
                "context_budget": budget.context_budget,
                "rag_tokens": budget.rag_tokens,
                "output_reserve": budget.output_reserve,
                "input_tokens": ctx_result.total_input_tokens,
                "remaining_tokens": remaining,
                "files_included": ctx_result.files_included,
                "files_truncated": ctx_result.files_truncated,
                "files_dropped": ctx_result.files_dropped,
            }
        });
        if rag_chunks_used > 0 {
            info["context_info"]["rag_chunks"] = serde_json::json!(rag_chunks_used);
        }
        send_sse(stream, &info);
    }

    let user = if is_review {
        if ctx_result.context_block.is_empty() {
            req.description.clone()
        } else {
            format!("{}\n\nCode:{}\n", req.description, ctx_result.context_block)
        }
    } else {
        if ctx_result.context_block.is_empty() {
            format!("Write {} code for: {}", req.language, req.description)
        } else {
            format!("Write {} code for: {}\n\nExisting code context:{}\n",
                req.language, req.description, ctx_result.context_block)
        }
    };

    let actual_input = budget.system_tokens
        + estimate_tokens_lang(&user, &req.language);
    let max_tokens = budget.model_ctx.saturating_sub(actual_input);

    if max_tokens < MIN_OUTPUT_TOKENS {
        send_sse(stream, &serde_json::json!({
            "error": format!("Context full — input uses ~{} of {} tokens, only {} left.",
                actual_input, cfg.ctx, max_tokens)
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

    stream_completion(
        stream, st, &cfg.endpoint(), &llama_req,
        serde_json::json!({"rag_chunks": rag_chunks_used}),
    );
}

/// Stream a chat-completions request to the llama server, relaying tokens to
/// the client as SSE. Shared by the write/review and chat pipelines. Kills the
/// upstream generation if the client disconnects, and folds `done_extra` into
/// the terminal `{done:true,...}` event. Updates session token counters.
fn stream_completion(
    stream: &mut TcpStream,
    st: &Shared,
    endpoint: &str,
    llama_req: &serde_json::Value,
    done_extra: serde_json::Value,
) {
    // Unique temp path per call — concurrent requests must not collide.
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = format!("/tmp/cw_{}_{}.json", std::process::id(), seq);

    if fs::write(&tmp, llama_req.to_string()).is_err() {
        send_sse(stream, &serde_json::json!({"error": "failed to write temp file"}));
        return;
    }

    let child = Command::new(curl_cmd())
        .args([
            "-s", "--no-buffer", "-X", "POST",
            endpoint,
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
    let mut aborted = false;

    for line in reader.lines().flatten() {
        let Some(data) = line.strip_prefix("data: ") else { continue };
        if data == "[DONE]" { break; }
        if let Ok(chunk) = serde_json::from_str::<ChatChunk>(data) {
            if let Some(content) = chunk.choices.first().and_then(|c| c.delta.content.as_deref()) {
                if !content.is_empty() {
                    token_count += 1;
                    if !send_sse(stream, &serde_json::json!({"token": content})) {
                        // Client disconnected (abort) — kill curl to stop generation.
                        eprintln!("[gen] client disconnected after {token_count} tokens, killing generation");
                        let _ = child.kill();
                        aborted = true;
                        break;
                    }
                }
            }
        }
    }

    let elapsed_ms = t0.elapsed().as_millis() as u64;
    if !aborted {
        let mut ev = serde_json::json!({
            "done": true, "tokens": token_count, "elapsed_ms": elapsed_ms,
        });
        if let Some(obj) = done_extra.as_object() {
            for (k, v) in obj { ev[k] = v.clone(); }
        }
        send_sse(stream, &ev);
    }

    let _ = fs::remove_file(&tmp);
    let _ = child.wait();

    let mut s = st.lock().unwrap();
    s.tokens_session += token_count;
    s.requests += 1;
}

// ── Streaming chat (multi-turn, text-domain RAG) ────────────
//
// Stateless server: the client owns the thread and sends it whole each turn
// (`req.messages`). The server prepends a general-assistant system prompt,
// optionally grounds it with retrieved *text* chunks, trims oldest turns to
// fit the context window, and streams the reply. No server-side session map —
// there is nothing to evict, persist, or race on.
fn handle_chat_stream(stream: &mut TcpStream, st: &Shared, req: &WriteReq, cfg: &RuntimeCfg) {
    const OUTPUT_RESERVE: u64 = 512;   // roomier reserve for conversational replies
    const RAG_CANDIDATE_POOL: usize = 20;
    const PER_MSG_OVERHEAD: u64 = 8;   // role/formatting tokens per message

    let model_ctx = cfg.ctx as u64;

    let mut system = "You are a helpful, knowledgeable assistant in an ongoing \
        conversation. Answer clearly and concisely, using the conversation history \
        for context. When reference material is provided below, ground your answers \
        in it and say so if it does not cover the question.".to_string();

    // ── Optional RAG over the text corpus ──
    let mut rag_chunks_used = 0usize;
    if req.use_rag {
        let query = req.messages.iter().rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.clone())
            .unwrap_or_default();

        let have_index = {
            let s = st.lock().unwrap();
            s.rag.cfg.enabled && s.rag.domain_count("text") > 0
        };

        if have_index && !query.trim().is_empty() {
            match ensure_embed_ready(st) {
                Ok(()) => {
                    let endpoint = cfg.embedding_endpoint();
                    match get_embedding(&endpoint, &query, TEXT_QUERY_PREFIX) {
                        Ok(qv) => {
                            let hits = {
                                let s = st.lock().unwrap();
                                s.rag.search_local(&qv, RAG_CANDIDATE_POOL, &query, "text")
                            };
                            if !hits.is_empty() {
                                let rag_budget = (model_ctx * 2) / 5;  // ≤40% of ctx
                                let mut block = String::from("\n\n--- reference material ---\n");
                                let mut used = estimate_tokens(&block);
                                let mut sources = Vec::new();
                                for (source, text, score) in &hits {
                                    let piece = format!("[{source}]\n{text}\n\n");
                                    let cost = estimate_tokens(&piece);
                                    if used + cost > rag_budget && rag_chunks_used > 0 { break; }
                                    block.push_str(&piece);
                                    used += cost;
                                    rag_chunks_used += 1;
                                    sources.push(serde_json::json!({"source": source, "score": score}));
                                }
                                system.push_str(&block);
                                send_sse(stream, &serde_json::json!({
                                    "rag_info": {
                                        "chunks_retrieved": rag_chunks_used,
                                        "rag_tokens": used,
                                        "sources": sources,
                                    }
                                }));
                            }
                        }
                        Err(e) => { send_sse(stream, &serde_json::json!({"rag_info": {"error": e}})); }
                    }
                }
                Err(e) => {
                    eprintln!("[chat] embed unavailable, answering without retrieval: {e}");
                    send_sse(stream, &serde_json::json!({"rag_info": {"error": e}}));
                }
            }
        }
    }

    // ── Trim history newest-first to fit the context window ──
    let system_tokens = estimate_tokens(&system);
    let mut budget_used = system_tokens + OUTPUT_RESERVE;
    let mut kept: Vec<&ChatMsg> = Vec::new();
    for m in req.messages.iter().rev() {
        if m.role != "user" && m.role != "assistant" { continue; }
        let cost = estimate_tokens(&m.content) + PER_MSG_OVERHEAD;
        if budget_used + cost > model_ctx && !kept.is_empty() { break; }
        budget_used += cost;
        kept.push(m);
    }
    kept.reverse();

    let turns_kept = kept.len();
    let turns_total = req.messages.iter()
        .filter(|m| m.role == "user" || m.role == "assistant").count();

    let mut messages = vec![serde_json::json!({"role": "system", "content": system})];
    for m in &kept {
        messages.push(serde_json::json!({"role": m.role, "content": m.content}));
    }

    let input_tokens = budget_used - OUTPUT_RESERVE;
    let max_tokens = model_ctx.saturating_sub(input_tokens);
    if max_tokens < MIN_OUTPUT_TOKENS {
        send_sse(stream, &serde_json::json!({
            "error": format!(
                "Thread too long — input ~{} of {} tokens. Start a new chat or clear older turns.",
                input_tokens, model_ctx)
        }));
        return;
    }

    send_sse(stream, &serde_json::json!({
        "context_info": {
            "model_ctx": model_ctx,
            "turns_kept": turns_kept,
            "turns_total": turns_total,
            "input_tokens": input_tokens,
            "remaining_tokens": max_tokens,
            "rag_chunks": rag_chunks_used,
        }
    }));

    let llama_req = serde_json::json!({
        "model": "local",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": cfg.temp,
        "top_p": cfg.top_p,
        "repeat_penalty": cfg.repeat_penalty,
        "stream": true,
    });

    stream_completion(
        stream, st, &cfg.endpoint(), &llama_req,
        serde_json::json!({"rag_chunks": rag_chunks_used, "turns_kept": turns_kept}),
    );
}

/// Send an SSE event to the client.  Returns `false` if the write fails
/// (client disconnected), allowing the caller to abort early.
fn send_sse(stream: &mut TcpStream, val: &serde_json::Value) -> bool {
    let data = serde_json::to_string(val).unwrap_or_default();
    if write!(stream, "data: {data}\n\n").is_err() { return false; }
    stream.flush().is_ok()
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