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
            hnsw_m: 16,
            hnsw_ef_construction: 150,
            hnsw_ef_search: 50,
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
    draft_max: u32,
    gpu_layers_draft: i32,
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

fn chunk_code_file(name: &str, content: &str, chunk_size: usize, overlap: usize) -> Vec<(String, String)> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() { return Vec::new(); }
    let mut chunks = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let end = std::cmp::min(i + chunk_size, lines.len());
        let chunk = lines[i..end].join("\n");
        let label = format!("{}:{}-{}", name, i + 1, end);
        chunks.push((label, chunk));
        if end == lines.len() { break; }
        i += chunk_size.saturating_sub(overlap);
    }
    chunks
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

    /// Prepare chunks from files (no network, safe to call under lock).
    fn prepare_chunks(&self, files: &[FileEntry]) -> (Vec<String>, Vec<String>) {
        let mut all_chunks: Vec<String> = Vec::new();
        let mut all_sources: Vec<String> = Vec::new();
        for f in files {
            let chunks = chunk_code_file(
                &f.name, &f.content,
                self.cfg.chunk_size, self.cfg.chunk_overlap,
            );
            for (label, text) in chunks {
                all_sources.push(label);
                all_chunks.push(text);
            }
        }
        (all_chunks, all_sources)
    }

    /// Store pre-computed embeddings and build the HNSW graph.
    fn store_embeddings(
        &mut self,
        chunks: Vec<String>,
        sources: Vec<String>,
        vectors: Vec<Vec<f32>>,
        file_names: Vec<String>,
    ) -> Result<usize, String> {
        if vectors.is_empty() { return Err("no embeddings".into()); }
        let dim = vectors[0].len();
        if dim == 0 { return Err("embedding dimension is 0".into()); }

        self.chunks.clear();
        self.chunks.reserve(chunks.len());
        let iter = chunks.into_iter()
            .zip(sources.into_iter())
            .zip(vectors.into_iter());
        for ((text, source), vector) in iter {
            self.chunks.push(VecChunk { text, source, vector });
        }
        self.indexed_files = file_names;

        // Build HNSW graph over the vectors
        let t0 = Instant::now();
        let vecs: Vec<Vec<f32>> = self.chunks.iter().map(|c| c.vector.clone()).collect();
        self.graph = HnswGraph::new(self.cfg.hnsw_m, self.cfg.hnsw_ef_construction);
        self.graph.build_all(&vecs);
        eprintln!("[rag] HNSW built in {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

        if let Err(e) = self.save() {
            eprintln!("[rag] save warning: {e}");
        }
        eprintln!("[rag] indexed {} chunks (dim={}, persisted to {})", self.chunks.len(), dim, self.cfg.db_path);
        Ok(self.chunks.len())
    }

    /// Search using HNSW graph.  Falls back to brute force if the graph
    /// is out of sync (shouldn't happen, but defensive).
    fn search_local(
        &self,
        query_vec: &[f32],
        limit: usize,
        _query_hint: &str,
    ) -> Vec<(String, String, f32)> {
        if self.chunks.is_empty() { return Vec::new(); }

        if !self.graph.is_empty() && self.graph.len() == self.chunks.len() {
            // HNSW search path — O(log n), zero-copy vector access
            let vec_refs: Vec<&[f32]> = self.chunks.iter().map(|c| c.vector.as_slice()).collect();
            let results = self.graph.knn_search(
                query_vec, limit, self.cfg.hnsw_ef_search, &vec_refs,
            );
            results.iter()
                .map(|dn| {
                    let c = &self.chunks[dn.id];
                    (c.source.clone(), c.text.clone(), distance_to_similarity(dn.dist))
                })
                .collect()
        } else {
            // Fallback: brute-force (graph missing or stale)
            eprintln!("[rag] WARN: HNSW graph missing/stale, falling back to brute force");
            let mut scored: Vec<(usize, f32)> = self.chunks.iter().enumerate()
                .map(|(i, c)| (i, cosine_distance(query_vec, &c.vector)))
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.iter()
                .take(limit)
                .map(|(i, d)| {
                    let c = &self.chunks[*i];
                    (c.source.clone(), c.text.clone(), distance_to_similarity(*d))
                })
                .collect()
        }
    }

    fn clear(&mut self) -> Result<(), String> {
        self.chunks.clear();
        self.graph.clear();
        self.indexed_files.clear();
        let _ = fs::remove_file(&self.cfg.db_path);
        eprintln!("[rag] index cleared");
        Ok(())
    }

    /// Persist index in binary format:
    ///   [u32 chunk_count] [u32 vector_dim]
    ///   for each chunk:
    ///     [u32 source_len] [source_bytes]
    ///     [u32 text_len] [text_bytes]
    ///     [f32 * dim]
    ///   [HNSW graph bytes]
    fn save(&self) -> Result<(), String> {
        if let Some(parent) = Path::new(&self.cfg.db_path).parent() {
            let _ = fs::create_dir_all(parent);
        }
        let dim = self.chunks.first().map(|c| c.vector.len()).unwrap_or(0) as u32;
        let count = self.chunks.len() as u32;

        let mut buf = Vec::with_capacity(8 + self.chunks.len() * (8 + 200 + dim as usize * 4));
        buf.extend_from_slice(&count.to_le_bytes());
        buf.extend_from_slice(&dim.to_le_bytes());

        for c in &self.chunks {
            let src = c.source.as_bytes();
            let txt = c.text.as_bytes();
            buf.extend_from_slice(&(src.len() as u32).to_le_bytes());
            buf.extend_from_slice(src);
            buf.extend_from_slice(&(txt.len() as u32).to_le_bytes());
            buf.extend_from_slice(txt);
            for &v in &c.vector {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }

        self.graph.save_to(&mut buf);

        fs::write(&self.cfg.db_path, &buf)
            .map_err(|e| format!("write {}: {e}", self.cfg.db_path))
    }

    /// Load persisted index from disk.
    fn load(&mut self) -> Result<(), String> {
        let path = Path::new(&self.cfg.db_path);
        if !path.exists() { return Ok(()); }
        let data = fs::read(path)
            .map_err(|e| format!("read {}: {e}", self.cfg.db_path))?;
        if data.len() < 8 { return Ok(()); }

        let count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        let mut pos = 8;

        self.chunks.clear();
        self.chunks.reserve(count);

        for i in 0..count {
            if pos + 4 > data.len() { return Err(format!("truncated at chunk {i} (source len)")); }
            let src_len = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + src_len > data.len() { return Err(format!("truncated at chunk {i} (source)")); }
            let source = String::from_utf8_lossy(&data[pos..pos+src_len]).to_string();
            pos += src_len;

            if pos + 4 > data.len() { return Err(format!("truncated at chunk {i} (text len)")); }
            let txt_len = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + txt_len > data.len() { return Err(format!("truncated at chunk {i} (text)")); }
            let text = String::from_utf8_lossy(&data[pos..pos+txt_len]).to_string();
            pos += txt_len;

            let vec_bytes = dim * 4;
            if pos + vec_bytes > data.len() { return Err(format!("truncated at chunk {i} (vector)")); }
            let vector: Vec<f32> = (0..dim)
                .map(|j| {
                    let o = pos + j * 4;
                    f32::from_le_bytes(data[o..o+4].try_into().unwrap())
                })
                .collect();
            pos += vec_bytes;

            self.chunks.push(VecChunk { text, source, vector });
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

        // Reconstruct indexed_files from chunk sources
        let mut files: Vec<String> = self.chunks.iter()
            .map(|c| c.source.split(':').next().unwrap_or("").to_string())
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
            "vector_dim": self.vector_dim(),
            "files": self.indexed_files,
            "db_path": self.cfg.db_path,
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

// ── Embedding server (second llama-server process) ──────────

struct EmbedServer {
    child: Option<Child>,
    status: LlamaStatus,
    model: String,
    pid: Option<u32>,
    port: u16,
}

impl EmbedServer {
    fn new(port: u16) -> Self {
        Self { child: None, status: LlamaStatus::Stopped, model: String::new(), pid: None, port }
    }

    fn start(&mut self, binary: &str, model_path: &str, model_name: &str, cfg: &EmbedCfg) -> Result<(), String> {
        self.stop();
        let ngl = if cfg.gpu_layers < 0 { 99 } else { cfg.gpu_layers };
        eprintln!("[embed] starting {} (ngl={ngl}, ctx={}, port={})", model_name, cfg.context_size, cfg.port);

        let ctx_str = cfg.context_size.to_string();
        let port_str = cfg.port.to_string();
        let ngl_str = ngl.to_string();
        let slots_str = cfg.parallel_slots.to_string();

        let mut args = vec![
            "-m", model_path,
            "--port", &port_str,
            "-ngl", &ngl_str,
            "-c", &ctx_str,
            "-ub", &ctx_str,
            "-np", &slots_str,
            "--host", "127.0.0.1",
            "--embedding",
        ];

        if !cfg.pooling.is_empty() {
            args.push("--pooling");
            args.push(&cfg.pooling);
        }

        let child = Command::new(binary)
            .args(&args)
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("spawn embed: {e}"))?;

        self.pid = Some(child.id());
        self.child = Some(child);
        self.status = LlamaStatus::Starting;
        self.model = model_name.to_string();
        self.port = cfg.port;
        Ok(())
    }

    fn wait_ready(&mut self, timeout_secs: u64) -> bool {
        let deadline = Instant::now() + Duration::from_secs(timeout_secs);
        std::thread::sleep(Duration::from_millis(400));

        while Instant::now() < deadline {
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

            if check_health(self.port) {
                eprintln!("[embed] ready (pid {:?})", self.pid);
                self.status = LlamaStatus::Ready;
                return true;
            }
            std::thread::sleep(Duration::from_millis(600));
        }
        self.status = LlamaStatus::Error(format!("timeout ({timeout_secs}s)"));
        false
    }

    fn stop(&mut self) {
        if let Some(mut c) = self.child.take() {
            eprintln!("[embed] killing pid {:?}", self.pid);
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
            "port": self.port,
            "error": match &self.status {
                LlamaStatus::Error(e) => Some(e.as_str()),
                _ => None,
            },
        })
    }
}

impl Drop for EmbedServer {
    fn drop(&mut self) { self.stop(); }
}

// ── Main llama server management ────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize)]
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

        // NOTE: No --embedding here. Embeddings are handled by the
        // dedicated EmbedServer process so the main model stays
        // focused on generation with maximum KV cache available.

        if !cfg.cache_type_k.is_empty() {
            args.extend(["--cache-type-k".into(), cfg.cache_type_k.clone()]);
        }
        if !cfg.cache_type_v.is_empty() {
            args.extend(["--cache-type-v".into(), cfg.cache_type_v.clone()]);
        }

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
                eprintln!("[llama]   WARNING: draft model '{}' not found", draft_path);
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
        let deadline = Instant::now() + Duration::from_secs(timeout_secs);
        std::thread::sleep(Duration::from_millis(500));

        while Instant::now() < deadline {
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

            if check_health(port) {
                eprintln!("[llama] ready (pid {:?})", self.pid);
                self.status = LlamaStatus::Ready;
                return true;
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

// ── Shared state ────────────────────────────────────────────

struct State {
    cfg: RuntimeCfg,
    models: Vec<Model>,
    llama: LlamaServer,
    embed: EmbedServer,
    rag: RagStore,
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
    #[serde(default)]
    use_rag: bool,
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
    #[serde(default)] draft_max: Option<u32>,
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
}

#[derive(Deserialize)]
struct RagSearchReq {
    query: String,
    #[serde(default)]
    limit: Option<usize>,
}

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
        embed: file_cfg.embed.clone(),
    };

    let mut llama = LlamaServer::new();
    let mut embed = EmbedServer::new(file_cfg.embed.port);

    // Auto-load main model
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

    // Auto-start embed server if configured
    if file_cfg.embed.enabled && !file_cfg.embed.model.is_empty() && llama_ok {
        let embed_path = format!("{}/{}", cfg.models_dir, file_cfg.embed.model);
        if Path::new(&embed_path).exists() {
            if embed.start(&cfg.llama_binary, &embed_path, &file_cfg.embed.model, &file_cfg.embed).is_ok() {
                embed.wait_ready(file_cfg.embed.startup_timeout);
            }
        } else {
            eprintln!("  WARNING: embed model '{}' not found", embed_path);
        }
    }

    let rag = RagStore::new(file_cfg.rag);

    let addr = format!("127.0.0.1:{}", cfg.port);
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("bind {addr}: {e}");
        std::process::exit(1);
    });
    eprintln!("  http://{addr}\n");

    let shared: Shared = Arc::new(Mutex::new(State {
        cfg, models, llama, embed, rag, tokens_session: 0, requests: 0,
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
        ("POST", "/api/rag/clear")   => respond_json(&mut stream, &handle_rag_clear(st)),
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
        "draft": {
            "model": if s.cfg.draft_model.is_empty() { None } else { Some(&s.cfg.draft_model) },
            "max": s.cfg.draft_max,
            "ngl": s.cfg.gpu_layers_draft,
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
    if let Some(v) = req.draft_max { cfg.draft_max = v; }
    if let Some(v) = req.gpu_layers_draft { cfg.gpu_layers_draft = v; }

    // Stop main model only (embed server stays up)
    st.lock().unwrap().llama.stop();

    let mut llama = LlamaServer::new();
    if let Err(e) = llama.start(&cfg, &model) {
        st.lock().unwrap().cfg.active_model.clear();
        return serde_json::json!({"error": e});
    }

    let status = llama.status_json();
    {
        let mut s = st.lock().unwrap();
        s.llama = llama;
        s.cfg = cfg.clone();
    }

    // Background poll for main model readiness
    let bg_st = Arc::clone(st);
    let llama_port = cfg.llama_port;
    let timeout = cfg.startup_timeout;
    std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(timeout);
        std::thread::sleep(Duration::from_millis(500));

        loop {
            if Instant::now() >= deadline {
                let mut s = bg_st.lock().unwrap();
                s.llama.status = LlamaStatus::Error(format!("timeout ({timeout}s)"));
                return;
            }
            {
                let mut s = bg_st.lock().unwrap();
                if let Some(ref mut c) = s.llama.child {
                    match c.try_wait() {
                        Ok(Some(st_code)) => {
                            s.llama.status = LlamaStatus::Error(format!("exited: {st_code}"));
                            s.llama.child = None;
                            return;
                        }
                        Err(e) => { s.llama.status = LlamaStatus::Error(format!("wait: {e}")); return; }
                        Ok(None) => {}
                    }
                } else {
                    s.llama.status = LlamaStatus::Error("process gone".into());
                    return;
                }
            }
            if check_health(llama_port) {
                let mut s = bg_st.lock().unwrap();
                s.llama.status = LlamaStatus::Ready;
                eprintln!("[load] ready");
                return;
            }
            std::thread::sleep(Duration::from_millis(800));
        }
    });

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

    let mut s = st.lock().unwrap();
    s.embed.stop();
    if let Err(e) = s.embed.start(&binary, &model_path, &embed_cfg.model, &embed_cfg) {
        return serde_json::json!({"error": e});
    }

    // Brief sync wait — embed models are small and load fast
    drop(s);
    let timeout = embed_cfg.startup_timeout;

    // Background poll
    let bg_st = Arc::clone(st);
    let embed_port = embed_cfg.port;
    std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(timeout);
        std::thread::sleep(Duration::from_millis(400));

        loop {
            if Instant::now() >= deadline {
                let mut s = bg_st.lock().unwrap();
                s.embed.status = LlamaStatus::Error(format!("timeout ({timeout}s)"));
                return;
            }
            {
                let mut s = bg_st.lock().unwrap();
                if let Some(ref mut c) = s.embed.child {
                    match c.try_wait() {
                        Ok(Some(st_code)) => {
                            s.embed.status = LlamaStatus::Error(format!("exited: {st_code}"));
                            s.embed.child = None;
                            return;
                        }
                        Err(e) => { s.embed.status = LlamaStatus::Error(format!("wait: {e}")); return; }
                        Ok(None) => {}
                    }
                } else {
                    s.embed.status = LlamaStatus::Error("process gone".into());
                    return;
                }
            }
            if check_health(embed_port) {
                let mut s = bg_st.lock().unwrap();
                s.embed.status = LlamaStatus::Ready;
                eprintln!("[embed] ready via API start");
                return;
            }
            std::thread::sleep(Duration::from_millis(600));
        }
    });

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

    // Phase 1: lock briefly to check state and prepare chunks
    let (embed_ready, endpoint, doc_prefix, all_chunks, all_sources) = {
        let s = st.lock().unwrap();
        let ready = s.embed.is_ready();
        let ep = s.cfg.embedding_endpoint();
        let pfx = s.cfg.embed.doc_prefix.clone();
        let (chunks, sources) = s.rag.prepare_chunks(&req.files);
        (ready, ep, pfx, chunks, sources)
    };
    // Lock released here — other API calls can proceed

    if !embed_ready {
        return serde_json::json!({"error": "embed server not ready — start it from settings"});
    }
    if all_chunks.is_empty() {
        return serde_json::json!({"error": "no chunks produced from files"});
    }

    // Phase 2: embedding call (network I/O, no lock held)
    eprintln!("[rag] embedding {} chunks from {} files...", all_chunks.len(), req.files.len());
    let t0 = Instant::now();

    let vectors = match get_embeddings_batch(&endpoint, &all_chunks, &doc_prefix) {
        Ok(v) => v,
        Err(e) => return serde_json::json!({"error": e}),
    };

    eprintln!("[rag] {} embeddings in {:.1}s", vectors.len(), t0.elapsed().as_secs_f64());

    // Phase 3: lock briefly to store results
    let file_names: Vec<String> = req.files.iter().map(|f| f.name.clone()).collect();
    let mut s = st.lock().unwrap();
    match s.rag.store_embeddings(all_chunks, all_sources, vectors, file_names) {
        Ok(count) => serde_json::json!({
            "ok": true,
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

    // Phase 1: lock briefly to get config
    let (endpoint, query_prefix, search_limit, has_chunks) = {
        let s = st.lock().unwrap();
        (s.cfg.embedding_endpoint(), s.cfg.embed.query_prefix.clone(),
         s.rag.cfg.search_results, !s.rag.chunks.is_empty())
    };

    if !has_chunks {
        return serde_json::json!({"ok": true, "results": []});
    }

    // Phase 2: embed query (no lock)
    let limit = req.limit.unwrap_or(search_limit);
    let query_vec = match get_embedding(&endpoint, &req.query, &query_prefix) {
        Ok(v) => v,
        Err(e) => return serde_json::json!({"error": e}),
    };

    // Phase 3: lock briefly for similarity search (CPU only, fast)
    let s = st.lock().unwrap();
    let hits = s.rag.search_local(&query_vec, limit, &req.query);
    let results: Vec<serde_json::Value> = hits.iter().map(|(src, text, dist)| {
        serde_json::json!({"source": src, "text": text, "distance": dist})
    }).collect();
    serde_json::json!({"ok": true, "results": results})
}

fn handle_rag_clear(st: &Shared) -> serde_json::Value {
    let mut s = st.lock().unwrap();
    match s.rag.clear() {
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
    files: &[FileEntry], extra_ctx: &str, rag_context: &str,
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

    if !extra_ctx.is_empty() {
        let block = format!("\n--- additional context ---\n```\n{}\n```\n", extra_ctx);
        let cost = estimate_tokens(&block);
        if files_used + cost <= file_budget {
            result.context_block.push_str(&block);
            files_used += cost;
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

    let _ = write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
         Cache-Control: no-cache\r\nConnection: keep-alive\r\n\
         Access-Control-Allow-Origin: *\r\n\r\n"
    );
    let _ = stream.flush();

    let is_review = req.mode == "review";

    // ── RAG retrieval (lock-free embedding) ──
    let mut rag_context = String::new();
    let mut rag_chunks_used = 0usize;
    if req.use_rag {
        // Phase 1: lock briefly to check state
        let (should_search, endpoint, search_limit) = {
            let s = st.lock().unwrap();
            let ok = !s.rag.chunks.is_empty() && s.rag.cfg.enabled && s.embed.is_ready();
            (ok, cfg.embedding_endpoint(), s.rag.cfg.search_results)
        };

        if should_search {
            // Phase 2: embed query (no lock)
            match get_embedding(&endpoint, &req.description, &cfg.embed.query_prefix) {
                Ok(query_vec) => {
                    // Phase 3: lock briefly for similarity (CPU only)
                    let s = st.lock().unwrap();
                    let hits = s.rag.search_local(&query_vec, search_limit, &req.description);
                    drop(s); // release immediately

                    if !hits.is_empty() {
                        rag_context.push_str("\n--- retrieved context (RAG) ---\n");
                        for (source, text, dist) in &hits {
                            rag_context.push_str(&format!("# {} (distance: {:.4})\n{}\n\n", source, dist, text));
                        }
                        rag_chunks_used = hits.len();
                        eprintln!("[rag] retrieved {} chunks for query", hits.len());
                        send_sse(stream, &serde_json::json!({
                            "rag_info": {
                                "chunks_retrieved": hits.len(),
                                "sources": hits.iter().map(|(s, _, d)| {
                                    serde_json::json!({"source": s, "distance": d})
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
        let rag_note = if rag_chunks_used > 0 {
            "\nRelevant code context has been retrieved from the project index. \
             Use it to ensure consistency with the existing codebase."
        } else { "" };
        format!(
            "You are an expert {} programmer. Write clean, efficient, well-documented code.\n\
             Output ONLY the code with clear comments. No markdown fences, no prose outside code.{}",
            req.language, rag_note
        )
    };

    let ctx_result = assemble_context(
        &req.files, &req.context, &rag_context,
        &req.description, &req.language, cfg.ctx, &system,
    );

    let remaining = ctx_result.model_ctx.saturating_sub(ctx_result.total_input_tokens);
    if !req.files.is_empty() || !req.context.is_empty() || rag_chunks_used > 0 {
        let mut info = serde_json::json!({
            "context_info": {
                "model_ctx": ctx_result.model_ctx,
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
            format!("{}\n\nCode to review:{}\n", req.description, ctx_result.context_block)
        }
    } else {
        if ctx_result.context_block.is_empty() {
            format!("Write {} code for: {}", req.language, req.description)
        } else {
            format!("Write {} code for: {}\n\nExisting code context:{}\n",
                req.language, req.description, ctx_result.context_block)
        }
    };

    let actual_input = estimate_tokens(&system) + estimate_tokens_lang(&user, &req.language);
    let max_tokens = (cfg.ctx as u64).saturating_sub(actual_input);

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
        "rag_chunks": rag_chunks_used,
    }));

    let _ = fs::remove_file(&tmp);
    let _ = child.wait();

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