# CodeWriter

Local code generation & review UI powered by [llama.cpp](https://github.com/ggerganov/llama.cpp), with built-in RAG.

Supported Versions
`ver. b9870 CUDA 13.3`
`ver. b9254 CUDA 13.1`

## Features

- **Write / Review modes** - generate code or get structured reviews from a local GGUF model
- **RAG** - upload files → chunk → embed → HNSW index → semantic search → inject relevant context into prompts
- **Dedicated embed server** - separate llama-server process for embeddings; configurable pooling strategy and query/doc prefixes per model
- **Speculative decoding** - optional draft model for faster inference
- **MTP Support** - MTP Capable models can run via `spec_type = "draft-mtp"` in `config.toml`
- **Flash attention** - enabled by default, configurable per model
- **KV cache quantization** - `q8_0` / `q4_0` cache types via config
- **Multi-model** - auto-discovers `.gguf` files, per-model params in `config.toml`
- **SSE streaming** - token-by-token output

## Build & Run

```bash
cargo build --release
cp config.toml target/release/
cd target/release
# place .gguf models in ./models/
# place llama-server binary in PATH or ./models/
cargo run --release
```

Open `http://localhost:8090`.

## Config

`config.toml` controls server ports, model defaults, embed server, RAG, and per-model overrides. Key sections:

| Section | Purpose |
|---|---|
| `[server]` | HTTP port (default 8090) |
| `[llama]` | llama-server binary, port, slots, timeout |
| `[limits]` | Session and daily token caps (0 = unlimited) |
| `[defaults]` | GPU layers, context size, flash attention, sampling params, KV cache types, draft model |
| `[embed]` | Embed server model, port, GPU layers, context, pooling, query/doc prefixes |
| `[rag]` | Chunk size/overlap, search results count, HNSW params (`M`, `ef_construction`, `ef_search`), DB path |
| `[[models]]` | Per-model filename, family, params |

## Stack

- **Backend:** Rust (v0.2.0) - raw TCP, no framework. Embeds HTML/CSS/JS via `include_str!`. Manages llama-server + embed-server as child processes. Proxies inference via curl.
- **Frontend:** Vanilla JS. SSE streaming. Drag-and-drop file upload with context/RAG destination toggle.
- **Inference:** llama-server OpenAI-compatible `/v1/chat/completions` (generation) + `/v1/embeddings` (RAG).
- **Vector store:** HNSW graph index over cosine distance. JSON-persisted chunks + serialized graph. Configurable `M`, `ef_construction`, `ef_search`.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/models` | GET | List models, active params, draft/embed/RAG status |
| `/api/status` | GET | Usage stats, server state |
| `/api/load` | POST | Load model (restarts llama-server) |
| `/api/stop` | POST | Kill llama-server |
| `/api/params` | POST | Update sampling params live |
| `/api/write` | POST | Stream code generation or review (SSE) |
| `/api/embed/status` | GET | Embed server status |
| `/api/embed/start` | POST | Start embed server |
| `/api/embed/stop` | POST | Stop embed server |
| `/api/embed/prefixes` | POST | Update query/doc embedding prefixes at runtime |
| `/api/rag/status` | GET | RAG index status (chunks, files, dim) |
| `/api/rag/index` | POST | Index files into vector store |
| `/api/rag/search` | POST | Semantic search over indexed chunks |
| `/api/rag/clear` | POST | Clear RAG index |

## License

MIT
