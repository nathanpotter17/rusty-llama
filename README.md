# Rusty Llama - Code Generation Tool

Local code generation UI powered by [llama.cpp](https://github.com/ggerganov/llama.cpp).

`ver. b9254 CUDA 13.1`

## What it does

Describe what you need → streams code from a local GGUF model via llama-server. Supports multiple models, speculative decoding, KV cache quantization, and per-model tuning.

## Build & Run

```bash
cargo build --release
cp config.toml target/release/
cd target/release
# put .gguf models in ./models/
# put llama-server binary in PATH or ./models/
./code-review-server
```

Open `http://localhost:8090`.

## Config

Edit `config.toml` to set models dir, GPU layers, context size, sampling params, draft model, etc. Unknown `.gguf` files in the models dir auto-load with defaults.

## Stack

- **Backend:** Rust — raw TCP, no web framework. Embeds HTML/CSS/JS. Manages llama-server as a child process.
- **Frontend:** Vanilla JS. SSE streaming. Settings overlay for model/param management.
- **Inference:** llama-server (OpenAI-compatible `/v1/chat/completions`), proxied via curl.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/models` | GET | List models + active params |
| `/api/status` | GET | Usage stats + server state |
| `/api/load` | POST | Load a model (restarts llama-server) |
| `/api/stop` | POST | Kill llama-server |
| `/api/params` | POST | Update sampling params live |
| `/api/write` | POST | Stream code generation (SSE) |

## License

MIT