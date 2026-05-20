# SocratiCode AWQ Embedding Setup

Reproducible local embedding stack for SocratiCode using a lazy vLLM proxy, Qwen3-Embedding AWQ, and Qdrant.

This setup was validated with:

- `drawais/Qwen3-Embedding-4B-AWQ-INT4`
- vLLM OpenAI-compatible `/v1/embeddings`
- Qdrant `v1.17.0`
- SocratiCode external Qdrant mode
- 2560-dimensional vectors

## Architecture

```text
SocratiCode MCP
  -> OpenAI-compatible embedding endpoint: http://127.0.0.1:8000/v1
  -> lazy FastAPI proxy
  -> vLLM pooling server on 127.0.0.1:18000
  -> Qwen3-Embedding-4B AWQ INT4 on GPU
  -> Qdrant on 127.0.0.1:17333
```

The lazy proxy keeps the public embedding endpoint alive and starts vLLM only when a request arrives. After an idle period, it stops vLLM to free GPU memory.

## Files

- `scripts/lazy_vllm_proxy_sleep.py` - Lazy FastAPI reverse proxy that starts/stops vLLM.
- `systemd/lazy-qwen3-embedding-awq.service` - systemd service template.
- `systemd/lazy-qwen3-embedding-awq.env.example` - vLLM/AWQ environment template.
- `systemd/socraticode-qdrant.service` - optional local Qdrant service template.
- `opencode/socraticode-mcp-env.json` - SocratiCode MCP env snippet for opencode.
- `scripts/bench_embeddings.mjs` - quick embedding endpoint benchmark.
- `docs/socraticode-indexing.md` - indexing and performance notes.

## Requirements

- Linux with systemd
- NVIDIA GPU with CUDA working
- Python environment with `vllm`, `fastapi`, `uvicorn`, and `httpx`
- Docker if using the Qdrant systemd service in this repo
- SocratiCode installed separately

Example Python packages:

```bash
pip install vllm fastapi uvicorn httpx
```

## Install Lazy AWQ Embedding Service

Copy the service files:

```bash
sudo cp systemd/lazy-qwen3-embedding-awq.service /etc/systemd/system/lazy-qwen3-embedding-awq.service
sudo cp systemd/lazy-qwen3-embedding-awq.env.example /etc/lazy-qwen3-embedding-awq.env
sudo cp scripts/lazy_vllm_proxy_sleep.py /usr/local/bin/lazy_vllm_proxy_sleep.py
sudo chmod +x /usr/local/bin/lazy_vllm_proxy_sleep.py
```

Edit `/etc/lazy-qwen3-embedding-awq.env` for your machine:

- Set `VLLM_BIN` to your vLLM executable.
- Set `GPU_DEVICE` to the CUDA GPU index to use.
- Keep `GPU_MEMORY_UTILIZATION=0.88` as a safe starting point on 12 GB GPUs.
- Increase only after testing real indexing workloads.

Start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now lazy-qwen3-embedding-awq.service
```

Verify:

```bash
curl http://127.0.0.1:8000/v1/models
node scripts/bench_embeddings.mjs http://127.0.0.1:8000/v1 drawais/Qwen3-Embedding-4B-AWQ-INT4 16
```

## Install Qdrant

If SocratiCode is already managing Qdrant, use that. Otherwise, this repo includes an optional Docker-backed Qdrant service on ports `17333` and `17334`:

```bash
sudo cp systemd/socraticode-qdrant.service /etc/systemd/system/socraticode-qdrant.service
sudo systemctl daemon-reload
sudo systemctl enable --now socraticode-qdrant.service
```

Verify:

```bash
curl http://127.0.0.1:17333/healthz
curl http://127.0.0.1:17333/collections
```

## SocratiCode Configuration

Use the env values in `opencode/socraticode-mcp-env.json` for your SocratiCode MCP server:

```json
{
  "EMBEDDING_PROVIDER": "lmstudio",
  "LMSTUDIO_URL": "http://127.0.0.1:8000/v1",
  "EMBEDDING_MODEL": "drawais/Qwen3-Embedding-4B-AWQ-INT4",
  "EMBEDDING_DIMENSIONS": "2560",
  "EMBEDDING_CONTEXT_LENGTH": "32768",
  "EMBEDDING_BATCH_SIZE": "16",
  "QDRANT_MODE": "external",
  "QDRANT_URL": "http://127.0.0.1:17333",
  "QDRANT_COLLECTION_PREFIX": ""
}
```

Restart opencode after changing MCP config. opencode does not hot-reload MCP env values.

## Performance Notes

On a 12 GB GPU, `GPU_MEMORY_UTILIZATION=0.94` caused vLLM OOM during real SocratiCode indexing. `0.88` left enough headroom.

Validated SocratiCode run:

- Files: `836`
- Chunks: `4663`
- Code graph: `352` files, `434` edges
- Total indexing time: `713s`
- Average: about `6.5 chunks/sec` end-to-end including embedding, BM25, Qdrant writes, metadata, and graph build

Small synthetic batches can look much faster than real indexing. Real codebase chunks are longer, include path prefixes, and SocratiCode also performs sparse BM25 indexing and checkpoint writes.

## Troubleshooting

Check service logs:

```bash
journalctl -u lazy-qwen3-embedding-awq.service -f
```

If vLLM OOMs:

- Lower `GPU_MEMORY_UTILIZATION` to `0.85` or `0.80`.
- Lower SocratiCode `EMBEDDING_BATCH_SIZE` to `8`.
- Restart the service after env changes.

If SocratiCode keeps reporting no index:

- Confirm the MCP process was restarted after config changes.
- Confirm Qdrant is reachable on the same URL SocratiCode uses.
- Check for stale SocratiCode locks in `/tmp/socraticode-locks`.
