# SocratiCode Indexing Notes

## Validated Configuration

SocratiCode MCP env:

```json
{
  "EMBEDDING_PROVIDER": "lmstudio",
  "LMSTUDIO_URL": "http://127.0.0.1:8000/v1",
  "EMBEDDING_MODEL": "drawais/Qwen3-Embedding-4B-AWQ-INT4",
  "EMBEDDING_DIMENSIONS": "2560",
  "EMBEDDING_CONTEXT_LENGTH": "32768",
  "EMBEDDING_BATCH_SIZE": "16",
  "QDRANT_MODE": "external",
  "QDRANT_URL": "http://127.0.0.1:17333"
}
```

## Why vLLM AWQ Can Be Slow Here

The AWQ model is fast once loaded, but SocratiCode indexing is not only dense embedding:

- Files are scanned and chunked.
- Every chunk is prefixed with `search_document: <path>`.
- Chunks are embedded in many batches.
- Qdrant stores dense vectors plus server-side BM25 sparse vectors.
- Metadata checkpoints are saved after each file batch.
- A code dependency graph is built after indexing.

Small synthetic embedding tests are not representative of full indexing throughput.

## Known Good Result

On the validated machine:

- Indexable files: `836`
- Stored chunks: `4663`
- Code graph: `352` files, `434` edges
- Total time: `713s`
- End-to-end throughput: about `6.5 chunks/sec`

## OOM Fix

Using `--gpu-memory-utilization 0.94` left only tens of MiB free on a 12 GB GPU and produced vLLM CUDA OOM during real indexing.

The stable setting was:

```text
GPU_MEMORY_UTILIZATION=0.88
EMBEDDING_BATCH_SIZE=16
```

If this still OOMs on another machine, lower one of these:

```text
GPU_MEMORY_UTILIZATION=0.85
EMBEDDING_BATCH_SIZE=8
```

## Commands

Check Qdrant:

```bash
curl http://127.0.0.1:17333/healthz
curl http://127.0.0.1:17333/collections
```

Check embeddings:

```bash
curl http://127.0.0.1:8000/v1/models
node scripts/bench_embeddings.mjs http://127.0.0.1:8000/v1 drawais/Qwen3-Embedding-4B-AWQ-INT4 16
```

Check vLLM logs:

```bash
journalctl -u lazy-qwen3-embedding-awq.service -f
```

Check SocratiCode locks if indexing gets stuck:

```bash
ls /tmp/socraticode-locks
```
