const baseUrl = process.argv[2] || "http://127.0.0.1:8000/v1";
const model = process.argv[3] || "drawais/Qwen3-Embedding-4B-AWQ-INT4";
const batchSize = Number(process.argv[4] || "16");

const input = Array.from({ length: batchSize }, (_, i) => (
  `search_document: bench_${i}\n${"function test() { return value + 1; }\n".repeat(60)}`
));

const startedAt = Date.now();
const response = await fetch(`${baseUrl.replace(/\/$/, "")}/embeddings`, {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({
    model,
    input,
    encoding_format: "float",
  }),
});

const text = await response.text();
const elapsedMs = Date.now() - startedAt;

if (!response.ok) {
  console.error(JSON.stringify({
    ok: false,
    status: response.status,
    elapsedMs,
    error: text.slice(0, 1000),
  }, null, 2));
  process.exit(1);
}

const json = JSON.parse(text);
console.log(JSON.stringify({
  ok: true,
  endpoint: baseUrl,
  model: json.model || model,
  batchSize,
  elapsedMs,
  embeddings: json.data?.length,
  dimensions: json.data?.[0]?.embedding?.length,
  embeddingsPerSecond: Number((batchSize / (elapsedMs / 1000)).toFixed(2)),
}, null, 2));
