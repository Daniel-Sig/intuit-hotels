# Bonus: scaling from 515K to 50M reviews (about 100x)

At 50M reviews (~100M chunks) the single-node BM25 approach stops fitting in memory and per-query work grows. The goal is **sub-second retrieval** with cost that grows with the data, not with traffic.

## Indexing
- Move from files on disk to a **search engine that does filtered lexical + vector search** (OpenSearch/Elasticsearch, or Qdrant plus a BM25 sparse index). Metadata filters are pre-filters, so query cost depends on the matching slice, not the whole corpus.
- Add **embeddings** (dense retrieval). Offline batch GPU jobs make them affordable at this scale. Use a small model and quantised vectors (int8 or PQ) to cut memory 4-8x. Hybrid BM25 + dense with reciprocal-rank fusion fixes the synonym weakness of BM25-only.
- Keep the **cross-encoder rerank on a small candidate set** (top 30-50). It is the dominant per-query retrieval cost and must not grow with the corpus.
- Index incrementally: new reviews are chunked and indexed through a queue (daily or streaming), with periodic segment merges and a full rebuild only when the embedding model changes. Index builds are versioned and swapped atomically.

## Partitioning
- **Partition by city (or region) and time** (for example month). Questions nearly always name a city and often a period, so most queries touch one or a few shards. Keep sentiment as a filter or sub-index rather than a shard once the engine supports it.
- Size shards at a few GB each. Hot shards (big cities, recent months) get more replicas. Cold, old partitions move to cheaper storage.
- Use **global IDF statistics** so scores are comparable across shards, which fixes the approximate cross-shard ranking of the prototype.
- Move aggregates out of in-process pandas into **precomputed rollups** (city x month x nationality x trip type: count, score sum, and mention counts for common terms), refreshed on each ingest. Rare ad-hoc terms fall back to a columnar engine (ClickHouse, or DuckDB over parquet) on the filtered slice.

## Keeping retrieval under a second
- Rough budget: filter and shard routing 10 ms, lexical/dense search 50-100 ms, rerank 200-300 ms (GPU or batched ONNX), network 50 ms. The LLM stays outside this budget and its answer is streamed.
- **Tiered caching:** normalised question + filters to full response (short TTL, invalidated per index version); filter set to candidate list; embedding cache; planner-output cache. CX questions repeat heavily, so hit rates can be high.
- **Autoscaled stateless API pods** in front of the search cluster, with the reranker as its own service so it scales on CPU/GPU load independently. Timeouts and a hedged request to a replica bound p99; if a shard is slow, return partial results with a flag.

## Managing cost
- The two LLM calls dominate marginal cost, so: merge planner and answerer where possible, use a small model, cache by question, skip the LLM when evidence is insufficient (already done), cap prompt size, and alert on daily token spend.
- Search cost is mostly memory: quantised vectors, tiered storage, and routing most traffic to recent, hot partitions. Run index builds and embeddings on spot/preemptible capacity.
- Keep the quality gates from the evaluation plan in CI, so a cost optimisation (smaller model, lower k, more compression) ships only when golden-set metrics hold.
