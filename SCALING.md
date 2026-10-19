# Bonus: scaling from 515K to 50M reviews (about 100x)

At 50M reviews (~100M chunks) the single-node BM25 approach stops fitting in memory and per-query work grows. The goal is **sub-second retrieval** with cost that grows with the data, not with traffic. The design stays retrieval-augmented generation; only the retrieval store changes.

## Indexing
- **Move to OpenSearch (or Elasticsearch)**, which does filtered text search, vector search, aggregations and sharding in one system. Exact filters ("Australian guests, London, June") are applied natively, and its aggregations compute the statistics over all matching reviews. A vector database alone treats filters as an add-on and returns only the nearest few.
- **Add embeddings** for meaning-based matching, built in offline GPU batches with a small model and quantised vectors (4-8x less memory). BM25 stays for hotel names, exact phrases and spellings like "Wi-Fi". Results from both are merged with reciprocal-rank fusion, which fixes BM25's blindness to synonyms.
- **Keep the cross-encoder rerank on a small candidate set** (top 30-50), so its cost does not grow with the corpus.
- **Index incrementally** through a queue (daily or streaming). Full rebuilds only when the embedding model changes; builds are versioned and swapped atomically.

## Partitioning
- **Partition by city and time.** Questions nearly always name a city, so most queries touch one or a few shards. Size shards at a few GB; hot ones get more replicas and old ones move to cheaper storage.
- **One shared index for small cities.** Cities with little data share a catch-all index (the `city` filter still keeps answers correct), which avoids thousands of tiny shards. A daily job promotes a city to its own index once it passes a size threshold: reindex, check against the golden set, then switch an alias. Cities are never demoted automatically.
- **Global IDF statistics** make scores comparable across shards.
- **Statistics move out of pandas** into precomputed rollups (city x month x nationality x trip type) refreshed on each ingest. Rare ad-hoc terms fall back to a columnar engine (ClickHouse, or DuckDB over parquet).

## Keeping retrieval under a second
- **Budget:** filter and routing 10 ms, search 50-100 ms, rerank 200-300 ms, network 50 ms. The LLM sits outside it and its answer is streamed.
- **Caching:** question plus filters to response (short TTL), filter set to candidates, and the planner output. Support questions repeat a lot.
- **Stateless API pods** behind an autoscaler, with the reranker as its own service. Timeouts and hedged requests bound the slowest responses.

## Cost
- The LLM calls dominate marginal cost. Merge the planner and answer calls, use a small model, cache by question, skip the LLM when evidence is insufficient (already done), cap prompt size and alert on daily token spend.
- Search cost is mostly memory: quantised vectors, tiered storage, spot capacity for builds.
- Keep the golden-set quality gates in CI so a cost saving (smaller model, lower k) ships only if quality holds.
