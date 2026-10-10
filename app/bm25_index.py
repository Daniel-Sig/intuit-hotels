"""BM25 retrieval over review chunks, with exact-match metadata filters.

Each review yields up to two chunks (its positive and its negative text), so sentiment is a
filter. One index is built per (city, sentiment), which makes the city filter free: it just
selects which index to search. Other filters are applied as a mask during scoring.
"""
import logging
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd
import Stemmer

from app.data_prep import PROCESSED_DIR

log = logging.getLogger(__name__)

INDEX_DIR = PROCESSED_DIR.parent / "bm25"
MIN_WORDS = 5  # retrieval floor on real word count: drops "Nothing", "Great location", "N A" etc.
DEDUPE_OVERFETCH = 3  # fetch extra per shard so dedupe doesn't leave fewer than k results
STEMMER = Stemmer.Stemmer("english")

META_COLS = [
    "review_id", "Hotel_Name", "city", "country", "Reviewer_Nationality", "review_year",
    "review_month", "trip_type", "traveler_group", "Reviewer_Score", "Average_Score",
]


def to_chunks(df: pd.DataFrame) -> pd.DataFrame:
    """One row per review -> one row per (review, sentiment) chunk with enough content."""
    parts = []
    for sentiment, col in (("positive", "Positive_Review"), ("negative", "Negative_Review")):
        part = df[META_COLS + [col]].rename(columns={col: "text"})
        # Counted from the text: the dataset's own *_Word_Counts columns don't match it.
        part = part[part["text"].str.split().str.len() >= MIN_WORDS].copy()
        part["sentiment"] = sentiment
        parts.append(part)
    chunks = pd.concat(parts, ignore_index=True)
    chunks["chunk_id"] = chunks["review_id"].astype(str) + "-" + chunks["sentiment"].str[0]
    return chunks


def _doc_tokens(texts: list[str]):
    return bm25s.tokenize(texts, stopwords="en", stemmer=STEMMER, show_progress=False)


def _query_tokens(query: str) -> list[str]:
    return bm25s.tokenize(query, stopwords="en", stemmer=STEMMER, return_ids=False, show_progress=False)[0]


def _shard_dir(index_dir: Path, city: str, sentiment: str) -> Path:
    return index_dir / f"{city}_{sentiment}"


def build_index(split: str = "train", limit: int | None = None, index_dir: Path = INDEX_DIR) -> int:
    """Build one BM25 index per (city, sentiment) from a split. Returns chunks indexed."""
    df = pd.read_parquet(PROCESSED_DIR / f"{split}.parquet")
    if limit:
        df = df.head(limit)
    chunks = to_chunks(df)
    log.info("%s: %d reviews -> %d chunks", split, len(df), len(chunks))

    for (city, sentiment), shard in chunks.groupby(["city", "sentiment"]):
        shard = shard.reset_index(drop=True)
        retriever = bm25s.BM25()
        retriever.index(_doc_tokens(shard["text"].tolist()), show_progress=False)
        path = _shard_dir(index_dir, city, sentiment)
        retriever.save(path)
        shard.to_parquet(path / "chunks.parquet", index=False)
        log.info("indexed %s/%s: %d chunks", city, sentiment, len(shard))
    return len(chunks)


class ReviewSearch:
    """Loads the shard indexes and answers filtered BM25 queries."""

    def __init__(self, index_dir: Path = INDEX_DIR):
        self.shards: dict[tuple[str, str], tuple[bm25s.BM25, pd.DataFrame]] = {}
        for path in sorted(index_dir.iterdir()):
            city, sentiment = path.name.rsplit("_", 1)
            self.shards[(city, sentiment)] = (bm25s.BM25.load(path, mmap=True), pd.read_parquet(path / "chunks.parquet"))
        if not self.shards:
            raise FileNotFoundError(f"no BM25 shards found in {index_dir}")

    def values(self, column: str) -> list[str]:
        """Distinct values of a metadata column across all shards (used to validate/parse filters)."""
        return sorted({v for _, chunks in self.shards.values() for v in chunks[column].dropna().unique()})

    def search(
        self, query: str, k: int = 10, city: str | list[str] | None = None, sentiment: str | None = None,
        dedupe: bool = True, **filters
    ) -> pd.DataFrame:
        """Top-k chunks for `query`. `filters` are exact-match column=value conditions;
        a list/tuple/set value means "any of these" (e.g. review_month=[3, 4, 5]).

        Scores come from separate shards, so cross-shard ranking is approximate (each shard
        has its own IDF statistics). Fine for a candidate list; document it as a trade-off.

        With `dedupe`, chunks whose text is identical (ignoring case/whitespace) collapse to the
        best-scoring one. This happens after filtering, never at index time, so identical text
        from different nationalities/months stays filterable.
        """
        tokens = _query_tokens(query)
        if not tokens:
            return pd.DataFrame()
        hits = []
        for (shard_city, shard_sent), (retriever, chunks) in self.shards.items():
            if (city and shard_city not in ([city] if isinstance(city, str) else city)) or (
                sentiment and shard_sent != sentiment
            ):
                continue
            mask = np.ones(len(chunks), dtype=bool)
            for col, value in filters.items():
                match = chunks[col].isin(value) if isinstance(value, (list, tuple, set)) else chunks[col] == value
                mask &= match.to_numpy()
            if not mask.any():
                continue
            fetch = k * DEDUPE_OVERFETCH if dedupe else k
            docs, scores = retriever.retrieve(
                [tokens], k=min(fetch, int(mask.sum())), weight_mask=mask.astype(np.float32), show_progress=False
            )
            top = chunks.iloc[docs[0]].copy()
            top["score"] = scores[0]
            hits.append(top[top["score"] > 0])
        if not hits:
            return pd.DataFrame()
        result = pd.concat(hits).sort_values("score", ascending=False)
        if dedupe:
            result = result[~result["text"].str.lower().str.strip().duplicated()]
        return result.head(k).reset_index(drop=True)
