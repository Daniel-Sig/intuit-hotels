"""Cross-encoder reranking of BM25 candidates."""
import numpy as np
import pandas as pd
from fastembed.rerank.cross_encoder import TextCrossEncoder

RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"  # 80 MB, CPU-friendly
MAX_WORDS = 80  # cost grows with the longest text in the batch, so score only the opening of each chunk


class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        self.model = TextCrossEncoder(model_name)

    def rerank(self, query: str, candidates: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Re-score `candidates` (needs a `text` column) against `query`; keep the best `top_n`.

        BM25's score is kept as `score`; the cross-encoder's is added as `rerank_score`.
        Only the first MAX_WORDS words of each chunk are scored; the full text is returned.
        """
        if candidates.empty:
            return candidates
        texts = [" ".join(t.split()[:MAX_WORDS]) for t in candidates["text"]]
        scores = np.fromiter(self.model.rerank(query, texts), dtype=np.float32)
        out = candidates.assign(rerank_score=scores)
        return out.nlargest(top_n, "rerank_score").reset_index(drop=True)
