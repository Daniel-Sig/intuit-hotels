"""Shared fixtures: a tiny synthetic dataset in the raw Kaggle schema, so no real data/models are needed."""
import pandas as pd
import pytest

from app import bm25_index
from app.bm25_index import ReviewSearch, build_index
from app.preprocess import clean
from app.service import AssistantService
from app.stats import StatsEngine

PARIS = "10 Rue de Rivoli 75001 Paris France"
AMSTERDAM = "s Gravesandestraat 55 Oost 1092 AA Amsterdam Netherlands"
LONDON = "1 Baker Street W1U 6TY London United Kingdom"
TAGS = "[' Leisure trip ', ' Couple ', ' Standard Room ', ' Stayed 2 nights ']"
NONE_P, NONE_N = "No Positive", "No Negative"


def _row(address, date, hotel, nationality, pos, neg, score, tags=TAGS):
    return dict(
        Hotel_Address=address, Review_Date=date, Average_Score=8.0, Hotel_Name=hotel,
        Reviewer_Nationality=f" {nationality} ", Positive_Review=pos, Negative_Review=neg,
        Reviewer_Score=score, Tags=tags,
    )


RAW_ROWS = [
    _row(PARIS, "6/15/2016", "Hotel Lumiere", "Australia", "The staff were friendly and very helpful", "Breakfast was noisy and the room was small", 8.3),
    _row(PARIS, "3/10/2016", "Hotel Lumiere", "Ireland", "Friendly staff made our stay wonderful", NONE_N, 9.0),
    _row(PARIS, "4/5/2016", "Hotel Lumiere", "Australia", "Lovely location near the river and shops", "Noisy street outside at night time", 7.5),
    _row(AMSTERDAM, "7/1/2016", "Hotel Arena", "Australia", "Great breakfast and a cozy canal view", "Elevator was slow and often broken", 8.0),
    _row(AMSTERDAM, "8/2/2016", "Hotel Arena", "United Kingdom", "Clean room and comfortable bed overall", "Wifi was slow in the evening hours", 7.0,
         "[' Business trip ', ' Solo traveler ', ' Double Room ', ' Stayed 1 night ']"),
    _row(LONDON, "6/20/2016", "The Baker", "Germany", "Quiet rooms despite the central location", "Very noisy construction work next door", 6.0),
    _row(LONDON, "6/2/2016", "The Baker", "Italy", "Staff was friendly and helpful", NONE_N, 9.0),
    _row(LONDON, "7/9/2016", "The Baker", "Canada", "Staff was friendly and helpful", NONE_N, 9.5),
]


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.DataFrame(RAW_ROWS)


@pytest.fixture
def clean_df(raw_df) -> pd.DataFrame:
    return clean(raw_df)


@pytest.fixture
def search(clean_df, tmp_path, monkeypatch) -> ReviewSearch:
    processed = tmp_path / "processed"
    processed.mkdir()
    clean_df.to_parquet(processed / "train.parquet", index=False)
    monkeypatch.setattr(bm25_index, "PROCESSED_DIR", processed)
    build_index(index_dir=tmp_path / "bm25")
    return ReviewSearch(tmp_path / "bm25")


class FakeReranker:
    """Stands in for the cross-encoder: a fixed relevance score for every candidate."""

    def __init__(self, score: float = 5.0):
        self.score = score

    def rerank(self, query, candidates, top_n=10):
        return candidates.assign(rerank_score=self.score).head(top_n).reset_index(drop=True)


@pytest.fixture
def make_service(search, clean_df):
    def _make(rerank_score: float | None = 5.0) -> AssistantService:
        reranker = FakeReranker(rerank_score) if rerank_score is not None else None
        return AssistantService(search, reranker, StatsEngine(clean_df))
    return _make
