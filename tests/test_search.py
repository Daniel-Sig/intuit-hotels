import pandas as pd
from app.bm25_index import MIN_WORDS, to_chunks


def test_chunks_skip_placeholder_and_short_text(clean_df):
    chunks = to_chunks(clean_df)
    assert (chunks["text"].str.split().str.len() >= MIN_WORDS).all()
    assert not chunks["text"].isin(["No Negative", "No Positive"]).any()
    assert chunks["chunk_id"].is_unique


def test_city_filter_only_returns_that_city(search):
    hits = search.search("breakfast", city="Paris")
    assert not hits.empty and set(hits["city"]) == {"Paris"}


def test_multi_value_filters(search):
    hits = search.search("noisy", review_month=[6, 7], Reviewer_Nationality={"Australia", "Germany"})
    assert not hits.empty
    assert set(hits["review_month"]) <= {6, 7}
    assert set(hits["Reviewer_Nationality"]) <= {"Australia", "Germany"}


def test_sentiment_filter(search):
    hits = search.search("noisy", sentiment="negative")
    assert set(hits["sentiment"]) == {"negative"}


def test_dedupe_collapses_identical_text_but_keeps_it_filterable(search):
    text = "Staff was friendly and helpful"
    assert (search.search("staff friendly helpful", dedupe=True)["text"] == text).sum() == 1
    assert (search.search("staff friendly helpful", dedupe=False)["text"] == text).sum() == 2
    # Dedupe happens after filtering, so a nationality filter still finds the review it belongs to.
    assert search.search("staff friendly helpful", Reviewer_Nationality="Canada")["Reviewer_Nationality"].tolist() == ["Canada"]


def test_no_match_returns_empty(search):
    assert search.search("zzzzqqq").empty
    assert search.search("breakfast", city="Amsterdam", Reviewer_Nationality="Germany").empty


def _ranked(pos, neg):
    rows = [dict(sentiment="negative", rerank_score=10 - i) for i in range(neg)]
    rows += [dict(sentiment="positive", rerank_score=1 - i * 0.1) for i in range(pos)]
    return pd.DataFrame(rows).sort_values("rerank_score", ascending=False).reset_index(drop=True)


def test_pick_balances_polarity_when_no_side_requested():
    from app.service import AssistantService
    top = AssistantService._pick(_ranked(pos=4, neg=8), k=6, balanced=True)
    assert top["sentiment"].value_counts().to_dict() == {"negative": 3, "positive": 3}
    assert list(top["rerank_score"]) == sorted(top["rerank_score"], reverse=True)


def test_pick_tops_up_from_the_other_side_and_respects_a_requested_side():
    from app.service import AssistantService
    top = AssistantService._pick(_ranked(pos=1, neg=8), k=6, balanced=True)
    assert len(top) == 6 and (top["sentiment"] == "positive").sum() == 1
    assert len(AssistantService._pick(_ranked(pos=4, neg=8), k=6, balanced=False)) == 6
