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
