import pytest

from app import stats as stats_module
from app.query_parser import QueryParser
from app.router import route
from app.stats import Operation, StatsEngine, mention_pattern


@pytest.fixture
def parser():
    return QueryParser(["Amsterdam", "London", "Paris"], ["Australia", "United Kingdom", "Ireland"], ["Hotel Arena"])


def test_parser_extracts_filters(parser):
    p = parser.parse("What do reviewers from Australia say about hotels in Amsterdam?")
    assert p.filters == {"city": ["Amsterdam"], "Reviewer_Nationality": ["Australia"]}
    assert p.topic == ""  # open-ended: nothing specific asked


def test_parser_months_and_seasons(parser):
    assert parser.parse("is it noisy in June").filters["review_month"] == [6]
    assert parser.parse("noise in spring").filters["review_month"] == [3, 4, 5]
    assert "review_month" not in parser.parse("rooms that may be small").filters  # "may" the verb


def test_parser_hotel_and_unsupported_city(parser):
    assert parser.parse("is Hotel Arena clean").filters == {"Hotel_Name": ["Hotel Arena"]}
    assert parser.parse("is spring a good time to visit Rome").unsupported == ["Rome"]


def test_topic_keeps_the_subject(parser):
    assert parser.parse("what do guests think of the pool").topic == "pool"


@pytest.mark.parametrize("question,name,group_by", [
    ("What's the average score for Amsterdam hotels?", "mean_score", None),
    ("Which city has the highest rated hotels?", "mean_score", "city"),
    ("How many reviews are from Australia?", "count", None),
    ("How many Paris reviews mention breakfast?", "mention_count", None),
])
def test_router_picks_operation(question, name, group_by):
    op = route(question)
    assert (op.name, op.group_by) == (name, group_by)


def test_router_returns_none_for_plain_questions():
    assert route("Is staff in Paris friendly?") is None


def test_mention_pattern_uses_stems():
    pattern = mention_pattern("noise")
    assert all(pattern.search(w) for w in ("noise", "Noises", "noisy"))
    assert not pattern.search("nothing")


def test_stats_overall_and_filtered(clean_df):
    engine = StatsEngine(clean_df)
    everything = engine.compute({})
    assert everything.reviews_matched == len(clean_df)
    assert everything.avg_reviewer_score == round(clean_df["Reviewer_Score"].mean(), 2)

    paris = engine.compute({"city": ["Paris"]})
    assert paris.reviews_matched == 3
    assert paris.avg_reviewer_score == round((8.3 + 9.0 + 7.5) / 3, 2)


def test_stats_operations(clean_df, monkeypatch):
    engine = StatsEngine(clean_df)
    mentions = engine.compute({"city": ["Paris"]}, Operation(name="mention_count", term="noise")).operation.rows[0]
    assert (mentions["mentions"], mentions["reviews"]) == (2, 3)  # "noisy" counts as "noise"

    monkeypatch.setattr(stats_module, "MIN_GROUP_SIZE", 1)
    by_city = engine.compute({}, Operation(name="mean_score", group_by="city")).operation.rows
    assert [r["group"] for r in by_city][0] == "London" or by_city[0]["value"] >= by_city[-1]["value"]
    assert {r["group"] for r in by_city} == {"Paris", "Amsterdam", "London"}


def test_mention_count_requires_a_term():
    with pytest.raises(ValueError):
        Operation(name="mention_count")


def test_mention_pattern_matches_noisy_and_noise():
    pat = mention_pattern("noisy")
    assert pat.search("a bit noisy at night") and pat.search("street noise") and pat.search("noisier rooms")


def test_topic_breakdown_counts_mentions_on_each_side(clean_df):
    engine = StatsEngine(clean_df)
    tb = engine.compute({"city": ["Paris"]}, None, topic="friendly staff").topic_breakdown
    assert (tb.reviews_in_scope, tb.reviews_mentioning) == (3, 2)
    assert (tb.mentions_in_positive_comment, tb.mentions_in_negative_comment) == (2, 0)
    assert tb.positive_share_of_mentions == 1.0
    none = engine.compute({"city": ["Paris"]}, None, topic="zebra").topic_breakdown
    assert none.reviews_mentioning == 0 and none.positive_share_of_mentions is None


def test_topic_breakdown_is_skipped_when_an_operation_runs_or_no_topic(clean_df):
    engine = StatsEngine(clean_df)
    assert engine.compute({}, Operation(name="count"), topic="staff").topic_breakdown is None
    assert engine.compute({}, None, topic="").topic_breakdown is None


def test_mention_pattern_keeps_a_real_final_i():
    pat = mention_pattern("wifi")
    assert pat.search("free wifi in the lobby") and not pat.search("my wife loved it")


def test_llm_stats_json_hides_comment_rates_and_has_the_mention_rate(clean_df):
    stats = StatsEngine(clean_df).compute({"city": ["Paris"]}, None, topic="friendly staff")
    assert "with_positive_comment_rate" in stats.model_dump_json()
    shown = stats.for_llm()
    assert "comment_rate" not in shown and '"mention_rate":0.667' in shown


def test_summaries_state_the_numbers_in_one_sentence(clean_df):
    engine = StatsEngine(clean_df)
    tb = engine.compute({"city": ["Paris"]}, None, topic="friendly staff").topic_breakdown
    assert tb.summary.startswith("2 of 3 reviews (66.7%) mention 'friendly staff'")
    assert "2 (100.0%) are in the guest's POSITIVE comment" in tb.summary and tb.negative_share_of_mentions == 0.0
    op = engine.compute({}, Operation(name="count", group_by="city")).operation
    assert op.summary.startswith("Review count by city, most first: ")
    assert engine.compute({}, Operation(name="mean_score")).operation.summary.startswith("Average reviewer score: ")


def test_counts_are_stated_as_minimums(clean_df):
    engine = StatsEngine(clean_df)
    one = engine.compute({"city": ["Paris"]}, Operation(name="count")).operation.summary
    assert one.startswith("Review count: at least 3 reviews") and "only the indexed period" in one
    assert "minimum counts" in engine.compute({}, Operation(name="count", group_by="city")).operation.summary
    assert "at least" not in engine.compute({}, Operation(name="mean_score")).operation.summary


def test_any_of_counts_reviews_with_either_word(clean_df):
    engine = StatsEngine(clean_df)
    both = engine.compute({}, None, topic="friendly noisy").topic_breakdown
    either = engine.compute({}, None, topic="friendly noisy", any_of=True).topic_breakdown
    assert either.term == "friendly or noisy" and either.reviews_mentioning > both.reviews_mentioning
