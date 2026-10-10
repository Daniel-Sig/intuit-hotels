import pandas as pd

from app.data_prep import split_dataset
from app.preprocess import clean


def test_clean_derives_filter_columns(clean_df):
    assert set(clean_df["city"]) == {"Paris", "Amsterdam", "London"}
    assert set(clean_df["country"]) == {"France", "Netherlands", "United Kingdom"}
    first = clean_df.iloc[0]
    assert (first.review_month, first.review_year) == (6, 2016)
    assert first.Reviewer_Nationality == "Australia"  # whitespace stripped
    assert (first.trip_type, first.traveler_group, first.nights) == ("Leisure trip", "Couple", 2)
    business = clean_df[clean_df.Hotel_Name == "Hotel Arena"].iloc[1]
    assert (business.trip_type, business.traveler_group, business.nights) == ("Business trip", "Solo traveler", 1)


def test_clean_drops_exact_duplicates_and_ids_are_stable(raw_df):
    doubled = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)
    out = clean(doubled)
    assert len(out) == len(raw_df)
    assert out["review_id"].tolist() == list(range(len(raw_df)))
    assert clean(doubled).equals(out)  # deterministic


def test_split_is_chronological_and_never_divides_a_day():
    days = pd.date_range("2016-01-01", periods=100)
    df = pd.DataFrame({"review_date": days.repeat(10), "review_id": range(1000)})
    train, val, test = split_dataset(df)

    assert train.review_date.max() < val.review_date.min() <= val.review_date.max() < test.review_date.min()
    assert not set(train.review_date) & set(val.review_date)
    assert not set(val.review_date) & set(test.review_date)
    assert len(train) + len(val) + len(test) == len(df)
    assert abs(len(train) / len(df) - 0.70) < 0.02
    assert abs(len(val) / len(df) - 0.15) < 0.02
