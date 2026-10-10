"""Cleaning and feature extraction for the 515K Hotel Reviews dataset."""
import ast
import re

import pandas as pd

TRIP_TYPES = {"Leisure trip", "Business trip"}
TRAVELER_GROUPS = {
    "Solo traveler", "Couple", "Group", "Travelers with friends",
    "Family with young children", "Family with older children",
}
NIGHTS_RE = re.compile(r"Stayed (\d+) night")


def _parse_tags(raw: str) -> list[str]:
    return [t.strip() for t in ast.literal_eval(raw)]


def _tag_features(raw: str) -> tuple[str | None, str | None, int | None]:
    trip = group = nights = None
    for tag in _parse_tags(raw):
        if tag in TRIP_TYPES:
            trip = tag
        elif tag in TRAVELER_GROUPS:
            group = tag
        elif m := NIGHTS_RE.match(tag):
            nights = int(m.group(1))
    return trip, group, nights


def _country_city(address: str) -> tuple[str, str]:
    if address.endswith("United Kingdom"):
        return "United Kingdom", "London"  # dataset's only UK city; UK addresses end in a postcode
    *rest, country = address.split()
    return country, rest[-1]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate, normalise, and derive filterable columns. Adds a stable `review_id`."""
    df = df.drop_duplicates().reset_index(drop=True)
    df["review_id"] = df.index
    df["Reviewer_Nationality"] = df["Reviewer_Nationality"].str.strip()
    df["Positive_Review"] = df["Positive_Review"].str.strip()
    df["Negative_Review"] = df["Negative_Review"].str.strip()

    dates = pd.to_datetime(df["Review_Date"], format="%m/%d/%Y")
    df["review_date"] = dates
    df["review_year"], df["review_month"] = dates.dt.year, dates.dt.month

    cc = [_country_city(a) for a in df["Hotel_Address"].unique()]
    lookup = dict(zip(df["Hotel_Address"].unique(), cc))
    df["country"] = df["Hotel_Address"].map(lambda a: lookup[a][0])
    df["city"] = df["Hotel_Address"].map(lambda a: lookup[a][1])

    tag_map = {t: _tag_features(t) for t in df["Tags"].unique()}
    feats = df["Tags"].map(tag_map)
    df["trip_type"] = feats.str[0]
    df["traveler_group"] = feats.str[1]
    df["nights"] = feats.str[2]
    return df
