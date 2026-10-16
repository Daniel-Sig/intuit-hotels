"""Rule-based question -> filters. A placeholder for the LLM-based parser, with the same output shape."""
import calendar
import re
from dataclasses import dataclass, field

from bm25s.stopwords import STOPWORDS_EN

SEASONS = {"spring": [3, 4, 5], "summer": [6, 7, 8], "autumn": [9, 10, 11], "fall": [9, 10, 11], "winter": [12, 1, 2]}
MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
# Cities people ask about that the dataset does not cover, so we can say so instead of guessing.
UNSUPPORTED_CITIES = {
    "rome", "berlin", "madrid", "lisbon", "prague", "budapest", "munich", "dublin", "brussels",
    "copenhagen", "stockholm", "athens", "florence", "venice", "edinburgh", "warsaw",
}


# Words that frame a question without saying what it is about ("what do guests say about ...").
QUESTION_FRAME = {
    "what", "do", "does", "did", "is", "are", "was", "were", "how", "reviewer", "reviewers", "guest", "guests",
    "people", "customer", "customers", "hotel", "hotels", "say", "says", "said", "think", "thinks", "thought",
    "review", "reviews", "tell", "describe", "described", "describes", "mention", "mentions",
    "from", "about", "over", "after", "before", "during", "between", "which", "who", "where", "when", "why",
    "any", "some", "can", "could", "would", "should", "i", "me", "my", "you", "your", "we", "us", "has", "have",
    "had", "been", "am", "time", "visit", "go", "travel", "worth", "recommend", "recommended",
}


@dataclass
class ParsedQuery:
    query: str  # question with recognised filter phrases removed, for BM25
    filters: dict = field(default_factory=dict)
    unsupported: list[str] = field(default_factory=list)
    llm_topic: str | None = None  # set by the LLM planner; the rule-based parser derives the topic instead

    @property
    def topic(self) -> str:
        """`query` minus the question frame and stopwords. Empty means nothing specific was asked."""
        if self.llm_topic is not None:
            return self.llm_topic
        words = re.findall(r"[A-Za-z0-9']+", self.query)
        return " ".join(w for w in words if w.lower() not in QUESTION_FRAME and w.lower() not in STOPWORDS_EN)


def _alternation(terms: list[str]) -> re.Pattern | None:
    terms = sorted({t for t in terms if t}, key=len, reverse=True)
    if not terms:
        return None
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)


class QueryParser:
    def __init__(self, cities: list[str], nationalities: list[str], hotels: list[str]):
        self.cities = {c.lower(): c for c in cities}
        self.nationalities = {n.lower(): n for n in nationalities}
        self.hotels = {h.lower(): h for h in hotels}
        self._hotel_re = _alternation(list(self.hotels))
        self._city_re = _alternation(list(self.cities))
        self._nat_re = _alternation(list(self.nationalities))
        self._unsupported_re = _alternation(list(UNSUPPORTED_CITIES))
        # "may" is also a common verb, so only treat it as a month after a preposition.
        self._month_re = _alternation([m for m in MONTHS if m != "may"])
        self._may_re = re.compile(r"\b(?:in|during|of|every)\s+(may)\b", re.IGNORECASE)
        self._season_re = _alternation(list(SEASONS))

    @staticmethod
    def _take(pattern: re.Pattern | None, text: str) -> tuple[list[str], str]:
        if pattern is None:
            return [], text
        found = [m.lower() for m in pattern.findall(text)]
        return list(dict.fromkeys(found)), pattern.sub(" ", text)

    def parse(self, question: str) -> ParsedQuery:
        text, filters = question, {}

        hotels, text = self._take(self._hotel_re, text)
        if hotels:
            filters["Hotel_Name"] = [self.hotels[h] for h in hotels]
        cities, text = self._take(self._city_re, text)
        if cities:
            filters["city"] = [self.cities[c] for c in cities]
        unsupported, text = self._take(self._unsupported_re, text)
        nats, text = self._take(self._nat_re, text)
        if nats:
            filters["Reviewer_Nationality"] = [self.nationalities[n] for n in nats]

        months: list[int] = []
        found, text = self._take(self._month_re, text)
        months += [MONTHS[m] for m in found]
        if m := self._may_re.search(text):
            months.append(MONTHS["may"])
            text = text.replace(m.group(1), " ")
        seasons, text = self._take(self._season_re, text)
        for s in seasons:
            months += SEASONS[s]
        if months:
            filters["review_month"] = sorted(set(months))

        return ParsedQuery(
            query=re.sub(r"\s+", " ", text).strip() or question,
            filters=filters,
            unsupported=[u.title() for u in unsupported],
        )
