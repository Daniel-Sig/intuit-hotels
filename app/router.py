"""Rule-based question -> Operation. The LLM router will replace this, returning the same Operation."""
import re

from app.stats import Operation

_MENTION = re.compile(
    r"\bhow many\b.*?\b(?:mention(?:s|ed)?|(?:talk|complain|write|say)(?:s|ed)? about)\s+"
    r"(?:the\s+|a\s+|an\s+)?([a-z][a-z' -]{1,30}?)(?=\s+(?:in|at|for|from|during|of|by|about)\b|[?.!,]|$)",
    re.IGNORECASE,
)
_COUNT = re.compile(r"\bhow many\b|\bnumber of\b|\bcount\b", re.IGNORECASE)
_SCORE = re.compile(r"\b(average|avg|mean|rating|ratings|rated|score|scores|best|worst|highest|lowest)\b", re.IGNORECASE)

_GROUPS = [
    ("city", re.compile(r"\b(which|what|each|every|per|by|across|compare|compared|between)\b.*\bcit(?:y|ies)\b|\bcit(?:y|ies)\b.*\b(compare|compared|best|worst|highest|lowest)\b", re.IGNORECASE)),
    ("nationality", re.compile(r"\b(which|what|each|per|by|across)\b.*\bnationalit(?:y|ies)\b|\bwhich countr(?:y|ies) (?:of|do|are) (?:the )?(?:reviewers|guests)", re.IGNORECASE)),
    ("month", re.compile(r"\b(which|what|each|per|by|across)\b.*\bmonths?\b", re.IGNORECASE)),
    ("trip_type", re.compile(r"\btrip type\b|\bbusiness (?:vs|versus|or) leisure\b|\bleisure (?:vs|versus|or) business\b", re.IGNORECASE)),
    ("traveler_group", re.compile(r"\btraveler (?:group|type)s?\b|\bfamilies (?:vs|versus)\b", re.IGNORECASE)),
]


def route(question: str) -> Operation | None:
    """Pick one operation from the menu, or None if the question doesn't ask for a number."""
    group_by = next((name for name, pat in _GROUPS if pat.search(question)), None)

    if m := _MENTION.search(question):
        return Operation(name="mention_count", term=m.group(1).strip(), group_by=group_by)
    if _COUNT.search(question):
        return Operation(name="count", group_by=group_by)
    if _SCORE.search(question) or group_by:
        return Operation(name="mean_score", group_by=group_by)
    return None
