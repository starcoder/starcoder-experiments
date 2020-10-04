import re
from typing import Collection, Pattern


def make_pattern(patterns: Collection[str]) -> Pattern:
    """
    Create a single regular expression that tests membership
    for multiple needles in a single haystack.

    Written by Tom Lippincott and Arya D. McCarthy
    """
    escaped_terms = []
    for pattern in patterns:
        kleene_start: bool = pattern.startswith("*")
        kleene_end: bool = pattern.endswith("*")
        escaped_term = (
            (r"\S*" if kleene_start else "")
            + re.escape(pattern.strip("*"))
            + (r"\S*" if kleene_end else "")
        )
        escaped_terms.append(escaped_term)
    combined = re.compile(fr"\b({'|'.join(escaped_terms)})\b")
    return combined


def test_combine_expressions_for_search():
    pattern = make_pattern(["hello", "every*", "stardom", "musing"])
    result = pattern.findall("Hi ! Hello everybody , how are you doing ?".lower())
    assert result == ["hello", "everybody"]


test_combine_expressions_for_search()
