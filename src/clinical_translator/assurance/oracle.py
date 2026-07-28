"""Executable CURB-65 oracle over the frozen five-fact schema."""

from __future__ import annotations

import itertools
import json
import sys
from collections.abc import Iterator, Mapping

from clinical_translator.contracts.validation import FACTS


def _values(facts: Mapping[str, bool]) -> tuple[bool, ...]:
    if set(facts) != set(FACTS):
        raise ValueError("facts must contain exactly the five CURB-65 fields")
    values = tuple(facts[fact] for fact in FACTS)
    if any(type(value) is not bool for value in values):
        raise TypeError("all CURB-65 facts must be Boolean")
    return values


def score(facts: Mapping[str, bool]) -> int:
    return sum(_values(facts))


def incomplete_score(facts: Mapping[str, bool]) -> int:
    return sum(_values(facts)[1:])


def combinations() -> Iterator[dict[str, bool]]:
    for values in itertools.product((False, True), repeat=len(FACTS)):
        yield dict(zip(FACTS, values, strict=True))


if __name__ == "__main__":
    print(score(json.loads(sys.stdin.read())))
