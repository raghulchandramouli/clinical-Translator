"""Strict prompt and output protocol for CURB-65 fact translation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from clinical_translator.assurance.oracle import score
from clinical_translator.contracts.validation import FACTS, reference

PROMPT_VERSION = "curb65-five-facts-v1"
INSTRUCTION = """Return exactly one JSON object with these five Boolean fields:
confusion, elevated_urea, high_respiratory_rate, low_blood_pressure, age_at_least_65.

Rules:
- confusion is true only when new confusion is explicitly present.
- elevated_urea is true only when urea is greater than 7 mmol/L.
- high_respiratory_rate is true when respiratory rate is at least 30 breaths/min.
- low_blood_pressure is true when systolic pressure is below 90 mmHg or diastolic pressure is at most 60 mmHg.
- age_at_least_65 is true when age is at least 65 years.

Boundary checks: urea 6.9 or 7.0 is false and 7.1 is true; respiratory
rate 29 is false and 30 is true; blood pressure 89/61 is true, 90/61 is
false, and 91/60 is true; age 64 is false and 65 is true.

Use JSON booleans true or false. Do not add fields, markdown, diagnosis, score,
explanation, or treatment advice.

Vignette:
"""


def prompt(vignette: str) -> str:
    return INSTRUCTION + vignette


def parse(raw: str) -> dict[str, bool]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate output field: {key}")
            result[key] = value
        return result

    try:
        facts = json.loads(raw, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as error:
        raise ValueError("model output must be exactly one JSON object") from error
    if not isinstance(facts, dict):
        raise ValueError("model output must be a JSON object")
    score(facts)
    return {name: facts[name] for name in FACTS}


def result(
    *,
    contract: dict[str, Any],
    source: dict[str, Any],
    role: str,
    facts: dict[str, bool],
) -> dict[str, Any]:
    model = contract["models"][role]
    return {
        "contract_ref": reference(contract),
        "prompt": {
            "id": source["id"],
            "sha256": hashlib.sha256(source["prompt"].encode()).hexdigest(),
            "protocol": PROMPT_VERSION,
        },
        "model": {
            "role": role,
            "repo_id": model["repo_id"],
            "revision": model["revision"],
        },
        "inference": contract["reproducibility"],
        "facts": facts,
    }
