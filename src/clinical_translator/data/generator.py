"""Deterministic synthetic CURB-65 vignette generation."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from clinical_translator.contracts.validation import FACTS, load, reference

TEMPLATES = (
    "Adult age: {age}. Mental state: {mental}. Urea: {urea}. Respiratory rate: {respiratory}. Systolic pressure: {systolic}. Diastolic pressure: {diastolic}.",
    "The adult is {age} old, is {mental}, has urea {urea}, breathes at {respiratory}, and has blood pressure {systolic} systolic and {diastolic} diastolic.",
    "Recorded findings for an adult: age {age}; mental status {mental}; urea {urea}; respirations {respiratory}; systolic pressure {systolic}; diastolic pressure {diastolic}.",
    "Adult observations show age {age}, mental state {mental}, serum urea {urea}, respiratory rate {respiratory}, systolic BP {systolic}, and diastolic BP {diastolic}.",
    "For this adult, age is {age}. Mental status is {mental}. Urea is {urea}. Breathing rate is {respiratory}. Blood pressure is {systolic} systolic and {diastolic} diastolic.",
    "Adult clinical values — age {age}; cognition {mental}; blood urea {urea}; breaths per minute {respiratory}; systolic {systolic}; diastolic {diastolic}.",
    "Review of an adult records {age} for age, {mental} for mental state, {urea} for urea, {respiratory} for respiratory rate, {systolic} for systolic pressure, and {diastolic} for diastolic pressure.",
    "Adult summary: age={age}; mental_state={mental}; urea={urea}; respiratory_rate={respiratory}; systolic_bp={systolic}; diastolic_bp={diastolic}.",
)
REQUIRED_CONCEPTS = (
    "mental_state",
    "urea_mmol_l",
    "respiratory_rate_per_minute",
    "systolic_bp_mmhg",
    "diastolic_bp_mmhg",
    "age_years",
)


def _values(facts: dict[str, bool], template: int) -> dict[str, Any]:
    normal_bp, low_bp = (
        ((90, 61), (89, 61)),
        ((91, 61), (91, 60)),
        ((90, 61), (90, 59)),
    )[template % 3]
    return {
        "mental_state": "new confusion present"
        if facts["confusion"]
        else "alert with no confusion",
        "urea_mmol_l": 7.1
        if facts["elevated_urea"]
        else (6.9, 7.0)[template % 2],
        "respiratory_rate_per_minute": (30, 31)[template % 2]
        if facts["high_respiratory_rate"]
        else 29,
        "systolic_bp_mmhg": (low_bp if facts["low_blood_pressure"] else normal_bp)[0],
        "diastolic_bp_mmhg": (low_bp if facts["low_blood_pressure"] else normal_bp)[1],
        "age_years": (65, 66)[template % 2] if facts["age_at_least_65"] else 64,
    }


def _render(template: int, values: dict[str, Any]) -> str:
    def show(name: str, unit: str) -> str:
        value = values[name]
        return "not reported" if value is None else f"{value:g} {unit}"

    mental = values["mental_state"]
    return TEMPLATES[template].format(
        age=show("age_years", "years"),
        mental="not reported" if mental is None else mental,
        urea=show("urea_mmol_l", "mmol/L"),
        respiratory=show("respiratory_rate_per_minute", "breaths/min"),
        systolic=show("systolic_bp_mmhg", "mmHg"),
        diastolic=show("diastolic_bp_mmhg", "mmHg"),
    )


def generate(contract: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contract_ref = reference(contract)
    grammar = contract["input_domain"]["grammar_version"]
    seed = contract["reproducibility"]["seeds"]["python"]
    records: list[dict[str, Any]] = []

    for bits in itertools.product((False, True), repeat=len(FACTS)):
        facts = dict(zip(FACTS, bits, strict=True))
        combination = "".join("1" if value else "0" for value in bits)
        for template in range(len(TEMPLATES)):
            values = _values(facts, template)
            records.append(
                {
                    "id": f"complete-t{template + 1:02d}-{combination}",
                    "kind": "complete",
                    "prompt": _render(template, values),
                    "facts": facts,
                    "values": values,
                    "provenance": {
                        "contract_ref": contract_ref,
                        "generator_version": "v1",
                        "grammar_version": grammar,
                        "seed": seed,
                        "synthetic": True,
                        "template_id": f"t{template + 1:02d}",
                        "combination": combination,
                    },
                }
            )

    base_facts = dict.fromkeys(FACTS, False)
    for concept in REQUIRED_CONCEPTS:
        values = _values(base_facts, 0)
        values[concept] = None
        facts: dict[str, bool | None] = dict(base_facts)
        fact = {
            "mental_state": "confusion",
            "urea_mmol_l": "elevated_urea",
            "respiratory_rate_per_minute": "high_respiratory_rate",
            "systolic_bp_mmhg": "low_blood_pressure",
            "diastolic_bp_mmhg": "low_blood_pressure",
            "age_years": "age_at_least_65",
        }[concept]
        facts[fact] = None
        records.append(
            {
                "id": f"incomplete-{concept}",
                "kind": "incomplete",
                "prompt": _render(0, values),
                "facts": facts,
                "values": values,
                "provenance": {
                    "contract_ref": contract_ref,
                    "generator_version": "v1",
                    "grammar_version": grammar,
                    "seed": seed,
                    "synthetic": True,
                    "template_id": "t01",
                    "missing_concept": concept,
                },
            }
        )

    pairs = []
    for template in range(len(TEMPLATES)):
        for bits in itertools.product((False, True), repeat=len(FACTS)):
            for index, criterion in enumerate(FACTS):
                if bits[index]:
                    continue
                changed = bits[:index] + (True,) + bits[index + 1 :]
                source = "".join("1" if value else "0" for value in bits)
                target = "".join("1" if value else "0" for value in changed)
                pairs.append(
                    {
                        "id": f"pair-t{template + 1:02d}-{source}-{criterion}",
                        "criterion": criterion,
                        "source_id": f"complete-t{template + 1:02d}-{source}",
                        "target_id": f"complete-t{template + 1:02d}-{target}",
                        "contract_ref": contract_ref,
                    }
                )
    return records, pairs


def write_jsonl(items: list[dict[str, Any]], path: Path) -> None:
    path.write_text(
        "".join(
            json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
            for item in items
        ),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the CURB-65 prompt set")
    parser.add_argument(
        "--contract",
        default="configs/contracts/curb65-medgemma-1.5-v1.json",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/vignettes"))
    args = parser.parse_args()
    records, pairs = generate(load(args.contract))
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, args.output / "vignettes.jsonl")
    write_jsonl(pairs, args.output / "counterfactual_pairs.jsonl")
    print(f"{len(records)} vignettes, {len(pairs)} counterfactual pairs")
