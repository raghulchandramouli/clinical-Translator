"""Finite behavioural evaluation of validated model predictions."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from clinical_translator.contracts.validation import FACTS, reference
from clinical_translator.data.generator import write_jsonl

BOUNDARIES = {
    "confusion": lambda record: record["values"]["mental_state"],
    "elevated_urea": lambda record: record["values"]["urea_mmol_l"],
    "high_respiratory_rate": lambda record: record["values"][
        "respiratory_rate_per_minute"
    ],
    "low_blood_pressure": lambda record: (
        f"{record['values']['systolic_bp_mmhg']}/"
        f"{record['values']['diastolic_bp_mmhg']}"
    ),
    "age_at_least_65": lambda record: record["values"]["age_years"],
}


def _rate(correct: int, total: int) -> float | None:
    return correct / total if total else None


def _summary(
    records: list[dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    validated = [
        record
        for record in records
        if outcomes[record["id"]]["status"] == "validated"
    ]
    parser_failures = sum(
        outcomes[record["id"]]["status"] == "parser_failure" for record in records
    )
    runner_failures = sum(
        outcomes[record["id"]]["status"] == "runner_failure" for record in records
    )
    exact = sum(
        outcomes[record["id"]]["prediction"] == record["facts"]
        for record in validated
    )
    result = {
        "records": len(records),
        "validated": len(validated),
        "format_failures": parser_failures,
        "parser_failures": parser_failures,
        "runner_failures": runner_failures,
        "exact_record": {
            "correct": exact,
            "accuracy": _rate(exact, len(records)),
            "accuracy_given_valid": _rate(exact, len(validated)),
        },
        "criteria": {},
    }
    for fact in FACTS:
        correct = sum(
            outcomes[record["id"]]["prediction"][fact] == record["facts"][fact]
            for record in validated
        )
        result["criteria"][fact] = {
            "correct": correct,
            "accuracy": _rate(correct, len(records)),
            "accuracy_given_valid": _rate(correct, len(validated)),
        }
    return result


def _boundary_summary(
    fact: str,
    records: list[dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    validated = [
        record
        for record in records
        if outcomes[record["id"]]["status"] == "validated"
    ]
    parser_failures = sum(
        outcomes[record["id"]]["status"] == "parser_failure" for record in records
    )
    runner_failures = sum(
        outcomes[record["id"]]["status"] == "runner_failure" for record in records
    )
    correct = sum(
        outcomes[record["id"]]["prediction"][fact] == record["facts"][fact]
        for record in validated
    )
    return {
        "records": len(records),
        "validated": len(validated),
        "format_failures": parser_failures,
        "parser_failures": parser_failures,
        "runner_failures": runner_failures,
        "correct": correct,
        "accuracy": _rate(correct, len(records)),
        "accuracy_given_valid": _rate(correct, len(validated)),
    }


def evaluate(
    contract: dict[str, Any],
    records: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    roles = tuple(contract["models"])
    expected = {(role, record["id"]) for role in roles for record in records}
    actual = [(item["model"], item["prompt_id"]) for item in predictions]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("predictions must contain every model/prompt exactly once")

    by_role = {
        role: {
            item["prompt_id"]: item
            for item in predictions
            if item["model"] == role
        }
        for role in roles
    }
    complete = [record for record in records if record["kind"] == "complete"]
    incomplete = [record for record in records if record["kind"] == "incomplete"]
    metrics: dict[str, Any] = {
        "contract_ref": reference(contract),
        "scope": {
            "generated_prompts": len(records),
            "complete_prompts": len(complete),
            "incomplete_prompts": len(incomplete),
            "models": len(roles),
            "model_prompt_outcomes": len(predictions),
            "counterfactual_pairs_per_model": len(pairs),
        },
        "models": {},
        "by_template": {},
        "by_combination": {},
        "by_boundary": {},
        "counterfactuals": {},
    }
    failures: list[dict[str, Any]] = []
    counterfactuals: list[dict[str, Any]] = []

    for role in roles:
        outcomes = by_role[role]
        baseline = _summary(complete, outcomes)
        baseline["incomplete"] = {
            "records": len(incomplete),
            "accepted_guesses": sum(
                outcomes[record["id"]]["status"] == "validated"
                for record in incomplete
            ),
            "parser_failures": sum(
                outcomes[record["id"]]["status"] == "parser_failure"
                for record in incomplete
            ),
            "runner_failures": sum(
                outcomes[record["id"]]["status"] == "runner_failure"
                for record in incomplete
            ),
        }
        metrics["models"][role] = baseline

        templates: dict[str, list[dict[str, Any]]] = defaultdict(list)
        combinations: dict[str, list[dict[str, Any]]] = defaultdict(list)
        boundaries: dict[str, dict[str, list[dict[str, Any]]]] = {
            fact: defaultdict(list) for fact in FACTS
        }
        for record in complete:
            templates[record["provenance"]["template_id"]].append(record)
            combinations[record["provenance"]["combination"]].append(record)
            for fact, value in BOUNDARIES.items():
                boundaries[fact][str(value(record))].append(record)
        metrics["by_template"][role] = {
            key: _summary(group, outcomes) for key, group in sorted(templates.items())
        }
        metrics["by_combination"][role] = {
            key: _summary(group, outcomes)
            for key, group in sorted(combinations.items())
        }
        metrics["by_boundary"][role] = {
            fact: {
                key: _boundary_summary(fact, group, outcomes)
                for key, group in sorted(groups.items())
            }
            for fact, groups in boundaries.items()
        }

        for record in records:
            outcome = outcomes[record["id"]]
            if outcome["status"] != "validated":
                failures.append(
                    {
                        "type": outcome["status"],
                        "model": role,
                        "prompt_id": record["id"],
                        "error": outcome["error"],
                    }
                )
            elif record["kind"] == "incomplete":
                failures.append(
                    {
                        "type": "incomplete_guess",
                        "model": role,
                        "prompt_id": record["id"],
                        "missing_concept": record["provenance"]["missing_concept"],
                        "prediction": outcome["prediction"],
                    }
                )
            else:
                errors = [
                    fact
                    for fact in FACTS
                    if outcome["prediction"][fact] != record["facts"][fact]
                ]
                if errors:
                    failures.append(
                        {
                            "type": "fact_error",
                            "model": role,
                            "prompt_id": record["id"],
                            "fact_errors": errors,
                            "expected": record["facts"],
                            "prediction": outcome["prediction"],
                        }
                    )

        available = consistent = target_changed = invariant = 0
        for pair in pairs:
            source, target = outcomes[pair["source_id"]], outcomes[pair["target_id"]]
            check = {
                "id": pair["id"],
                "model": role,
                "criterion": pair["criterion"],
                "source_id": pair["source_id"],
                "target_id": pair["target_id"],
            }
            if source["status"] != "validated" or target["status"] != "validated":
                check |= {
                    "status": "unavailable",
                    "source_status": source["status"],
                    "target_status": target["status"],
                }
            else:
                available += 1
                changed = [
                    fact
                    for fact in FACTS
                    if source["prediction"][fact] != target["prediction"][fact]
                ]
                only_intended = changed == [pair["criterion"]]
                changed_target = pair["criterion"] in changed
                others_invariant = not any(
                    fact in changed for fact in FACTS if fact != pair["criterion"]
                )
                consistent += only_intended
                target_changed += changed_target
                invariant += others_invariant
                check |= {
                    "status": "validated",
                    "changed_facts": changed,
                    "target_fact_changed": changed_target,
                    "other_facts_invariant": others_invariant,
                    "only_intended_fact_changed": only_intended,
                }
                if not only_intended:
                    failures.append(
                        {
                            "type": "counterfactual_failure",
                            **check,
                        }
                    )
            counterfactuals.append(check)
        metrics["counterfactuals"][role] = {
            "pairs": len(pairs),
            "available": available,
            "unavailable": len(pairs) - available,
            "only_intended_fact_changed": consistent,
            "consistency": _rate(consistent, len(pairs)),
            "consistency_given_available": _rate(consistent, available),
            "target_fact_changed": target_changed,
            "target_sensitivity_given_available": _rate(target_changed, available),
            "other_facts_invariant": invariant,
            "invariance_given_available": _rate(invariant, available),
        }

    return metrics, counterfactuals, failures


def write_evidence(
    output: Path,
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
    counterfactuals: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(predictions, output / "predictions.jsonl")
    write_jsonl(counterfactuals, output / "counterfactuals.jsonl")
    write_jsonl(failures, output / "failures.jsonl")
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
