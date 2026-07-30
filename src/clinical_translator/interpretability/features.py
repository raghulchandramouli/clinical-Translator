"""Minimal feature-discovery contract for the five CURB-65 facts."""

from __future__ import annotations

import math
from typing import Any

from clinical_translator.contracts.validation import FACTS

TRAIN_TEMPLATES = tuple(f"t{index:02d}" for index in range(1, 7))
HELD_OUT_TEMPLATES = ("t07", "t08")
DICTIONARY_RANK = 128
DICTIONARY_ACTIVE_FEATURES = 64


def feature_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [record for record in records if record["kind"] == "complete"]
    selected.sort(key=lambda record: record["id"])
    counts = {
        template: sum(
            record["provenance"]["template_id"] == template for record in selected
        )
        for template in TRAIN_TEMPLATES + HELD_OUT_TEMPLATES
    }
    if len(selected) != 256 or set(counts.values()) != {32}:
        raise ValueError("feature discovery requires all 256 complete prompts")
    return selected


def held_out_interventions(
    pairs: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    primary = {
        item["prompt_id"]: item["prediction"]
        for item in predictions
        if item["model"] == "primary"
        and item["kind"] == "complete"
        and item["status"] == "validated"
    }
    cases = []
    for pair in pairs:
        template = pair["id"].split("-")[1]
        source = primary.get(pair["source_id"])
        target = primary.get(pair["target_id"])
        criterion = pair["criterion"]
        if (
            template not in HELD_OUT_TEMPLATES
            or source is None
            or target is None
            or source[criterion]
            or not target[criterion]
            or any(
                source[fact] != target[fact] for fact in FACTS if fact != criterion
            )
        ):
            continue
        cases.append(
            {
                "id": pair["id"],
                "criterion": criterion,
                "source_id": pair["source_id"],
                "target_id": pair["target_id"],
                "source_facts": source,
                "target_facts": target,
            }
        )
    cases.sort(key=lambda case: case["id"])
    if not set(FACTS) <= {case["criterion"] for case in cases}:
        raise ValueError("held-out predictions lack an intervention case for every fact")
    return cases


def validate_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != 1:
        raise ValueError("unsupported feature report")
    split = report.get("split", {})
    if split.get("train_templates") != list(TRAIN_TEMPLATES):
        raise ValueError("unexpected training templates")
    if split.get("held_out_templates") != list(HELD_OUT_TEMPLATES):
        raise ValueError("unexpected held-out templates")
    if split.get("train_prompts") != 192 or split.get("held_out_prompts") != 64:
        raise ValueError("unexpected feature-discovery split")

    candidates = report.get("ranked_candidates", [])
    if {candidate.get("criterion") for candidate in candidates} != set(FACTS):
        raise ValueError("ranked candidates must cover all five facts")
    for candidate in candidates:
        accuracy = candidate.get("held_out_accuracy")
        if not isinstance(accuracy, (int, float)) or not 0 <= accuracy <= 1:
            raise ValueError("candidate lacks a held-out decodability result")

    dictionary = report.get("dictionary", {})
    if dictionary.get("source") != "task_specific_topk_linear_autoencoder":
        raise ValueError("feature dictionary must be task-specific")
    if dictionary.get("rank") != DICTIONARY_RANK:
        raise ValueError("unexpected dictionary rank")
    if dictionary.get("active_features") != DICTIONARY_ACTIVE_FEATURES:
        raise ValueError("unexpected sparse dictionary activation count")
    if dictionary.get("external_dictionary_used") is not False:
        raise ValueError("an unmeasured external dictionary was accepted")
    measurements = dictionary.get("reconstruction", [])
    if not measurements or not any(item.get("adequate") is True for item in measurements):
        raise ValueError("no dictionary passed the held-out reconstruction gate")
    for item in measurements:
        for key in ("train_explained_variance", "held_out_explained_variance"):
            if not isinstance(item.get(key), (int, float)) or not math.isfinite(item[key]):
                raise ValueError("dictionary reconstruction metric is missing")

    intervention = report.get("causal_intervention", {})
    if intervention.get("success") is not True:
        raise ValueError("no candidate passed causal intervention")
    if intervention.get("template_id") not in HELD_OUT_TEMPLATES:
        raise ValueError("causal intervention must use a held-out template")
