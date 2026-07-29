"""Load and validate frozen experiment contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

FACTS = (
    "confusion",
    "elevated_urea",
    "high_respiratory_rate",
    "low_blood_pressure",
    "age_at_least_65",
)
UNSUPPORTED_CLAIMS = {
    "unrestricted natural-language equivalence",
    "diagnosis or treatment safety",
    "whole-model formal verification",
}


def load(path: str | Path) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text())
    errors = validate(contract)
    if errors:
        raise ValueError("\n".join(errors))
    return contract


def reference(contract: dict[str, Any]) -> str:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return f"{contract['contract_id']}@sha256:{hashlib.sha256(payload).hexdigest()}"


def validate(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    schema = contract.get("output_schema", {})
    properties = schema.get("properties", {})
    models = contract.get("models", {})
    domain = contract.get("input_domain", {})
    reproducibility = contract.get("reproducibility", {})
    claims = contract.get("claims", {})

    if contract.get("status") != "frozen":
        errors.append("status must be frozen")
    if not contract.get("contract_id"):
        errors.append("contract_id is required")
    if schema.get("type") != "object":
        errors.append("output_schema.type must be object")
    if tuple(schema.get("required", ())) != FACTS:
        errors.append("output_schema.required must contain the five facts in order")
    if set(properties) != set(FACTS):
        errors.append("output_schema.properties must contain only the five facts")
    if any(properties.get(fact) != {"type": "boolean"} for fact in FACTS):
        errors.append("every output fact must be Boolean")
    if schema.get("additionalProperties") is not False:
        errors.append("output_schema must reject additional properties")

    for role in ("primary", "control"):
        model = models.get(role, {})
        revision = model.get("revision", "")
        if not model.get("repo_id") or len(revision) != 40 or any(
            char not in "0123456789abcdef" for char in revision
        ):
            errors.append(f"models.{role} must pin a repo_id and 40-character revision")

    if domain.get("grammar_version") != "curb65-vignette-grammar-v1":
        errors.append("the v1 prompt grammar must be pinned")
    if domain.get("logical_combinations") != 2 ** len(FACTS):
        errors.append("input_domain must contain all 32 logical combinations")
    if domain.get("patient_data") is not False:
        errors.append("patient data must be excluded")

    if not {"seeds", "software", "hardware", "generation"} <= reproducibility.keys():
        errors.append("seeds, software, hardware, and generation settings are required")
    if not UNSUPPORTED_CLAIMS <= set(claims.get("unsupported", ())):
        errors.append("all mandatory unsupported claims must be explicit")

    modal = contract.get("model_execution", {})
    if modal.get("backend") != "modal" or modal.get("allow_local_weights") is not False:
        errors.append("model weights and inference must be restricted to Modal")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a frozen experiment contract")
    parser.add_argument(
        "path",
        nargs="?",
        default="configs/contracts/curb65-llama-v2.json",
    )
    contract = load(parser.parse_args().path)
    print(reference(contract))
