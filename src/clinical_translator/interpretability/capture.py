"""Bounded activation-capture contract for Goal 06."""

from __future__ import annotations

from typing import Any

SMOKE_COMBINATIONS = ("00000", "00001", "00010", "00100", "01000", "10000")
CAPTURE_LAYERS = (0, 15, 31)


def smoke_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        record
        for record in records
        if record["kind"] == "complete"
        and record["provenance"]["template_id"] == "t01"
        and record["provenance"]["combination"] in SMOKE_COMBINATIONS
    ]
    selected.sort(key=lambda record: record["id"])
    expected = [f"complete-t01-{bits}" for bits in SMOKE_COMBINATIONS]
    if [record["id"] for record in selected] != expected:
        raise ValueError("fixed Goal 06 smoke set is incomplete")
    return selected


def tensor_names() -> tuple[str, ...]:
    names = ["token_embedding"]
    for layer in CAPTURE_LAYERS:
        names.extend(
            (
                f"layer_{layer:02d}_residual",
                f"layer_{layer:02d}_attention",
                f"layer_{layer:02d}_mlp",
            )
        )
    names.append("next_token_logits")
    return tuple(names)


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported activation manifest")
    if manifest.get("parity", {}).get("exact_match") is not True:
        raise ValueError("instrumented outputs do not match the reference")
    capture = manifest.get("capture", {})
    if capture.get("layers") != list(CAPTURE_LAYERS):
        raise ValueError("unexpected capture layers")
    if capture.get("token_policy") != "last_prompt_token":
        raise ValueError("unexpected token policy")
    if capture.get("dtype") != "torch.bfloat16":
        raise ValueError("reference activations must use BF16")
    model = manifest.get("model", {})
    if not model.get("repo_id") or not model.get("revision"):
        raise ValueError("activation manifest must pin the model")
    records = manifest.get("records", [])
    if len(records) != len(SMOKE_COMBINATIONS):
        raise ValueError("activation manifest must cover the fixed smoke set")
    expected_ids = [f"complete-t01-{bits}" for bits in SMOKE_COMBINATIONS]
    if [record.get("prompt", {}).get("id") for record in records] != expected_ids:
        raise ValueError("activation manifest uses the wrong smoke prompts")
    expected_names = set(tensor_names())
    for record in records:
        token_position = record.get("token", {}).get("position")
        if not isinstance(token_position, int):
            raise ValueError("activation token position is missing")
        tensors = record.get("tensors", [])
        if {tensor.get("name") for tensor in tensors} != expected_names:
            raise ValueError("activation tensor set is incomplete")
        for tensor in tensors:
            if tensor.get("dtype") != "torch.bfloat16":
                raise ValueError("captured tensor is not BF16")
            if tensor.get("token_position") != token_position:
                raise ValueError("tensor metadata uses the wrong token")
            shape = tensor.get("shape", [])
            if len(shape) != 3 or shape[:2] != [1, 1] or shape[2] <= 0:
                raise ValueError("captured tensor has an unexpected shape")
