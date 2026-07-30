"""Matched causal validation for retained CURB-65 variables and circuits."""

from __future__ import annotations

import math
from typing import Any

from clinical_translator.contracts.validation import FACTS

HELD_OUT_TEMPLATES = ("t07", "t08")
BATCH_SIZE = 8
UNRELATED_STABILITY_GATE = 0.90


def causal_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        record
        for record in records
        if record["kind"] == "complete"
        and record["provenance"]["template_id"] in HELD_OUT_TEMPLATES
    ]
    selected.sort(key=lambda record: record["id"])
    if len(selected) != 64:
        raise ValueError("causal validation requires all 64 held-out prompts")
    return selected


def causal_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        pair
        for pair in pairs
        if pair["id"].split("-")[1] in HELD_OUT_TEMPLATES
    ]
    selected.sort(key=lambda pair: pair["id"])
    counts = {
        fact: sum(pair["criterion"] == fact for pair in selected) for fact in FACTS
    }
    if len(selected) != 160 or set(counts.values()) != {32}:
        raise ValueError("causal validation requires 32 held-out pairs per fact")
    return selected


def validate_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != 1:
        raise ValueError("unsupported causal-validation report")
    scope = report.get("scope", {})
    if scope.get("templates") != list(HELD_OUT_TEMPLATES):
        raise ValueError("unexpected causal-validation templates")
    if scope.get("matched_pairs") != 160:
        raise ValueError("matched counterfactual coverage is incomplete")

    variables = report.get("variables", {})
    if set(variables) != set(FACTS):
        raise ValueError("causal evidence must cover all five variables")
    for variable in variables.values():
        interchange = variable.get("interchange", {})
        if interchange.get("eligible_pairs", 0) <= 0:
            raise ValueError("variable has no observable matched intervention")
        if interchange.get("predicted_direction_rate") != 1.0:
            raise ValueError("matched patch did not control the emitted fact")
        stability = interchange.get("unrelated_output_stability")
        if not isinstance(stability, (int, float)) or not 0 <= stability <= 1:
            raise ValueError("unrelated-output stability is missing")
        if set(variable.get("ablations", {})) != {
            "feature",
            "attention_heads",
            "mlp",
            "path",
        }:
            raise ValueError("feature/head/MLP/path ablations are incomplete")
        if set(variable.get("controls", {})) != {
            "positive_counterfactual",
            "negative_sham",
            "negative_unrelated_feature",
        }:
            raise ValueError("positive and negative controls are required")
        for section in (
            interchange,
            *variable["ablations"].values(),
            *variable["controls"].values(),
        ):
            for key, value in section.items():
                if key.endswith(("effect_size", "accuracy", "stability")) and (
                    not isinstance(value, (int, float)) or not math.isfinite(value)
                ):
                    raise ValueError("causal metric is missing or non-finite")
        if variable.get("status") not in {"pass", "mixed", "fail"}:
            raise ValueError("variable lacks explicit pass/fail evidence")

    if not isinstance(report.get("counterexamples"), list):
        raise ValueError("failed and mixed interventions must remain visible")


def validate(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    circuits: dict[str, Any],
    directions: dict[str, Any],
) -> dict[str, Any]:
    import torch

    by_id = {record["id"]: record for record in records}
    layers = model.model.layers
    feature_layers = {
        fact: next(
            node["layer"]
            for node in circuits[fact]["graph"]["nodes"]
            if node["type"] == "feature"
        )
        for fact in FACTS
    }
    if len(set(feature_layers.values())) != 1:
        raise ValueError("causal validation requires one shared feature boundary")
    layer_index = next(iter(feature_layers.values()))
    layer = layers[layer_index]
    head_count = model.config.num_attention_heads
    head_dim = model.config.hidden_size // head_count
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_ids = {
        value: tokenizer.encode(value, add_special_tokens=False)[0]
        for value in ("true", "false")
    }

    def assistant_prefix(record: dict[str, Any], fact: str) -> str:
        fields = []
        for name in FACTS:
            if name == fact:
                break
            fields.append(f'"{name}": {str(record["expected"][name]).lower()}')
        return "{" + (", ".join(fields) + ", " if fields else "") + f'"{fact}": '

    def batch_inputs(ids: list[str], fact: str) -> Any:
        rendered = [
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": by_id[item]["prompt"]},
                    {
                        "role": "assistant",
                        "content": assistant_prefix(by_id[item], fact),
                    },
                ],
                continue_final_message=True,
                tokenize=False,
            )
            for item in ids
        ]
        return tokenizer(
            rendered,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

    baseline: dict[str, dict[str, dict[str, Any]]] = {
        fact: {} for fact in FACTS
    }
    ordered_ids = [record["id"] for record in records]
    for fact in FACTS:
        for start in range(0, len(ordered_ids), BATCH_SIZE):
            ids = ordered_ids[start : start + BATCH_SIZE]
            captured: Any = None

            def capture(_module: Any, _inputs: Any, output: Any) -> None:
                nonlocal captured
                tensor = output[0] if isinstance(output, tuple) else output
                captured = tensor[:, -1, :].detach().float().cpu()

            handle = layer.register_forward_hook(capture)
            try:
                with torch.inference_mode():
                    logits = model(
                        **batch_inputs(ids, fact),
                        use_cache=False,
                        logits_to_keep=1,
                    ).logits
            finally:
                handle.remove()
            if logits.ndim == 3:
                logits = logits[:, -1, :]
            margins = (
                logits[:, token_ids["true"]] - logits[:, token_ids["false"]]
            ).detach().float().cpu()
            for item, margin, hidden in zip(ids, margins, captured, strict=True):
                baseline[fact][item] = {
                    "margin": float(margin),
                    "hidden": hidden,
                    "choice": bool(margin >= 0),
                }

    def patched_margins(
        subset: list[dict[str, Any]],
        query_fact: str,
        mode: str,
        variable_fact: str,
    ) -> list[float]:
        values = []
        direction = directions[variable_fact].detach().float().cpu()
        for start in range(0, len(subset), BATCH_SIZE):
            batch = subset[start : start + BATCH_SIZE]
            sources = [pair["source_id"] for pair in batch]
            source_hidden = torch.stack(
                [baseline[query_fact][item]["hidden"] for item in sources]
            )
            target_hidden = torch.stack(
                [
                    baseline[query_fact][pair["target_id"]]["hidden"]
                    for pair in batch
                ]
            )
            if mode == "full":
                replacement = target_hidden
            elif mode == "sham":
                replacement = source_hidden
            else:
                selected_direction = (
                    direction
                    if mode == "feature"
                    else directions[
                        FACTS[(FACTS.index(variable_fact) + 1) % len(FACTS)]
                    ].detach().float().cpu()
                )
                delta = (
                    (target_hidden - source_hidden) @ selected_direction
                )[:, None] * selected_direction
                replacement = source_hidden + delta
            replacement = replacement.to("cuda", dtype=torch.bfloat16)

            def patch(_module: Any, _inputs: Any, output: Any) -> Any:
                tensor = output[0] if isinstance(output, tuple) else output
                changed = tensor.clone()
                changed[:, -1, :] = replacement
                return (changed, *output[1:]) if isinstance(output, tuple) else changed

            handle = layer.register_forward_hook(patch)
            try:
                with torch.inference_mode():
                    logits = model(
                        **batch_inputs(sources, query_fact),
                        use_cache=False,
                        logits_to_keep=1,
                    ).logits
            finally:
                handle.remove()
            if logits.ndim == 3:
                logits = logits[:, -1, :]
            values.extend(
                (
                    logits[:, token_ids["true"]] - logits[:, token_ids["false"]]
                )
                .detach()
                .float()
                .cpu()
                .tolist()
            )
        return values

    def ablated_margins(
        ids: list[str],
        fact: str,
        mode: str,
        selected: set[str],
    ) -> list[float]:
        handles = []
        selected_heads = {
            int(component.rsplit(":", 1)[-1])
            for component in selected
            if component.startswith(f"head:{layer_index:02d}:")
        }
        if mode in {"attention_heads", "path"}:

            def remove_heads(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
                changed = inputs[0].clone()
                heads = changed[:, -1, :].view(-1, head_count, head_dim)
                if selected_heads:
                    heads[:, sorted(selected_heads), :] = 0
                return (changed, *inputs[1:])

            handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(remove_heads)
            )
        if mode in {"mlp", "path"} and f"mlp:{layer_index:02d}" in selected:

            def remove_mlp(_module: Any, _inputs: Any, output: Any) -> Any:
                changed = output.clone()
                changed[:, -1, :] = 0
                return changed

            handles.append(layer.mlp.register_forward_hook(remove_mlp))
        if mode == "feature":
            direction = directions[fact].detach().to("cuda", dtype=torch.bfloat16)

            def remove_feature(_module: Any, _inputs: Any, output: Any) -> Any:
                tensor = output[0] if isinstance(output, tuple) else output
                changed = tensor.clone()
                final = changed[:, -1, :]
                final -= (final @ direction)[:, None] * direction
                return (changed, *output[1:]) if isinstance(output, tuple) else changed

            handles.append(layer.register_forward_hook(remove_feature))

        values = []
        try:
            with torch.inference_mode():
                for start in range(0, len(ids), BATCH_SIZE):
                    logits = model(
                        **batch_inputs(ids[start : start + BATCH_SIZE], fact),
                        use_cache=False,
                        logits_to_keep=1,
                    ).logits
                    if logits.ndim == 3:
                        logits = logits[:, -1, :]
                    values.extend(
                        (
                            logits[:, token_ids["true"]]
                            - logits[:, token_ids["false"]]
                        )
                        .detach()
                        .float()
                        .cpu()
                        .tolist()
                    )
        finally:
            for handle in handles:
                handle.remove()
        return values

    results = {}
    counterexamples = []
    for fact in FACTS:
        fact_pairs = [pair for pair in pairs if pair["criterion"] == fact]
        source_margins = [
            baseline[fact][pair["source_id"]]["margin"] for pair in fact_pairs
        ]
        target_margins = [
            baseline[fact][pair["target_id"]]["margin"] for pair in fact_pairs
        ]
        full = patched_margins(fact_pairs, fact, "full", fact)
        feature = patched_margins(fact_pairs, fact, "feature", fact)
        sham = patched_margins(fact_pairs, fact, "sham", fact)
        unrelated_feature = patched_margins(
            fact_pairs,
            fact,
            "unrelated",
            fact,
        )
        eligible = [
            index
            for index, (source, target) in enumerate(
                zip(source_margins, target_margins, strict=True)
            )
            if source < 0 <= target
        ]
        full_flips = [index for index in eligible if full[index] >= 0]
        feature_flips = [index for index in eligible if feature[index] >= 0]

        unrelated_total = 0
        unrelated_stable = 0
        unrelated_changed = []
        for other in FACTS:
            if other == fact:
                continue
            patched = patched_margins(fact_pairs, other, "full", fact)
            for pair, value in zip(fact_pairs, patched, strict=True):
                unchanged = (
                    value >= 0
                ) == baseline[other][pair["source_id"]]["choice"]
                unrelated_total += 1
                unrelated_stable += unchanged
                if not unchanged:
                    unrelated_changed.append(
                        {"pair_id": pair["id"], "output": other}
                    )

        selected = {
            node["id"]
            for node in circuits[fact]["graph"]["nodes"]
            if node["type"] in {"attention_head", "mlp"}
        }
        target_ids = sorted({pair["target_id"] for pair in fact_pairs})
        target_full = [baseline[fact][item]["margin"] for item in target_ids]
        target_accuracy = sum(value >= 0 for value in target_full) / len(target_full)
        ablations = {}
        for mode in ("feature", "attention_heads", "mlp", "path"):
            ablated = ablated_margins(target_ids, fact, mode, selected)
            accuracy = sum(value >= 0 for value in ablated) / len(ablated)
            margin_drop = sum(
                before - after
                for before, after in zip(target_full, ablated, strict=True)
            ) / len(ablated)
            accuracy_drop = target_accuracy - accuracy
            passed = margin_drop > 0 or accuracy_drop > 0
            ablations[mode] = {
                "prompts": len(ablated),
                "effect_size": margin_drop,
                "accuracy": accuracy,
                "accuracy_drop": accuracy_drop,
                "pass": passed,
            }
            if not passed:
                counterexamples.append(
                    {
                        "criterion": fact,
                        "experiment": f"{mode}_ablation",
                        "result": "no_predicted_degradation",
                        "effect_size": margin_drop,
                    }
                )

        stability = unrelated_stable / unrelated_total
        feature_effect = sum(
            after - before
            for before, after in zip(source_margins, feature, strict=True)
        ) / len(feature)
        source_true_rate = sum(value >= 0 for value in source_margins) / len(
            source_margins
        )
        positive_effect = sum(
            target - source
            for source, target in zip(
                source_margins,
                target_margins,
                strict=True,
            )
        ) / len(source_margins)
        sham_stability = sum(
            (after >= 0) == (before >= 0)
            for before, after in zip(source_margins, sham, strict=True)
        ) / len(sham)
        negative_stability = sum(
            (after >= 0) == (before >= 0)
            for before, after in zip(
                source_margins,
                unrelated_feature,
                strict=True,
            )
        ) / len(unrelated_feature)
        interchange_pass = bool(eligible) and len(full_flips) == len(eligible)
        status = (
            "pass"
            if interchange_pass
            and stability >= UNRELATED_STABILITY_GATE
            and all(item["pass"] for item in ablations.values())
            else "mixed"
            if interchange_pass
            else "fail"
        )
        results[fact] = {
            "status": status,
            "interchange": {
                "method": (
                    "replace the source layer-31 residual with its matched "
                    "counterfactual target residual at the fact-token decision"
                ),
                "matched_pairs": len(fact_pairs),
                "eligible_pairs": len(eligible),
                "eligible_flips": len(full_flips),
                "predicted_direction_rate": (
                    len(full_flips) / len(eligible) if eligible else 0.0
                ),
                "effect_size": sum(
                    after - before
                    for before, after in zip(source_margins, full, strict=True)
                )
                / len(full),
                "accuracy": sum(value >= 0 for value in full) / len(full),
                "feature_effect_size": feature_effect,
                "feature_eligible_flips": len(feature_flips),
                "unrelated_output_stability": stability,
                "unrelated_outputs_checked": unrelated_total,
                "failed_pair_ids": [
                    fact_pairs[index]["id"]
                    for index in eligible
                    if index not in full_flips
                ],
                "feature_failed_pair_ids": [
                    fact_pairs[index]["id"]
                    for index in eligible
                    if index not in feature_flips
                ],
                "unrelated_changed": unrelated_changed,
            },
            "ablations": ablations,
            "controls": {
                "positive_counterfactual": {
                    "effect_size": positive_effect,
                    "accuracy": target_accuracy,
                    "source_true_rate": source_true_rate,
                    "pass": positive_effect > 0 and target_accuracy > source_true_rate,
                },
                "negative_sham": {
                    "effect_size": sum(
                        abs(after - before)
                        for before, after in zip(source_margins, sham, strict=True)
                    )
                    / len(sham),
                    "stability": sham_stability,
                    "pass": sham_stability == 1.0,
                },
                "negative_unrelated_feature": {
                    "effect_size": sum(
                        abs(after - before)
                        for before, after in zip(
                            source_margins,
                            unrelated_feature,
                            strict=True,
                        )
                    )
                    / len(unrelated_feature),
                    "stability": negative_stability,
                    "pass": negative_stability >= UNRELATED_STABILITY_GATE,
                    "feature": FACTS[(FACTS.index(fact) + 1) % len(FACTS)],
                },
            },
        }
        if status != "pass":
            counterexamples.append(
                {
                    "criterion": fact,
                    "experiment": "variable_validation",
                    "result": status,
                    "unrelated_output_stability": stability,
                    "feature_effect_size": feature_effect,
                }
            )

    return {
        "scope": {
            "templates": list(HELD_OUT_TEMPLATES),
            "prompts": len(records),
            "matched_pairs": len(pairs),
            "pairs_per_variable": 32,
            "feature_layer": layer_index,
            "decision": "teacher-forced Boolean fact-token choice",
            "unrelated_stability_gate": UNRELATED_STABILITY_GATE,
        },
        "variables": results,
        "counterexamples": counterexamples,
    }
