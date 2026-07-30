"""Bounded fact-token circuit extraction for Goal 08."""

from __future__ import annotations

import math
from typing import Any

from clinical_translator.contracts.validation import FACTS

TRAIN_TEMPLATE = "t06"
HELD_OUT_TEMPLATES = ("t07", "t08")
KEEP_TOLERANCE = 0.90
CANDIDATE_SIZES = (2, 4, 8, 16, 24, 32)
BATCH_SIZE = 8


def circuit_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        record
        for record in records
        if record["kind"] == "complete"
        and record["provenance"]["template_id"]
        in (TRAIN_TEMPLATE, *HELD_OUT_TEMPLATES)
    ]
    selected.sort(key=lambda record: record["id"])
    counts = {
        template: sum(
            record["provenance"]["template_id"] == template for record in selected
        )
        for template in (TRAIN_TEMPLATE, *HELD_OUT_TEMPLATES)
    }
    if counts != {TRAIN_TEMPLATE: 32, "t07": 32, "t08": 32}:
        raise ValueError("circuit extraction requires complete t06-t08 prompt sets")
    return selected


def validate_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != 1:
        raise ValueError("unsupported circuit report")
    scope = report.get("scope", {})
    if scope.get("train_templates") != [TRAIN_TEMPLATE]:
        raise ValueError("unexpected circuit-selection template")
    if scope.get("held_out_templates") != list(HELD_OUT_TEMPLATES):
        raise ValueError("unexpected circuit-evaluation templates")
    if scope.get("held_out_prompts") != 64 or not scope.get(
        "numeric_boundaries_covered"
    ):
        raise ValueError("held-out boundary coverage is incomplete")

    models = report.get("models", {})
    if set(models) != {"primary", "control"}:
        raise ValueError("Med42 and Meta-Llama circuit results are required")
    for result in models.values():
        circuits = result.get("circuits", {})
        if set(circuits) != set(FACTS):
            raise ValueError("circuit results must cover all five facts")
        for circuit in circuits.values():
            selection = circuit.get("selection", {})
            if not 0 < selection.get("retained_components", 0) < selection.get(
                "candidate_components", 0
            ):
                raise ValueError("circuit was not causally pruned")
            keep = circuit.get("keep_only", {})
            if keep.get("agreement_with_full_model", 0) < KEEP_TOLERANCE:
                raise ValueError("keep-only circuit missed the declared tolerance")
            remove = circuit.get("remove_only", {})
            if remove.get("degradation_pass") is not True:
                raise ValueError("remove-only circuit caused no predicted degradation")
            if circuit.get("causal_ablation", {}).get("pass") is not True:
                raise ValueError("retained path failed causal ablation")
            node_types = {
                node.get("type") for node in circuit.get("graph", {}).get("nodes", [])
            }
            if not {"feature", "attention_head", "mlp", "fact_logit"} <= node_types:
                raise ValueError("sparse graph is missing required node types")
            if not circuit.get("graph", {}).get("edges"):
                raise ValueError("sparse graph has no retained edges")

    comparison = report.get("comparison", {})
    if set(comparison) != set(FACTS):
        raise ValueError("model circuit comparison must cover all five facts")
    for item in comparison.values():
        overlap = item.get("component_jaccard")
        if not isinstance(overlap, (int, float)) or not 0 <= overlap <= 1:
            raise ValueError("invalid Med42/Meta-Llama circuit comparison")


def extract(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    feature_layers: dict[str, int],
) -> dict[str, Any]:
    import torch

    layers = model.model.layers
    head_count = model.config.num_attention_heads
    head_dim = model.config.hidden_size // head_count
    if len(set(feature_layers.values())) != 1:
        raise ValueError("Goal 08 requires one shared feature-dictionary boundary")
    boundary_layer = next(iter(feature_layers.values()))
    candidate_layers = tuple(range(boundary_layer, len(layers)))
    candidate_count = len(candidate_layers) * (head_count + 1)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    token_ids = {}
    for value in ("true", "false"):
        encoded = tokenizer.encode(value, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"{value!r} is not a single fact token")
        token_ids[value] = encoded[0]
    logit_direction = (
        model.lm_head.weight[token_ids["true"]]
        - model.lm_head.weight[token_ids["false"]]
    ).detach()

    train = [record for record in records if record["template_id"] == TRAIN_TEMPLATE]
    held_out = [
        record
        for record in records
        if record["template_id"] in HELD_OUT_TEMPLATES
    ]

    def assistant_prefix(record: dict[str, Any], fact: str) -> str:
        fields = []
        for name in FACTS:
            if name == fact:
                break
            fields.append(f'"{name}": {str(record["baseline"][name]).lower()}')
        before = ", ".join(fields)
        return (
            "{"
            + (before + ", " if before else "")
            + f'"{fact}": '
        )

    def batch_inputs(batch: list[dict[str, Any]], fact: str) -> Any:
        rendered = [
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": record["prompt"]},
                    {
                        "role": "assistant",
                        "content": assistant_prefix(record, fact),
                    },
                ],
                continue_final_message=True,
                tokenize=False,
            )
            for record in batch
        ]
        return tokenizer(
            rendered,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

    def intervention_hooks(mode: str, selected: set[str]) -> list[Any]:
        handles = []
        for layer_index in candidate_layers:
            layer = layers[layer_index]
            selected_heads = {
                int(component.rsplit(":", 1)[-1])
                for component in selected
                if component.startswith(f"head:{layer_index:02d}:")
            }

            def edit_heads(
                _module: Any,
                inputs: tuple[Any, ...],
                *,
                allowed: set[int] = selected_heads,
            ) -> tuple[Any, ...]:
                tensor = inputs[0]
                changed = tensor.clone()
                heads = changed[:, -1, :].view(-1, head_count, head_dim)
                if mode == "keep":
                    removed = set(range(head_count)) - allowed
                else:
                    removed = allowed
                if removed:
                    heads[:, sorted(removed), :] = 0
                return (changed, *inputs[1:])

            mlp_id = f"mlp:{layer_index:02d}"

            def edit_mlp(
                _module: Any,
                _inputs: Any,
                output: Any,
                *,
                component: str = mlp_id,
            ) -> Any:
                remove = (
                    component not in selected
                    if mode == "keep"
                    else component in selected
                )
                if not remove:
                    return output
                changed = output.clone()
                changed[:, -1, :] = 0
                return changed

            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(edit_heads))
            handles.append(layer.mlp.register_forward_hook(edit_mlp))
        return handles

    def margins(
        subset: list[dict[str, Any]],
        fact: str,
        mode: str | None = None,
        selected: set[str] | None = None,
    ) -> Any:
        handles = intervention_hooks(mode, selected or set()) if mode else []
        values = []
        try:
            with torch.inference_mode():
                for start in range(0, len(subset), BATCH_SIZE):
                    inputs = batch_inputs(subset[start : start + BATCH_SIZE], fact)
                    logits = model(
                        **inputs,
                        use_cache=False,
                        logits_to_keep=1,
                    ).logits
                    if logits.ndim == 3:
                        logits = logits[:, -1, :]
                    values.append(
                        (
                            logits[:, token_ids["true"]]
                            - logits[:, token_ids["false"]]
                        )
                        .float()
                        .cpu()
                    )
        finally:
            for handle in handles:
                handle.remove()
        return torch.cat(values)

    def rank_components(
        fact: str,
        full_train_margins: Any,
    ) -> list[dict[str, Any]]:
        head_sums = [torch.zeros(head_count) for _ in layers]
        mlp_sums = torch.zeros(len(layers))
        current_sign: Any = None
        handles = []
        for layer_index in candidate_layers:
            layer = layers[layer_index]
            effective = (
                layer.self_attn.o_proj.weight.T @ logit_direction
            ).view(head_count, head_dim)

            def score_heads(
                _module: Any,
                inputs: tuple[Any, ...],
                *,
                index: int = layer_index,
                projection: Any = effective,
            ) -> None:
                heads = inputs[0][:, -1, :].view(-1, head_count, head_dim)
                scores = (heads * projection).sum(dim=-1) * current_sign[:, None]
                head_sums[index].add_(scores.sum(dim=0).detach().float().cpu())

            def score_mlp(
                _module: Any,
                _inputs: Any,
                output: Any,
                *,
                index: int = layer_index,
            ) -> None:
                scores = (output[:, -1, :] * logit_direction).sum(dim=-1)
                mlp_sums[index] += float((scores * current_sign).sum())

            handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(score_heads)
            )
            handles.append(layer.mlp.register_forward_hook(score_mlp))
        try:
            with torch.inference_mode():
                for start in range(0, len(train), BATCH_SIZE):
                    end = start + BATCH_SIZE
                    current_sign = torch.where(
                        full_train_margins[start:end].to("cuda") >= 0,
                        1.0,
                        -1.0,
                    )
                    model(
                        **batch_inputs(train[start:end], fact),
                        use_cache=False,
                        logits_to_keep=1,
                    )
        finally:
            for handle in handles:
                handle.remove()

        ranked = []
        count = len(train)
        for layer_index in candidate_layers:
            scores = head_sums[layer_index]
            for head, score in enumerate(scores):
                ranked.append(
                    {
                        "id": f"head:{layer_index:02d}:{head:02d}",
                        "type": "attention_head",
                        "layer": layer_index,
                        "head": head,
                        "direct_attribution": float(score / count),
                    }
                )
            ranked.append(
                {
                    "id": f"mlp:{layer_index:02d}",
                    "type": "mlp",
                    "layer": layer_index,
                    "direct_attribution": float(mlp_sums[layer_index] / count),
                }
            )
        ranked.sort(key=lambda item: item["direct_attribution"], reverse=True)
        return ranked

    def select_components(ranked: list[dict[str, Any]], size: int) -> set[str]:
        heads = [item for item in ranked if item["type"] == "attention_head"]
        mlps = [item for item in ranked if item["type"] == "mlp"]
        mlp_count = max(1, min(len(mlps), round(size * len(mlps) / candidate_count)))
        head_count_for_size = min(len(heads), size - mlp_count)
        return {
            item["id"]
            for item in (
                heads[:head_count_for_size]
                + mlps[:mlp_count]
            )
        }

    def metrics(
        values: Any,
        full_values: Any,
        subset: list[dict[str, Any]],
        fact: str,
    ) -> dict[str, float | int]:
        choices = values >= 0
        full_choices = full_values >= 0
        expected = torch.tensor([record["expected"][fact] for record in subset])
        full_sign = torch.where(full_choices, 1.0, -1.0)
        return {
            "prompts": len(subset),
            "agreement_with_full_model": float((choices == full_choices).float().mean()),
            "expected_accuracy": float((choices == expected).float().mean()),
            "full_choice_margin": float((full_sign * values).mean()),
        }

    circuits = {}
    for fact in FACTS:
        full_train = margins(train, fact)
        full_held_out = margins(held_out, fact)
        ranked = rank_components(fact, full_train)
        full_metrics = metrics(full_held_out, full_held_out, held_out, fact)
        pruning = []
        accepted = None
        for size in CANDIDATE_SIZES:
            selected = select_components(ranked, size)
            kept = margins(held_out, fact, "keep", selected)
            keep_metrics = metrics(kept, full_held_out, held_out, fact)
            trial: dict[str, Any] = {
                "retained_components": len(selected),
                "retained_heads": sum(item.startswith("head:") for item in selected),
                "retained_mlps": sum(item.startswith("mlp:") for item in selected),
                "keep_only": keep_metrics,
            }
            if keep_metrics["agreement_with_full_model"] >= KEEP_TOLERANCE:
                removed = margins(held_out, fact, "remove", selected)
                remove_metrics = metrics(removed, full_held_out, held_out, fact)
                remove_metrics["full_choice_margin_drop"] = (
                    full_metrics["full_choice_margin"]
                    - remove_metrics["full_choice_margin"]
                )
                remove_metrics["expected_accuracy_drop"] = (
                    full_metrics["expected_accuracy"]
                    - remove_metrics["expected_accuracy"]
                )
                remove_metrics["degradation_pass"] = (
                    remove_metrics["full_choice_margin_drop"] > 0
                    or remove_metrics["expected_accuracy_drop"] > 0
                )
                trial["remove_only"] = remove_metrics
                if remove_metrics["degradation_pass"]:
                    accepted = (selected, keep_metrics, remove_metrics)
            pruning.append(trial)
            if accepted:
                break
        if accepted is None:
            raise RuntimeError(
                f"no compact circuit passed for {fact}: {pruning[-1]}"
            )

        selected, keep_metrics, remove_metrics = accepted
        selected_details = [item for item in ranked if item["id"] in selected]
        feature_id = f"feature:{fact}:layer:{feature_layers[fact]:02d}"
        output_id = f"fact_logit:{fact}"
        nodes = [
            {
                "id": feature_id,
                "type": "feature",
                "layer": feature_layers[fact],
                "source": "Goal 07 mean-difference direction",
            },
            *selected_details,
            {
                "id": output_id,
                "type": "fact_logit",
                "positive_token": "true",
                "negative_token": "false",
            },
        ]
        edges = [{"source": feature_id, "target": output_id, "kind": "logit_path"}]
        for component in selected_details:
            if component["layer"] <= feature_layers[fact]:
                edges.append(
                    {
                        "source": component["id"],
                        "target": feature_id,
                        "kind": "upstream",
                        "weight": component["direct_attribution"],
                    }
                )
            else:
                edges.extend(
                    (
                        {
                            "source": feature_id,
                            "target": component["id"],
                            "kind": "downstream",
                            "weight": component["direct_attribution"],
                        },
                        {
                            "source": component["id"],
                            "target": output_id,
                            "kind": "logit_contribution",
                            "weight": component["direct_attribution"],
                        },
                    )
                )
        circuits[fact] = {
            "fact_token_logit": {
                "positive_token": "true",
                "positive_token_id": token_ids["true"],
                "negative_token": "false",
                "negative_token_id": token_ids["false"],
                "trace_direction": "backward_from_fact_token_logit",
            },
            "selection": {
                "procedure": (
                    "rank last-token heads and MLPs by signed direct attribution "
                    "on t06; retain the smallest tested prefix passing both gates"
                ),
                "candidate_components": candidate_count,
                "retained_components": len(selected),
                "pruned_components": candidate_count - len(selected),
                "keep_tolerance": KEEP_TOLERANCE,
            },
            "full_model": full_metrics,
            "pruning": pruning,
            "keep_only": keep_metrics,
            "remove_only": remove_metrics,
            "causal_ablation": {
                "pass": remove_metrics["degradation_pass"],
                "intervention": (
                    "zero retained component outputs at the final decision token"
                ),
            },
            "graph": {
                "nodes": nodes,
                "edges": edges,
            },
        }

    return {
        "scope": {
            "feature_boundary_layer": boundary_layer,
            "candidate_layers": list(candidate_layers),
            "attention_heads_per_layer": head_count,
            "candidate_components_per_fact": candidate_count,
            "selection_prompts": len(train),
            "held_out_prompts": len(held_out),
            "intervention_position": "final structured fact decision token",
            "context_policy": (
                f"layers 0-{boundary_layer - 1} are the frozen context encoder; "
                "keep/remove interventions cover the feature-to-logit stage"
            ),
        },
        "circuits": circuits,
    }
