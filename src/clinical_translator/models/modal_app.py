"""Modal-only model caching and deterministic CURB-65 translation."""

from pathlib import Path
from typing import Any

import modal

MODEL_DIR = Path("/models")
ACTIVATION_DIR = Path("/activations")
REVISION_FILE = ".clinical-translator-revision"
app = modal.App("clinical-translator-models")
volume = modal.Volume.from_name("clinical-translator-models", create_if_missing=True)
activation_volume = modal.Volume.from_name(
    "clinical-translator-activations",
    create_if_missing=True,
)
cache_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub==1.25.1")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)
runner_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface_hub==1.25.1",
        "torch==2.13.0",
        "transformers==5.14.1",
    )
    .env(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    .add_local_python_source("clinical_translator")
)


@app.function(
    image=cache_image,
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def check_access(repo_id: str, revision: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().model_info(repo_id, revision=revision).sha


@app.function(
    image=cache_image,
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
    timeout=60 * 60,
    volumes={MODEL_DIR: volume},
)
def cache_model(repo_id: str, revision: str) -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id, revision=revision, local_dir=MODEL_DIR / repo_id)
    (Path(path) / REVISION_FILE).write_text(revision, encoding="ascii")
    volume.commit()
    return path


@app.cls(
    image=runner_image,
    gpu="A10G",
    timeout=30 * 60,
    volumes={MODEL_DIR: volume, ACTIVATION_DIR: activation_volume},
)
class Translator:
    repo_id: str = modal.parameter()
    revision: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = MODEL_DIR / self.repo_id
        if (path / REVISION_FILE).read_text(encoding="ascii") != self.revision:
            raise RuntimeError(f"cached revision mismatch for {self.repo_id}")
        torch.use_deterministic_algorithms(True)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="eager",
        ).to("cuda").eval()

    @modal.method()
    def translate(
        self,
        prompts: list[str],
        generation: dict[str, Any],
        seeds: dict[str, int],
    ) -> list[str]:
        import torch

        torch.manual_seed(seeds["torch"])
        torch.cuda.manual_seed_all(seeds["cuda"])
        outputs = []
        for text in prompts:
            inputs = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": "{"},
                ],
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            ).to("cuda")
            with torch.inference_mode():
                tokens = self.model.generate(
                    **inputs,
                    do_sample=generation["do_sample"],
                    max_new_tokens=generation["max_new_tokens"],
                )
            outputs.append(
                "{" + self.tokenizer.decode(
                    tokens[0, inputs["input_ids"].shape[-1] :],
                    skip_special_tokens=True,
                ).strip()
            )
        return outputs

    @modal.method()
    def capture(
        self,
        prompts: list[str],
        prompt_refs: list[dict[str, str]],
        generation: dict[str, Any],
        seeds: dict[str, int],
        artifact_root: str,
    ) -> dict[str, Any]:
        import torch

        from clinical_translator.interpretability.capture import (
            CAPTURE_LAYERS,
            tensor_names,
        )

        if len(prompts) != len(prompt_refs):
            raise ValueError("prompts and prompt references must match")
        layers = self.model.model.layers
        if max(CAPTURE_LAYERS) >= len(layers):
            raise ValueError("capture layer is outside the model")

        torch.manual_seed(seeds["torch"])
        torch.cuda.manual_seed_all(seeds["cuda"])
        raw_outputs = []
        records = []
        for text, prompt_ref in zip(prompts, prompt_refs, strict=True):
            inputs = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": "{"},
                ],
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            ).to("cuda")
            captured: dict[str, Any] = {}

            def save(name: str):
                def hook(_module: Any, _inputs: Any, output: Any) -> None:
                    tensor = output[0] if isinstance(output, tuple) else output
                    if name in captured:
                        return
                    if tensor.ndim == 2:
                        captured[name] = tensor[:, None, :].detach().to("cpu")
                    elif tensor.ndim == 3 and tensor.shape[1] > 1:
                        captured[name] = tensor[:, -1:, :].detach().to("cpu")

                return hook

            handles = [
                self.model.model.embed_tokens.register_forward_hook(
                    save("token_embedding")
                ),
            ]
            for layer_index in CAPTURE_LAYERS:
                layer = layers[layer_index]
                handles.extend(
                    (
                        layer.register_forward_hook(
                            save(f"layer_{layer_index:02d}_residual")
                        ),
                        layer.self_attn.register_forward_hook(
                            save(f"layer_{layer_index:02d}_attention")
                        ),
                        layer.mlp.register_forward_hook(
                            save(f"layer_{layer_index:02d}_mlp")
                        ),
                    )
                )
            try:
                with torch.inference_mode():
                    tokens = self.model.generate(
                        **inputs,
                        do_sample=generation["do_sample"],
                        max_new_tokens=generation["max_new_tokens"],
                    )
                    logits = self.model(
                        **inputs,
                        use_cache=False,
                        logits_to_keep=1,
                    ).logits
                    if logits.ndim == 2:
                        logits = logits[:, None, :]
                    captured["next_token_logits"] = logits.detach().to("cpu")
            finally:
                for handle in handles:
                    handle.remove()

            missing = set(tensor_names()) - captured.keys()
            if missing:
                raise RuntimeError(f"capture hooks missed tensors: {sorted(missing)}")
            if any(tensor.dtype != torch.bfloat16 for tensor in captured.values()):
                raise RuntimeError("capture produced a non-BF16 tensor")

            raw_outputs.append(
                "{" + self.tokenizer.decode(
                    tokens[0, inputs["input_ids"].shape[-1] :],
                    skip_special_tokens=True,
                ).strip()
            )
            destination = ACTIVATION_DIR / artifact_root / f"{prompt_ref['id']}.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(captured, destination)
            token_id = int(inputs["input_ids"][0, -1])
            tensor_metadata = []
            for name in tensor_names():
                layer = (
                    int(name.split("_")[1])
                    if name.startswith("layer_")
                    else None
                )
                tensor_metadata.append(
                    {
                        "name": name,
                        "key": name,
                        "layer": layer,
                        "token_position": inputs["input_ids"].shape[-1] - 1,
                        "shape": list(captured[name].shape),
                        "dtype": str(captured[name].dtype),
                    }
                )
            records.append(
                {
                    "prompt": prompt_ref,
                    "token": {
                        "position": inputs["input_ids"].shape[-1] - 1,
                        "id": token_id,
                        "text": self.tokenizer.convert_ids_to_tokens(token_id),
                    },
                    "artifact": {
                        "volume": "clinical-translator-activations",
                        "path": str(destination.relative_to(ACTIVATION_DIR)),
                        "format": "torch.save",
                        "bytes": destination.stat().st_size,
                    },
                    "tensors": tensor_metadata,
                }
            )
        activation_volume.commit()
        return {
            "raw_outputs": raw_outputs,
            "records": records,
        }

    @modal.method()
    def discover_features(
        self,
        prompts: list[str],
        records: list[dict[str, Any]],
        intervention_cases: list[dict[str, Any]],
        generation: dict[str, Any],
        seeds: dict[str, int],
        artifact_root: str,
    ) -> dict[str, Any]:
        import torch

        from clinical_translator.contracts.validation import FACTS
        from clinical_translator.interpretability.capture import CAPTURE_LAYERS
        from clinical_translator.interpretability.features import (
            DICTIONARY_ACTIVE_FEATURES,
            DICTIONARY_RANK,
            HELD_OUT_TEMPLATES,
            TRAIN_TEMPLATES,
        )
        from clinical_translator.models.protocol import parse

        if len(prompts) != len(records):
            raise ValueError("prompts and feature records must match")
        layers = self.model.model.layers
        activations = {layer: [] for layer in CAPTURE_LAYERS}
        current: dict[int, Any] = {}

        def save(layer: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                current[layer] = tensor[:, -1, :].detach().float().cpu()

            return hook

        handles = [
            layers[layer].register_forward_hook(save(layer))
            for layer in CAPTURE_LAYERS
        ]
        try:
            with torch.inference_mode():
                for text in prompts:
                    current.clear()
                    inputs = self.tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": text},
                            {"role": "assistant", "content": "{"},
                        ],
                        continue_final_message=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to("cuda")
                    self.model(**inputs, use_cache=False, logits_to_keep=1)
                    for layer in CAPTURE_LAYERS:
                        activations[layer].append(current[layer])
        finally:
            for handle in handles:
                handle.remove()

        matrices = {
            layer: torch.cat(rows, dim=0) for layer, rows in activations.items()
        }
        train_mask = torch.tensor(
            [
                record["template_id"] in TRAIN_TEMPLATES
                for record in records
            ],
            dtype=torch.bool,
        )
        held_out_mask = torch.tensor(
            [
                record["template_id"] in HELD_OUT_TEMPLATES
                for record in records
            ],
            dtype=torch.bool,
        )
        labels = {
            fact: torch.tensor(
                [record["facts"][fact] for record in records],
                dtype=torch.bool,
            )
            for fact in FACTS
        }
        candidates = []
        directions: dict[int, dict[str, Any]] = {}
        dictionaries: dict[int, Any] = {}
        reconstruction = []
        for layer, matrix in matrices.items():
            directions[layer] = {}
            for fact in FACTS:
                fact_labels = labels[fact]
                positive = matrix[train_mask & fact_labels].mean(dim=0)
                negative = matrix[train_mask & ~fact_labels].mean(dim=0)
                direction = positive - negative
                direction /= direction.norm()
                projection = matrix @ direction
                positive_mean = float(projection[train_mask & fact_labels].mean())
                negative_mean = float(projection[train_mask & ~fact_labels].mean())
                threshold = (positive_mean + negative_mean) / 2
                predicted = projection >= threshold
                directions[layer][fact] = direction
                candidates.append(
                    {
                        "id": f"mean-direction-layer-{layer:02d}-{fact}",
                        "criterion": fact,
                        "layer": layer,
                        "method": "train_mean_difference",
                        "train_accuracy": float(
                            (predicted[train_mask] == fact_labels[train_mask])
                            .float()
                            .mean()
                        ),
                        "held_out_accuracy": float(
                            (predicted[held_out_mask] == fact_labels[held_out_mask])
                            .float()
                            .mean()
                        ),
                        "threshold": threshold,
                        "train_projection_gap": positive_mean - negative_mean,
                    }
                )

            center = matrix[train_mask].mean(dim=0)
            train_centered = matrix[train_mask] - center
            gram = train_centered @ train_centered.T
            eigenvalues, eigenvectors = torch.linalg.eigh(gram)
            order = eigenvalues.argsort(descending=True)[:DICTIONARY_RANK]
            values = eigenvalues[order].clamp_min(1e-12).sqrt()
            dictionary = (
                eigenvectors[:, order].T @ train_centered
            ) / values[:, None]
            dictionaries[layer] = dictionary

            def reconstruction_score(mask: Any) -> tuple[float, float]:
                centered = matrix[mask] - center
                encoded = centered @ dictionary.T
                keep = encoded.abs().topk(
                    DICTIONARY_ACTIVE_FEATURES,
                    dim=1,
                ).indices
                sparse = torch.zeros_like(encoded).scatter(
                    1,
                    keep,
                    encoded.gather(1, keep),
                )
                rebuilt = sparse @ dictionary
                error = (centered - rebuilt).square().sum()
                total = centered.square().sum()
                return (
                    float(1 - error / total),
                    float((error / centered.numel()).sqrt()),
                )

            train_ev, train_rmse = reconstruction_score(train_mask)
            held_out_ev, held_out_rmse = reconstruction_score(held_out_mask)
            reconstruction.append(
                {
                    "layer": layer,
                    "train_explained_variance": train_ev,
                    "held_out_explained_variance": held_out_ev,
                    "train_rmse": train_rmse,
                    "held_out_rmse": held_out_rmse,
                    "adequate": held_out_ev >= 0.5,
                }
            )

        candidates.sort(
            key=lambda item: (
                -item["held_out_accuracy"],
                -item["train_accuracy"],
                item["layer"],
                item["criterion"],
            )
        )
        for rank, candidate in enumerate(candidates, 1):
            candidate["rank"] = rank

        by_id = {record["id"]: index for index, record in enumerate(records)}
        successful: dict[str, Any] | None = None
        attempts = []
        torch.manual_seed(seeds["torch"])
        torch.cuda.manual_seed_all(seeds["cuda"])
        for candidate in candidates:
            cases = [
                case
                for case in intervention_cases
                if case["criterion"] == candidate["criterion"]
            ][:1]
            for case in cases:
                layer = candidate["layer"]
                direction = directions[layer][candidate["criterion"]]
                source = matrices[layer][by_id[case["source_id"]]]
                target = matrices[layer][by_id[case["target_id"]]]
                pair_gap = max(
                    float((target - source) @ direction),
                    candidate["train_projection_gap"],
                )
                for multiplier in (1, 2, 4, 8, 16, 32):
                    delta = (
                        direction * pair_gap * multiplier
                    ).to(device="cuda", dtype=torch.bfloat16)

                    def steer(_module: Any, _inputs: Any, output: Any) -> Any:
                        tensor = output[0] if isinstance(output, tuple) else output
                        changed = tensor.clone()
                        changed[:, -1, :] += delta
                        return (
                            (changed, *output[1:])
                            if isinstance(output, tuple)
                            else changed
                        )

                    text = prompts[by_id[case["source_id"]]]
                    inputs = self.tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": text},
                            {"role": "assistant", "content": "{"},
                        ],
                        continue_final_message=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to("cuda")
                    handle = layers[layer].register_forward_hook(steer)
                    try:
                        with torch.inference_mode():
                            tokens = self.model.generate(
                                **inputs,
                                do_sample=generation["do_sample"],
                                max_new_tokens=generation["max_new_tokens"],
                            )
                    finally:
                        handle.remove()
                    raw = "{" + self.tokenizer.decode(
                        tokens[0, inputs["input_ids"].shape[-1] :],
                        skip_special_tokens=True,
                    ).strip()
                    attempt = {
                        "candidate_id": candidate["id"],
                        "pair_id": case["id"],
                        "multiplier": multiplier,
                    }
                    try:
                        facts = parse(raw)
                    except (TypeError, ValueError) as error:
                        attempt["result"] = "parser_failure"
                        attempt["error"] = str(error)
                    else:
                        fact = candidate["criterion"]
                        target_changed = (
                            facts[fact] is True
                            and case["source_facts"][fact] is False
                        )
                        others_stable = all(
                            facts[name] == case["source_facts"][name]
                            for name in FACTS
                            if name != fact
                        )
                        attempt.update(
                            {
                                "result": "validated",
                                "facts": facts,
                                "target_changed": target_changed,
                                "other_facts_invariant": others_stable,
                            }
                        )
                        if target_changed and others_stable:
                            successful = {
                                "success": True,
                                **attempt,
                                "criterion": fact,
                                "layer": layer,
                                "source_id": case["source_id"],
                                "target_id": case["target_id"],
                                "template_id": case["source_id"].split("-")[1],
                                "policy": (
                                    "add the learned mean direction at the final "
                                    "sequence position on every decode step"
                                ),
                                "baseline_source_facts": case["source_facts"],
                                "baseline_target_facts": case["target_facts"],
                            }
                    attempts.append(attempt)
                    if successful:
                        break
                if successful:
                    break
            if successful:
                break
        if successful is None:
            raise RuntimeError(
                f"no successful held-out causal intervention in {len(attempts)} attempts"
            )

        best_dictionary = max(
            reconstruction,
            key=lambda item: item["held_out_explained_variance"],
        )
        destination = ACTIVATION_DIR / artifact_root / "feature_dictionary.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        layer = best_dictionary["layer"]
        torch.save(
            {
                "facts": FACTS,
                "layer": layer,
                "directions": {
                    fact: tensor.to(torch.bfloat16)
                    for fact, tensor in directions[layer].items()
                },
                "pca_dictionary": dictionaries[layer].to(torch.bfloat16),
            },
            destination,
        )
        activation_volume.commit()
        return {
            "ranked_candidates": candidates,
            "dictionary": {
                "source": "task_specific_topk_linear_autoencoder",
                "rank": DICTIONARY_RANK,
                "active_features": DICTIONARY_ACTIVE_FEATURES,
                "encoder": "PCA projection followed by top-k activation",
                "decoder": "tied PCA dictionary",
                "external_dictionary_used": False,
                "external_dictionary_reason": (
                    "no feature dictionary pinned to the exact Med42 revision "
                    "was available for a parity measurement"
                ),
                "reconstruction_gate": "held_out_explained_variance >= 0.5",
                "reconstruction": reconstruction,
                "artifact": {
                    "volume": "clinical-translator-activations",
                    "path": str(destination.relative_to(ACTIVATION_DIR)),
                    "format": "torch.save",
                    "dtype": "torch.bfloat16",
                    "bytes": destination.stat().st_size,
                    "layer": layer,
                },
            },
            "causal_intervention": successful,
            "intervention_attempts": attempts,
        }


def _smoke(contract: dict[str, Any], output: Path) -> None:
    from clinical_translator.data.generator import generate, write_jsonl
    from clinical_translator.interpretability.capture import smoke_records
    from clinical_translator.models.protocol import parse, prompt, result

    records = smoke_records(generate(contract)[0])
    prompts = [prompt(record["prompt"]) for record in records]
    stored = []
    for role, model in contract["models"].items():
        runner = Translator(
            repo_id=model["repo_id"],
            revision=model["revision"],
        )
        raw_runs = [
            runner.translate.remote(
                prompts,
                contract["reproducibility"]["generation"],
                contract["reproducibility"]["seeds"],
            )
            for _ in range(2)
        ]
        runs = [[parse(raw) for raw in raw_run] for raw_run in raw_runs]
        if runs[0] != runs[1]:
            raise RuntimeError(f"{role} produced nondeterministic smoke outputs")
        for source, facts in zip(records, runs[0], strict=True):
            stored.append(
                result(
                    contract=contract,
                    source=source,
                    role=role,
                    facts=facts,
                )
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(stored, output)
    print(f"{len(stored)} validated outputs written to {output}")


def _capture(contract: dict[str, Any], output: Path) -> None:
    import hashlib
    import json

    from clinical_translator.contracts.validation import reference
    from clinical_translator.data.generator import generate
    from clinical_translator.interpretability.capture import (
        CAPTURE_LAYERS,
        smoke_records,
        validate_manifest,
    )
    from clinical_translator.models.protocol import parse, prompt, prompt_reference

    records = smoke_records(generate(contract)[0])
    prompts = [prompt(record["prompt"]) for record in records]
    prompt_refs = [prompt_reference(record) for record in records]
    model = contract["models"]["primary"]
    runner = Translator(repo_id=model["repo_id"], revision=model["revision"])
    reference_outputs = runner.translate.remote(
        prompts,
        contract["reproducibility"]["generation"],
        contract["reproducibility"]["seeds"],
    )
    contract_ref = reference(contract)
    artifact_root = (
        f"{contract['contract_id']}/{contract_ref.rsplit(':', 1)[-1]}/primary"
    )
    instrumented = runner.capture.remote(
        prompts,
        prompt_refs,
        contract["reproducibility"]["generation"],
        contract["reproducibility"]["seeds"],
        artifact_root,
    )
    if reference_outputs != instrumented["raw_outputs"]:
        raise RuntimeError("instrumented outputs differ from the reference runner")
    manifest = {
        "schema_version": 1,
        "contract_ref": contract_ref,
        "model": {"role": "primary", **model},
        "capture": {
            "backend": "native_pytorch_forward_hooks",
            "layers": list(CAPTURE_LAYERS),
            "token_policy": "last_prompt_token",
            "dtype": "torch.bfloat16",
            "stored_device": "cpu",
            "scope": "six fixed smoke prompts; one token per tensor",
        },
        "parity": {
            "exact_match": True,
            "prompt_count": len(records),
            "output_sha256": [
                hashlib.sha256(raw.encode()).hexdigest()
                for raw in reference_outputs
            ],
        },
        "records": [
            remote | {"facts": parse(raw)}
            for remote, raw in zip(
                instrumented["records"],
                reference_outputs,
                strict=True,
            )
        ],
    }
    validate_manifest(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{len(records)} parity-checked captures written to {output}")


def _evaluate(contract: dict[str, Any], output: Path) -> None:
    import json

    from clinical_translator.contracts.validation import reference
    from clinical_translator.data.generator import generate, write_jsonl
    from clinical_translator.evaluation.evaluator import evaluate, write_evidence
    from clinical_translator.models.protocol import parse, prompt, prompt_reference

    records, pairs = generate(contract)
    output.mkdir(parents=True, exist_ok=True)
    partial = output / "predictions.partial.jsonl"
    final = output / "predictions.jsonl"
    resume = partial if partial.exists() else final
    predictions = (
        [json.loads(line) for line in resume.read_text().splitlines()]
        if resume.exists()
        else []
    )
    records_by_id = {record["id"]: record for record in records}
    for item in predictions:
        model = contract["models"][item["model"]]
        if (
            item["contract_ref"] != reference(contract)
            or item["repo_id"] != model["repo_id"]
            or item["revision"] != model["revision"]
        ):
            raise ValueError("stored predictions use a different contract or checkpoint")
        expected_prompt = prompt_reference(records_by_id[item["prompt_id"]])
        if "prompt" in item and item["prompt"] != expected_prompt:
            raise ValueError("stored predictions use a different prompt protocol")
        item["prompt"] = expected_prompt
    completed = {(item["model"], item["prompt_id"]) for item in predictions}
    for role, model in contract["models"].items():
        runner = Translator(
            repo_id=model["repo_id"],
            revision=model["revision"],
        )
        pending = [
            record for record in records if (role, record["id"]) not in completed
        ]
        for start in range(0, len(pending), 16):
            batch = pending[start : start + 16]
            try:
                raw = runner.translate.remote(
                    [prompt(record["prompt"]) for record in batch],
                    contract["reproducibility"]["generation"],
                    contract["reproducibility"]["seeds"],
                )
                if len(raw) != len(batch):
                    raise RuntimeError("runner returned the wrong number of outputs")
            except Exception as error:
                outcomes = [
                    {
                        "status": "runner_failure",
                        "error": f"{type(error).__name__}: {error}",
                    }
                    for _ in batch
                ]
            else:
                outcomes = []
                for text in raw:
                    try:
                        outcomes.append(
                            {"status": "validated", "prediction": parse(text)}
                        )
                    except (TypeError, ValueError) as error:
                        outcomes.append(
                            {"status": "parser_failure", "error": str(error)}
                        )
            for record, outcome in zip(batch, outcomes, strict=True):
                predictions.append(
                    {
                        "contract_ref": reference(contract),
                        "model": role,
                        "repo_id": model["repo_id"],
                        "revision": model["revision"],
                        "prompt_id": record["id"],
                        "prompt": prompt_reference(record),
                        "kind": record["kind"],
                        "provenance": record["provenance"],
                        "expected": record["facts"],
                        **outcome,
                    }
                )
            write_jsonl(predictions, partial)
            done = sum(item["model"] == role for item in predictions)
            print(f"{role}: {done}/{len(records)}")
    metrics, counterfactuals, failures = evaluate(
        contract,
        records,
        pairs,
        predictions,
    )
    write_evidence(output, metrics, predictions, counterfactuals, failures)
    partial.unlink(missing_ok=True)
    print(f"{len(predictions)} outcomes and {len(failures)} failures written to {output}")


def _discover_features(
    contract: dict[str, Any],
    predictions_path: Path,
    output: Path,
) -> None:
    import json

    from clinical_translator.contracts.validation import reference
    from clinical_translator.data.generator import generate
    from clinical_translator.interpretability.capture import CAPTURE_LAYERS
    from clinical_translator.interpretability.features import (
        HELD_OUT_TEMPLATES,
        TRAIN_TEMPLATES,
        feature_records,
        held_out_interventions,
        validate_report,
    )
    from clinical_translator.models.protocol import prompt

    generated, pairs = generate(contract)
    records = feature_records(generated)
    predictions = [
        json.loads(line) for line in predictions_path.read_text().splitlines()
    ]
    cases = held_out_interventions(pairs, predictions)
    model = contract["models"]["primary"]
    contract_ref = reference(contract)
    runner = Translator(repo_id=model["repo_id"], revision=model["revision"])
    discovered = runner.discover_features.remote(
        [prompt(record["prompt"]) for record in records],
        [
            {
                "id": record["id"],
                "template_id": record["provenance"]["template_id"],
                "facts": record["facts"],
            }
            for record in records
        ],
        cases,
        contract["reproducibility"]["generation"],
        contract["reproducibility"]["seeds"],
        f"{contract['contract_id']}/{contract_ref.rsplit(':', 1)[-1]}/features",
    )
    report = {
        "schema_version": 1,
        "contract_ref": contract_ref,
        "model": {"role": "primary", **model},
        "capture": {
            "backend": "native_pytorch_forward_hooks",
            "layers": list(CAPTURE_LAYERS),
            "token_policy": "last_prompt_token",
            "dtype": "torch.bfloat16",
        },
        "split": {
            "policy": "paraphrase_template",
            "train_templates": list(TRAIN_TEMPLATES),
            "held_out_templates": list(HELD_OUT_TEMPLATES),
            "train_prompts": 192,
            "held_out_prompts": 64,
        },
        **discovered,
        "claim_boundary": (
            "Probe decodability and one intervention nominate candidates; "
            "they are not circuit-level causal explanations."
        ),
    }
    validate_report(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{len(report['ranked_candidates'])} candidates written to {output}")


@app.local_entrypoint()
def main(
    contract_path: str = "configs/contracts/curb65-llama-v2.json",
    download: bool = False,
    smoke: bool = False,
    capture: bool = False,
    discover_features: bool = False,
    evaluate_all: bool = False,
    output: str = "evidence/goal-04-smoke.jsonl",
    capture_output: str = "evidence/goal-06/manifest.json",
    feature_output: str = "evidence/goal-07/report.json",
    predictions: str = "evidence/goal-05/predictions.jsonl",
) -> None:
    from clinical_translator.contracts.validation import load

    contract = load(contract_path)
    if smoke:
        _smoke(contract, Path(output))
        return
    if capture:
        _capture(contract, Path(capture_output))
        return
    if discover_features:
        _discover_features(contract, Path(predictions), Path(feature_output))
        return
    if evaluate_all:
        destination = (
            Path("evidence/goal-05") if output == "evidence/goal-04-smoke.jsonl"
            else Path(output)
        )
        _evaluate(contract, destination)
        return
    function = cache_model if download else check_access
    for model in contract["models"].values():
        result = function.remote(model["repo_id"], model["revision"])
        if not download and result != model["revision"]:
            raise RuntimeError(f"revision mismatch for {model['repo_id']}")
        print(result)
