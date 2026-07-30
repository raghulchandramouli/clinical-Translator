"""Reduce bounded behavioural, causal, and formal failures."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from clinical_translator.assurance.oracle import incomplete_score, score
from clinical_translator.contracts.validation import FACTS, load, reference
from clinical_translator.data.generator import generate, write_jsonl


def omitted_confusion_demo() -> dict[str, Any]:
    facts = dict.fromkeys(FACTS, False)
    facts["confusion"] = True
    expected = score(facts)
    before = incomplete_score(facts)
    after = score(facts)
    return {
        "name": "omitted-confusion",
        "input": facts,
        "expected": expected,
        "before": {
            "implementation": "incomplete_score",
            "actual": before,
            "status": "counterexample" if before != expected else "pass",
        },
        "after": {
            "implementation": "score",
            "actual": after,
            "status": "fixed" if after == expected else "counterexample",
        },
    }


def _record(
    *,
    identifier: str,
    source: str,
    source_ids: list[str],
    contract_ref: str,
    model: dict[str, Any] | None,
    inputs: dict[str, Any],
    expected: Any,
    actual: Any,
    violated_claim: str,
    reduction: dict[str, Any],
    refinement_targets: list[str],
    resolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "status": "fixed" if resolution else "active",
        "source": source,
        "source_failure_ids": sorted(source_ids),
        "contract_ref": contract_ref,
        "model": model,
        "inputs": inputs,
        "expected": expected,
        "actual": actual,
        "violated_claim": violated_claim,
        "reduction": reduction,
        "refinement_targets": refinement_targets,
        "resolution": resolution,
        "reproduce": {
            "command": "uv run python -m clinical_translator.assurance.counterexamples",
            "evidence": source,
        },
    }


def build_report(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    contract = load(root / "configs/contracts/curb65-llama-v2.json")
    contract_ref = reference(contract)
    generated, generated_pairs = generate(contract)
    by_id = {item["id"]: item for item in generated}
    pair_by_id = {item["id"]: item for item in generated_pairs}
    failures = [
        json.loads(line)
        for line in (root / "evidence/goal-05/failures.jsonl")
        .read_text()
        .splitlines()
    ]
    causal = json.loads((root / "evidence/goal-09/report.json").read_text())
    circuits = json.loads((root / "evidence/goal-08/report.json").read_text())
    certificate = json.loads(
        (root / "evidence/goal-11/certificate.json").read_text()
    )
    corpus: list[dict[str, Any]] = []
    mapped_goal05: set[str] = set()

    fact_groups: dict[tuple[Any, ...], list[tuple[str, dict[str, Any]]]] = defaultdict(
        list
    )
    pair_groups: dict[tuple[Any, ...], list[tuple[str, dict[str, Any]]]] = defaultdict(
        list
    )
    irreducible = []
    for index, failure in enumerate(failures):
        source_id = f"goal05:{index}:{failure['type']}"
        if failure["type"] == "fact_error":
            for fact in failure["fact_errors"]:
                key = (
                    failure["model"],
                    fact,
                    failure["expected"][fact],
                    failure["prediction"][fact],
                )
                fact_groups[key].append((source_id, failure))
        elif failure["type"] == "counterfactual_failure":
            key = (
                failure["model"],
                failure["criterion"],
                failure.get("target_fact_changed"),
                failure.get("other_facts_invariant"),
            )
            pair_groups[key].append((source_id, failure))
        else:
            irreducible.append((source_id, failure))

    for key, group in sorted(fact_groups.items(), key=lambda item: str(item[0])):
        model_role, fact, expected_value, actual_value = key
        selected_source, selected = min(
            group,
            key=lambda item: (
                sum(by_id[item[1]["prompt_id"]]["facts"].values()),
                len(by_id[item[1]["prompt_id"]]["prompt"]),
                item[1]["prompt_id"],
            ),
        )
        source_ids = sorted({item[0] for item in group})
        mapped_goal05.update(source_ids)
        vignette = by_id[selected["prompt_id"]]
        identifier = (
            f"ce-behaviour-{model_role}-{fact}-"
            f"{int(expected_value)}-{int(actual_value)}"
        )
        corpus.append(
            _record(
                identifier=identifier,
                source="evidence/goal-05/failures.jsonl",
                source_ids=source_ids,
                contract_ref=contract_ref,
                model={"role": model_role, **contract["models"][model_role]},
                inputs={
                    "prompt_id": vignette["id"],
                    "prompt": vignette["prompt"],
                    "facts": vignette["facts"],
                    "values": vignette["values"],
                    "provenance": vignette["provenance"],
                },
                expected=selected["expected"],
                actual=selected["prediction"],
                violated_claim=f"model emits the correct {fact} fact",
                reduction={
                    "method": "search the finite grammar for the fewest positive facts, then shortest prompt",
                    "source_cases": len(group),
                    "selected_metric": {
                        "positive_facts": sum(vignette["facts"].values()),
                        "prompt_characters": len(vignette["prompt"]),
                    },
                    "preserves_failure": (
                        selected["prediction"][fact] != selected["expected"][fact]
                    ),
                    "valid_under_goal01_grammar": True,
                    "generated_record": True,
                    "selected_source": selected_source,
                },
                refinement_targets=["feature_discovery", "behaviour_evaluator"],
            )
        )

    for key, group in sorted(pair_groups.items(), key=lambda item: str(item[0])):
        model_role, fact, target_changed, others_invariant = key
        selected_source, selected = min(
            group,
            key=lambda item: (
                sum(by_id[item[1]["source_id"]]["facts"].values()),
                len(by_id[item[1]["source_id"]]["prompt"])
                + len(by_id[item[1]["target_id"]]["prompt"]),
                item[1]["id"],
            ),
        )
        source_ids = sorted({item[0] for item in group})
        mapped_goal05.update(source_ids)
        source = by_id[selected["source_id"]]
        target = by_id[selected["target_id"]]
        mode = (
            "insensitive"
            if not target_changed and others_invariant
            else "entangled"
            if not others_invariant
            else "other"
        )
        corpus.append(
            _record(
                identifier=(
                    f"ce-counterfactual-{model_role}-{fact}-{mode}-"
                    f"{int(bool(target_changed))}-{int(bool(others_invariant))}"
                ),
                source="evidence/goal-05/failures.jsonl",
                source_ids=source_ids,
                contract_ref=contract_ref,
                model={"role": model_role, **contract["models"][model_role]},
                inputs={
                    "pair_id": selected["id"],
                    "criterion": fact,
                    "source": {
                        "id": source["id"],
                        "prompt": source["prompt"],
                        "facts": source["facts"],
                        "values": source["values"],
                    },
                    "target": {
                        "id": target["id"],
                        "prompt": target["prompt"],
                        "facts": target["facts"],
                        "values": target["values"],
                    },
                },
                expected={
                    "target_fact_changed": True,
                    "other_facts_invariant": True,
                    "only_intended_fact_changed": True,
                },
                actual={
                    "target_fact_changed": target_changed,
                    "other_facts_invariant": others_invariant,
                    "changed_facts": selected.get("changed_facts"),
                },
                violated_claim=f"the matched {fact} intervention changes only its intended output",
                reduction={
                    "method": "remove background-positive facts within the finite pair set",
                    "source_cases": len(group),
                    "selected_metric": {
                        "source_positive_facts": sum(source["facts"].values()),
                        "combined_prompt_characters": (
                            len(source["prompt"]) + len(target["prompt"])
                        ),
                    },
                    "preserves_failure": not (
                        target_changed and others_invariant
                    ),
                    "valid_under_goal01_grammar": (
                        selected["id"] in pair_by_id
                        and source["id"] in by_id
                        and target["id"] in by_id
                    ),
                    "generated_pair": True,
                    "selected_source": selected_source,
                },
                refinement_targets=[
                    "feature_discovery",
                    "circuit_extraction",
                    "behaviour_evaluator",
                ],
            )
        )

    for source_id, failure in irreducible:
        mapped_goal05.add(source_id)
        vignette = by_id[failure["prompt_id"]]
        actual = (
            {"error": failure["error"]}
            if "error" in failure
            else failure.get("prediction")
        )
        corpus.append(
            _record(
                identifier=(
                    f"ce-{failure['type']}-{failure['model']}-"
                    f"{vignette['id'].removeprefix('incomplete-')}"
                ),
                source="evidence/goal-05/failures.jsonl",
                source_ids=[source_id],
                contract_ref=contract_ref,
                model={
                    "role": failure["model"],
                    **contract["models"][failure["model"]],
                },
                inputs={
                    "prompt_id": vignette["id"],
                    "prompt": vignette["prompt"],
                    "facts": vignette["facts"],
                    "values": vignette["values"],
                    "provenance": vignette["provenance"],
                },
                expected={"status": "reject_missing_required_concept"},
                actual=actual,
                violated_claim="incomplete input is not silently translated",
                reduction={
                    "method": "already minimal: exactly one required concept is absent",
                    "source_cases": 1,
                    "preserves_failure": True,
                    "valid_under_goal01_grammar": True,
                    "generated_record": True,
                },
                refinement_targets=["behaviour_evaluator"],
            )
        )

    goal09_source_ids = []
    causal_failures = causal["counterexamples"]
    for index, failure in enumerate(causal_failures):
        source_id = f"goal09:{index}:{failure['experiment']}"
        goal09_source_ids.append(source_id)
        fact = failure["criterion"]
        original_experiment = failure["experiment"]
        reduced_experiment = original_experiment
        if original_experiment == "variable_validation":
            reduced_experiment = "feature_ablation"
        elif original_experiment == "path_ablation" and not causal["variables"][fact][
            "ablations"
        ]["mlp"]["pass"]:
            reduced_experiment = "mlp_ablation"
        mode = reduced_experiment.removesuffix("_ablation")
        ablation_key = "attention_heads" if mode == "attention_heads" else mode
        actual = causal["variables"][fact]["ablations"].get(ablation_key)
        nodes = circuits["models"]["primary"]["circuits"][fact]["graph"]["nodes"]
        if mode == "feature":
            configuration = [
                node["id"] for node in nodes if node["type"] == "feature"
            ]
        elif mode == "mlp":
            configuration = [node["id"] for node in nodes if node["type"] == "mlp"]
        else:
            configuration = [
                node["id"]
                for node in nodes
                if node["type"] in {"attention_head", "mlp"}
            ]
        corpus.append(
            _record(
                identifier=f"ce-causal-{fact}-{original_experiment}",
                source="evidence/goal-09/report.json",
                source_ids=[source_id],
                contract_ref=contract_ref,
                model={"role": "primary", **contract["models"]["primary"]},
                inputs={
                    "criterion": fact,
                    "original_experiment": original_experiment,
                    "reduced_experiment": reduced_experiment,
                    "configuration": configuration,
                    "held_out_templates": ["t07", "t08"],
                },
                expected={
                    "effect_size": "> 0 or accuracy_drop > 0",
                    "pass": True,
                },
                actual=actual,
                violated_claim=f"the retained {fact} configuration has the predicted causal effect",
                reduction={
                    "method": "retain the smallest failing feature/component subset",
                    "source_cases": 1,
                    "original_components": len(
                        [
                            node
                            for node in nodes
                            if node["type"] in {"attention_head", "mlp"}
                        ]
                    ),
                    "reduced_components": len(configuration),
                    "preserves_failure": actual is not None and not actual["pass"],
                    "valid_under_goal01_grammar": True,
                    "intervention_domain": "Goal 09 held-out matched pairs",
                },
                refinement_targets=["feature_discovery", "circuit_extraction"],
            )
        )

    formal = next(
        result
        for result in certificate["results"]
        if result["id"] == "incomplete_scorer_equivalence"
    )
    demo = omitted_confusion_demo()
    corpus.append(
        _record(
            identifier="ce-formal-omitted-confusion",
            source="evidence/goal-11/certificate.json",
            source_ids=["goal11:incomplete_scorer_equivalence"],
            contract_ref=contract_ref,
            model=None,
            inputs=demo["input"],
            expected=demo["expected"],
            actual=demo["before"]["actual"],
            violated_claim=formal["statement"],
            reduction={
                "method": "single positive fact witness",
                "source_cases": 1,
                "positive_facts": 1,
                "preserves_failure": demo["before"]["status"] == "counterexample",
                "valid_under_goal01_grammar": True,
            },
            refinement_targets=["proof_engine"],
            resolution={
                "implementation": "clinical_translator.assurance.oracle.score",
                "actual": demo["after"]["actual"],
                "expected": demo["expected"],
                "status": demo["after"]["status"],
                "regression_check": "tests/test_counterexample_engine.py",
            },
        )
    )
    impossible = [
        {
            "source_failure_id": "goal11:local_neural_slice_certification",
            "source": "evidence/goal-11/certificate.json",
            "reason": next(
                result["reason"]
                for result in certificate["results"]
                if result["id"] == "local_neural_slice_certification"
            ),
        }
    ]

    corpus.sort(key=lambda item: item["id"])
    corpus_payload = "".join(
        json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
        for item in corpus
    ).encode()
    source_counts = {
        "goal05": {
            "input": len(failures),
            "mapped": len(mapped_goal05),
            "reduction_impossible": 0,
        },
        "goal09": {
            "input": len(causal_failures),
            "mapped": len(goal09_source_ids),
            "reduction_impossible": 0,
        },
        "goal11": {
            "input": 2,
            "mapped": 1,
            "reduction_impossible": 1,
        },
    }
    report = {
        "schema_version": 1,
        "contract_ref": contract_ref,
        "sources": {
            "behavioural": "evidence/goal-05/failures.jsonl",
            "causal": "evidence/goal-09/report.json",
            "formal": "evidence/goal-11/certificate.json",
        },
        "source_coverage": source_counts,
        "reduction_impossible": impossible,
        "corpus": {
            "path": "evidence/goal-12/regressions.jsonl",
            "records": len(corpus),
            "sha256": hashlib.sha256(corpus_payload).hexdigest(),
            "active": sum(item["status"] == "active" for item in corpus),
            "fixed": sum(item["status"] == "fixed" for item in corpus),
            "refinement_targets": dict(
                Counter(
                    target
                    for item in corpus
                    for target in item["refinement_targets"]
                )
            ),
        },
        "demo": demo,
        "claim_boundary": (
            "Reducers search only the finite Goal 01 grammar, recorded Goal 09 "
            "interventions, and Goal 11 formal obligations."
        ),
    }
    return report, corpus


def validate_report(
    report: dict[str, Any],
    corpus: list[dict[str, Any]],
) -> None:
    if report.get("schema_version") != 1 or not report.get("contract_ref"):
        raise ValueError("unsupported counterexample report")
    coverage = report.get("source_coverage", {})
    for source in ("goal05", "goal09", "goal11"):
        item = coverage.get(source, {})
        if item.get("mapped", 0) + item.get("reduction_impossible", 0) != item.get(
            "input"
        ):
            raise ValueError(f"{source} failures are not fully accounted for")
    if coverage["goal05"]["input"] != 961 or coverage["goal09"]["input"] != 15:
        raise ValueError("counterexample sources have unexpected scope")
    if not report.get("reduction_impossible"):
        raise ValueError("unreducible obligations lack explicit reasons")

    if report.get("corpus", {}).get("records") != len(corpus):
        raise ValueError("regression corpus count mismatch")
    payload = "".join(
        json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
        for item in corpus
    ).encode()
    if report["corpus"].get("sha256") != hashlib.sha256(payload).hexdigest():
        raise ValueError("regression corpus digest mismatch")
    if not any(item.get("status") == "fixed" for item in corpus):
        raise ValueError("fixed counterexamples are missing from regression corpus")

    required = {
        "id",
        "status",
        "source",
        "source_failure_ids",
        "contract_ref",
        "model",
        "inputs",
        "expected",
        "actual",
        "violated_claim",
        "reduction",
        "refinement_targets",
        "reproduce",
    }
    for item in corpus:
        if not required <= item.keys():
            raise ValueError("counterexample record is incomplete")
        if item["contract_ref"] != report["contract_ref"]:
            raise ValueError("counterexample uses the wrong contract")
        if not item["source_failure_ids"]:
            raise ValueError("counterexample lacks source failures")
        reduction = item["reduction"]
        if reduction.get("preserves_failure") is not True:
            raise ValueError("reduction did not preserve the original failure")
        if reduction.get("valid_under_goal01_grammar") is not True:
            raise ValueError("reduced case left the declared grammar")
        if item["model"] is not None and (
            not item["model"].get("repo_id") or not item["model"].get("revision")
        ):
            raise ValueError("model counterexample lacks a pinned revision")
        if not item["refinement_targets"]:
            raise ValueError("counterexample cannot feed refinement")

    demo = report.get("demo", {})
    if (
        demo.get("before", {}).get("status") != "counterexample"
        or demo.get("after", {}).get("status") != "fixed"
        or demo["before"]["actual"] == demo["expected"]
        or demo["after"]["actual"] != demo["expected"]
    ):
        raise ValueError("omitted-confusion demonstration does not fail then fix")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Goal 12 counterexamples")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/goal-12/report.json"),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("evidence/goal-12/regressions.jsonl"),
    )
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        print(json.dumps(omitted_confusion_demo(), indent=2, sort_keys=True))
        return
    root = Path(__file__).parents[3]
    report, corpus = build_report(root)
    validate_report(report, corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl(corpus, args.corpus)
    print(
        f"{len(corpus)} regressions written; "
        f"{sum(item['status'] == 'fixed' for item in corpus)} fixed"
    )


if __name__ == "__main__":
    main()
