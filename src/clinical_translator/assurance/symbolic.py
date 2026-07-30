"""Smallest symbolic program supported by the causal CURB-65 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from clinical_translator.assurance.oracle import combinations, score
from clinical_translator.contracts.validation import FACTS

EXPRESSION = "score = " + " + ".join(FACTS)


def build_report(
    causal: dict[str, Any],
    circuits: dict[str, Any],
) -> dict[str, Any]:
    circuit_variables = circuits["models"]["primary"]["circuits"]
    variables = {}
    for fact in FACTS:
        evidence = causal.get("variables", {}).get(fact)
        circuit = circuit_variables.get(fact)
        status = evidence.get("status", "unknown") if evidence else "unknown"
        nodes = circuit.get("graph", {}).get("nodes", []) if circuit else []
        variables[fact] = {
            "symbol": fact,
            "status": status,
            "causal_evidence": f"evidence/goal-09/report.json#/variables/{fact}",
            "circuit_state": {
                "feature": next(
                    (node["id"] for node in nodes if node["type"] == "feature"),
                    None,
                ),
                "retained_components": [
                    node["id"]
                    for node in nodes
                    if node["type"] in {"attention_head", "mlp"}
                ],
                "readout": "true iff the structured fact-token margin is >= 0",
            },
            "alignment": {
                "mapping": (
                    "matched target residual -> corresponding Boolean fact token"
                    if evidence
                    else None
                ),
                "eligible_pairs": (
                    evidence["interchange"]["eligible_pairs"] if evidence else 0
                ),
                "predicted_direction_rate": (
                    evidence["interchange"]["predicted_direction_rate"]
                    if evidence
                    else None
                ),
                "unrelated_output_stability": (
                    evidence["interchange"]["unrelated_output_stability"]
                    if evidence
                    else None
                ),
                "reason": (
                    "mixed causal evidence; mapping remains a candidate"
                    if status == "mixed"
                    else "causal validation passed"
                    if status == "pass"
                    else "causal evidence is insufficient"
                ),
            },
        }

    truth_table = []
    for facts in combinations():
        symbolic_score = sum(facts[name] for name in FACTS)
        oracle_score = score(facts)
        truth_table.append(
            {
                "facts": facts,
                "symbolic_score": symbolic_score,
                "oracle_score": oracle_score,
                "match": symbolic_score == oracle_score,
            }
        )

    incomplete = [fact for fact, item in variables.items() if item["status"] != "pass"]
    counterexamples = []
    for fact in incomplete:
        witness = dict.fromkeys(FACTS, False)
        witness[fact] = True
        counterexamples.append(
            {
                "variable": fact,
                "status": variables[fact]["status"],
                "witness": witness,
                "oracle_score": score(witness),
                "aligned_score": None,
                "reason": "incomplete neural alignment; score is not forced",
            }
        )

    return {
        "schema_version": 1,
        "variables": variables,
        "candidate_mechanism": {
            "expression": EXPRESSION,
            "inputs": list(FACTS),
            "hidden_variables": [],
            "implementation": "sum(bool(fact) for fact in the five declared inputs)",
        },
        "oracle_alignment": {
            "python": "clinical_translator.assurance.oracle.score",
            "lean": "ClinicalTranslator.score",
            "theorem": "ClinicalTranslator.score_correct",
            "combinations_checked": len(truth_table),
            "matches": all(item["match"] for item in truth_table),
            "truth_table": truth_table,
        },
        "neural_alignment": {
            "complete": not incomplete,
            "status": "pass" if not incomplete else "incomplete",
            "aligned_score": None,
            "incomplete_variables": incomplete,
        },
        "counterexamples": counterexamples,
        "claim_boundary": (
            "The five-input sum is the Goal 03 oracle. Neural mappings remain "
            "limited to Goal 09's tested synthetic interventions."
        ),
    }


def validate_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != 1:
        raise ValueError("unsupported symbolic-mechanism report")
    variables = report.get("variables", {})
    if set(variables) != set(FACTS):
        raise ValueError("symbolic variables must be exactly the five CURB-65 facts")
    for fact, variable in variables.items():
        if variable.get("symbol") != fact:
            raise ValueError("symbolic mapping uses the wrong fact")
        if variable.get("status") not in {"pass", "mixed", "fail", "unknown"}:
            raise ValueError("symbolic variable lacks explicit alignment status")
        if variable.get("causal_evidence") != (
            f"evidence/goal-09/report.json#/variables/{fact}"
        ):
            raise ValueError("symbolic variable does not point to Goal 09 evidence")
        state = variable.get("circuit_state", {})
        if not state.get("feature") or not state.get("retained_components"):
            raise ValueError("symbolic variable lacks an explicit circuit-state mapping")

    mechanism = report.get("candidate_mechanism", {})
    if mechanism.get("expression") != EXPRESSION:
        raise ValueError("unexpected symbolic mechanism")
    if mechanism.get("inputs") != list(FACTS):
        raise ValueError("symbolic mechanism inputs differ from the frozen schema")
    if mechanism.get("hidden_variables") != []:
        raise ValueError("unsupported hidden variable asserted")

    oracle = report.get("oracle_alignment", {})
    if (
        oracle.get("python") != "clinical_translator.assurance.oracle.score"
        or oracle.get("theorem") != "ClinicalTranslator.score_correct"
        or oracle.get("combinations_checked") != 32
        or oracle.get("matches") is not True
        or len(oracle.get("truth_table", [])) != 32
        or not all(item.get("match") is True for item in oracle["truth_table"])
    ):
        raise ValueError("symbolic mechanism does not match the Goal 03 scorer")

    incomplete = [
        fact for fact, variable in variables.items() if variable["status"] != "pass"
    ]
    alignment = report.get("neural_alignment", {})
    counterexamples = report.get("counterexamples", [])
    if incomplete:
        if (
            alignment.get("complete") is not False
            or alignment.get("aligned_score") is not None
            or set(alignment.get("incomplete_variables", [])) != set(incomplete)
            or {item.get("variable") for item in counterexamples} != set(incomplete)
            or any(item.get("aligned_score") is not None for item in counterexamples)
        ):
            raise ValueError("incomplete alignment was forced into a symbolic match")
    elif alignment.get("complete") is not True or counterexamples:
        raise ValueError("complete alignment has inconsistent counterexamples")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Goal 10 mechanism")
    parser.add_argument(
        "--causal",
        type=Path,
        default=Path("evidence/goal-09/report.json"),
    )
    parser.add_argument(
        "--circuits",
        type=Path,
        default=Path("evidence/goal-08/report.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/goal-10/report.json"),
    )
    args = parser.parse_args()
    report = build_report(
        json.loads(args.causal.read_text()),
        json.loads(args.circuits.read_text()),
    )
    validate_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"symbolic mechanism and {len(report['counterexamples'])} counterexamples written")


if __name__ == "__main__":
    main()
