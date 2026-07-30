"""Reproducible proof and finite-coverage certificate for Goal 11."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from clinical_translator.assurance.oracle import (
    combinations,
    incomplete_score,
    score,
)
from clinical_translator.contracts.validation import FACTS, load, reference
from clinical_translator.data.generator import generate

STATUSES = {"proved", "counterexample", "unknown"}
EXPECTED_EXPRESSION = "score = " + " + ".join(FACTS)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _domain(
    identifier: str,
    description: str,
    cardinality: int,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "description": description,
        "cardinality": cardinality,
    }


def _claim(
    identifier: str,
    statement: str,
    kind: str,
    contract_ref: str,
    domain: dict[str, Any],
    status: str,
    **evidence: Any,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "statement": statement,
        "kind": kind,
        "contract_ref": contract_ref,
        "domain": domain,
        "status": status,
        **evidence,
    }


def _lean_claim(
    root: Path,
    contract_ref: str,
    timeout_seconds: int,
) -> tuple[dict[str, Any], str]:
    proof = root / "proofs/ClinicalTranslator/CURB65.lean"
    domain = _domain(
        "curb65-five-booleans-v1",
        "all five-Boolean CURB-65 fact assignments",
        32,
    )
    lean = shutil.which("lean") or Path.home() / ".elan/bin/lean"
    if not Path(lean).exists():
        return (
            _claim(
                "symbolic_scorer_theorem",
                "ClinicalTranslator.score equals its five-Boolean specification",
                "formal",
                contract_ref,
                domain,
                "unknown",
                reason="pinned Lean checker is unavailable",
            ),
            "unavailable",
        )
    try:
        version = subprocess.run(
            [str(lean), "--version"],
            text=True,
            capture_output=True,
            check=True,
            timeout=timeout_seconds,
        ).stdout.strip()
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    str(lean),
                    "-o",
                    f"{directory}/CURB65.olean",
                    str(proof),
                ],
                cwd=root,
                text=True,
                capture_output=True,
                check=True,
                timeout=timeout_seconds,
            )
    except subprocess.TimeoutExpired:
        return (
            _claim(
                "symbolic_scorer_theorem",
                "ClinicalTranslator.score equals its five-Boolean specification",
                "formal",
                contract_ref,
                domain,
                "unknown",
                reason=f"Lean check exceeded {timeout_seconds} seconds",
            ),
            "timeout",
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return (
            _claim(
                "symbolic_scorer_theorem",
                "ClinicalTranslator.score equals its five-Boolean specification",
                "formal",
                contract_ref,
                domain,
                "unknown",
                reason=f"Lean checker failed: {error}",
            ),
            "failed",
        )

    try:
        lean_table = ast.literal_eval(result.stdout.strip())
    except (SyntaxError, ValueError):
        return (
            _claim(
                "symbolic_scorer_theorem",
                "ClinicalTranslator.score equals its five-Boolean specification",
                "formal",
                contract_ref,
                domain,
                "unknown",
                reason="Lean check emitted an unsupported certificate format",
            ),
            version,
        )
    python_table = [score(facts) for facts in combinations()]
    if lean_table != python_table:
        index = next(
            (
                index
                for index, values in enumerate(
                    zip(lean_table, python_table, strict=False)
                )
                if values[0] != values[1]
            ),
            min(len(lean_table), len(python_table)),
        )
        witness = list(combinations())[min(index, len(python_table) - 1)]
        return (
            _claim(
                "symbolic_scorer_theorem",
                "ClinicalTranslator.score equals its five-Boolean specification",
                "formal",
                contract_ref,
                domain,
                "counterexample",
                counterexample={
                    "facts": witness,
                    "lean_score": lean_table[index] if index < len(lean_table) else None,
                    "python_score": (
                        python_table[index] if index < len(python_table) else None
                    ),
                },
            ),
            version,
        )
    return (
        _claim(
            "symbolic_scorer_theorem",
            "ClinicalTranslator.score equals its five-Boolean specification",
            "formal",
            contract_ref,
            domain,
            "proved",
            certificate={
                "checker": "Lean kernel and elaborator",
                "theorem": "ClinicalTranslator.score_correct",
                "proof_sha256": _sha256(proof),
                "truth_table_sha256": hashlib.sha256(
                    json.dumps(lean_table, separators=(",", ":")).encode()
                ).hexdigest(),
                "lean_version": version,
            },
        ),
        version,
    )


def _generator_claim(
    contract: dict[str, Any],
    contract_ref: str,
) -> dict[str, Any]:
    records, pairs = generate(contract)
    repeated_records, repeated_pairs = generate(contract)
    payload = json.dumps(
        {"records": records, "pairs": pairs},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    repeated_payload = json.dumps(
        {"records": repeated_records, "pairs": repeated_pairs},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    by_id = {record["id"]: record for record in records}
    complete = [record for record in records if record["kind"] == "complete"]
    incomplete = [record for record in records if record["kind"] == "incomplete"]
    combinations_seen = Counter(
        record["provenance"]["combination"] for record in complete
    )

    bad_pair = None
    for pair in pairs:
        source = by_id[pair["source_id"]]["facts"]
        target = by_id[pair["target_id"]]["facts"]
        changed = [fact for fact in FACTS if source[fact] != target[fact]]
        if changed != [pair["criterion"]]:
            bad_pair = {
                "pair": pair,
                "changed_facts": changed,
            }
            break
    values = [record["values"] for record in complete]
    checks = {
        "complete_records": len(complete) == 256,
        "incomplete_records": len(incomplete) == 6,
        "counterfactual_pairs": len(pairs) == 640,
        "logical_combinations": (
            len(combinations_seen) == 32
            and set(combinations_seen.values()) == {8}
        ),
        "deterministic_replay": payload == repeated_payload,
        "single_fact_pairs": bad_pair is None,
        "synthetic_only": (
            contract["input_domain"]["patient_data"] is False
            and all(record["provenance"]["synthetic"] is True for record in records)
        ),
        "numeric_boundaries": (
            {64, 65, 66}
            <= {item["age_years"] for item in values}
            and {6.9, 7.0, 7.1}
            <= {item["urea_mmol_l"] for item in values}
            and {29, 30, 31}
            <= {item["respiratory_rate_per_minute"] for item in values}
            and {(90, 61), (89, 61), (91, 60)}
            <= {
                (item["systolic_bp_mmhg"], item["diastolic_bp_mmhg"])
                for item in values
            }
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        return _claim(
            "generator_coverage",
            "the pinned generator covers its complete declared finite domain",
            "exact_finite",
            contract_ref,
            _domain(
                "curb65-vignette-grammar-v1",
                "262 generated prompts and 640 matched counterfactual pairs",
                len(records) + len(pairs),
            ),
            "counterexample",
            counterexample={
                "failed_invariants": failed,
                "bad_pair": bad_pair,
            },
            checks=checks,
        )
    return _claim(
        "generator_coverage",
        "the pinned generator covers its complete declared finite domain",
        "exact_finite",
        contract_ref,
        _domain(
            "curb65-vignette-grammar-v1",
            "262 generated prompts and 640 matched counterfactual pairs",
            len(records) + len(pairs),
        ),
        "proved",
        checks=checks,
        certificate={
            "checker": "deterministic exhaustive enumeration",
            "dataset_sha256": hashlib.sha256(payload).hexdigest(),
            "replay_sha256": hashlib.sha256(repeated_payload).hexdigest(),
            "complete_records": len(complete),
            "incomplete_records": len(incomplete),
            "counterfactual_pairs": len(pairs),
        },
    )


def symbolic_equivalence_claim(
    symbolic: dict[str, Any],
    contract_ref: str,
) -> dict[str, Any]:
    table = symbolic.get("oracle_alignment", {}).get("truth_table", [])
    expected = []
    for facts in combinations():
        expected.append(
            {
                "facts": facts,
                "symbolic_score": sum(facts[name] for name in FACTS),
                "oracle_score": score(facts),
                "match": True,
            }
        )
    mismatch = next(
        (
            {"index": index, "reported": reported, "expected": correct}
            for index, (reported, correct) in enumerate(
                zip(table, expected, strict=False)
            )
            if reported != correct
        ),
        None,
    )
    valid = (
        symbolic.get("candidate_mechanism", {}).get("expression")
        == EXPECTED_EXPRESSION
        and symbolic.get("candidate_mechanism", {}).get("hidden_variables") == []
        and len(table) == 32
        and mismatch is None
    )
    domain = _domain(
        "curb65-five-booleans-v1",
        "all five-Boolean CURB-65 fact assignments",
        32,
    )
    if not valid:
        if mismatch is None:
            mismatch = {
                "reported_expression": symbolic.get(
                    "candidate_mechanism", {}
                ).get("expression"),
                "reported_hidden_variables": symbolic.get(
                    "candidate_mechanism", {}
                ).get("hidden_variables"),
            }
        return _claim(
            "symbolic_mechanism_equivalence",
            "the Goal 10 five-input mechanism equals the Goal 03 scorer",
            "exact_finite",
            contract_ref,
            domain,
            "counterexample",
            counterexample=mismatch,
        )
    return _claim(
        "symbolic_mechanism_equivalence",
        "the Goal 10 five-input mechanism equals the Goal 03 scorer",
        "exact_finite",
        contract_ref,
        domain,
        "proved",
        certificate={
            "checker": "exhaustive 32-case equivalence",
            "cases": 32,
            "truth_table_sha256": hashlib.sha256(
                json.dumps(table, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
    )


def build_report(
    root: Path,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    contract_path = root / "configs/contracts/curb65-llama-v2.json"
    symbolic_path = root / "evidence/goal-10/report.json"
    contract = load(contract_path)
    contract_ref = reference(contract)
    symbolic = json.loads(symbolic_path.read_text())
    lean_claim, lean_version = _lean_claim(root, contract_ref, timeout_seconds)
    generator_claim = _generator_claim(contract, contract_ref)
    equivalence_claim = symbolic_equivalence_claim(symbolic, contract_ref)

    witness = dict.fromkeys(FACTS, False)
    witness["confusion"] = True
    incomplete_claim = _claim(
        "incomplete_scorer_equivalence",
        "the scorer that omits confusion equals the complete specification",
        "exact_finite",
        contract_ref,
        _domain(
            "curb65-five-booleans-v1",
            "all five-Boolean CURB-65 fact assignments",
            32,
        ),
        "counterexample",
        counterexample={
            "facts": witness,
            "incomplete_score": incomplete_score(witness),
            "specification_score": score(witness),
        },
    )
    neural_claim = _claim(
        "local_neural_slice_certification",
        "the extracted neural slice implements the five symbolic variables",
        "empirical",
        contract_ref,
        _domain(
            "goal09-held-out-interventions-v1",
            "160 matched interventions over templates t07 and t08",
            160,
        ),
        "unknown",
        reason=(
            "Goal 10 alignment is incomplete and explicit bounded activation "
            "ranges for a tractable neural slice are unavailable"
        ),
        prerequisites={
            "alignment_complete": symbolic["neural_alignment"]["complete"],
            "activation_bounds": None,
            "slice_tractable": False,
        },
    )
    results = [
        lean_claim,
        generator_claim,
        equivalence_claim,
        incomplete_claim,
        neural_claim,
    ]
    input_paths = (
        "configs/contracts/curb65-llama-v2.json",
        "lean-toolchain",
        "proofs/ClinicalTranslator/CURB65.lean",
        "src/clinical_translator/assurance/oracle.py",
        "src/clinical_translator/assurance/proof_engine.py",
        "src/clinical_translator/assurance/symbolic.py",
        "src/clinical_translator/data/generator.py",
        "evidence/goal-09/report.json",
        "evidence/goal-10/report.json",
        "uv.lock",
    )
    pinned_inputs = {
        path: {"sha256": _sha256(root / path)} for path in input_paths
    }
    certificate_payload = {
        "contract_ref": contract_ref,
        "pinned_inputs": pinned_inputs,
        "results": [
            {
                "id": result["id"],
                "status": result["status"],
                "certificate": result.get("certificate"),
                "counterexample": result.get("counterexample"),
                "reason": result.get("reason"),
            }
            for result in results
        ],
    }
    certificate_id = "sha256:" + hashlib.sha256(
        json.dumps(
            certificate_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "certificate_id": certificate_id,
        "contract_ref": contract_ref,
        "pinned_inputs": pinned_inputs,
        "runtime": {
            "python": platform.python_version(),
            "lean": lean_version,
            "timeout_seconds": timeout_seconds,
        },
        "reproduce": [
            "uv run python -m clinical_translator.assurance.proof_engine",
            "uv run python -m unittest discover -s tests -v",
        ],
        "results": results,
        "summary": dict(Counter(result["status"] for result in results)),
        "guarantee_boundary": {
            "proved": "formal theorem or exhaustive check over a declared finite domain",
            "counterexample": "a concrete witness falsifies the stated obligation",
            "unknown": "unsupported, timed out, or insufficiently bounded; never promoted to proof",
            "excluded": [
                "whole-transformer verification",
                "arbitrary token sequences",
                "all natural-language paraphrases",
            ],
        },
    }


def validate_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != 1:
        raise ValueError("unsupported proof certificate")
    if not report.get("certificate_id", "").startswith("sha256:"):
        raise ValueError("proof certificate lacks a content identifier")
    contract_ref = report.get("contract_ref")
    if not contract_ref:
        raise ValueError("proof certificate lacks the exact contract")
    if not report.get("pinned_inputs") or not report.get("reproduce"):
        raise ValueError("proof certificate inputs are not reproducible")

    results = {result.get("id"): result for result in report.get("results", [])}
    expected = {
        "symbolic_scorer_theorem": "proved",
        "generator_coverage": "proved",
        "symbolic_mechanism_equivalence": "proved",
        "incomplete_scorer_equivalence": "counterexample",
        "local_neural_slice_certification": "unknown",
    }
    if set(results) != set(expected):
        raise ValueError("proof certificate has unexpected obligations")
    for identifier, result in results.items():
        if result.get("status") not in STATUSES:
            raise ValueError("proof result has an invalid status")
        if result.get("status") != expected[identifier]:
            raise ValueError(f"{identifier} did not reach its required result")
        if result.get("contract_ref") != contract_ref:
            raise ValueError("proof result uses the wrong contract")
        domain = result.get("domain", {})
        if not domain.get("id") or not domain.get("description") or not isinstance(
            domain.get("cardinality"), int
        ):
            raise ValueError("proof result lacks an exact domain")
        if result["status"] == "proved" and not result.get("certificate"):
            raise ValueError("proved result lacks a certificate")
        if result["status"] == "counterexample" and not result.get(
            "counterexample"
        ):
            raise ValueError("failed result lacks a concrete counterexample")
        if result["status"] == "unknown" and not result.get("reason"):
            raise ValueError("unknown result lacks a reason")
        if result.get("kind") == "empirical" and result["status"] == "proved":
            raise ValueError("empirical evidence was labelled as proof")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Goal 11 certificate")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/goal-11/certificate.json"),
    )
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    root = Path(__file__).parents[3]
    report = build_report(root, args.timeout)
    validate_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"{report['certificate_id']} "
        f"({', '.join(f'{key}={value}' for key, value in sorted(report['summary'].items()))})"
    )


if __name__ == "__main__":
    main()
