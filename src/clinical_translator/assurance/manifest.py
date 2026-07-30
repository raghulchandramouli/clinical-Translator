"""Build the final versioned assurance package without blurring claim types."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from clinical_translator.contracts.validation import FACTS, load, reference
from clinical_translator.models.protocol import PROMPT_VERSION

CLAIM_STATUSES = {"proved", "empirical", "counterexample", "unknown"}
OUTPUTS = {
    "manifest": "evidence/goal-13/assurance-manifest-v1.yaml",
    "benchmark": "evidence/goal-13/benchmark.json",
    "report": "evidence/goal-13/technical-report.txt",
    "notebook": "notebooks/assurance-audit.ipynb",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _load(root: Path, path: str) -> dict[str, Any]:
    return json.loads((root / path).read_text())


def _source_digest(root: Path) -> tuple[str, int]:
    paths = sorted(
        [
            *root.glob("src/**/*.py"),
            *root.glob("proofs/**/*.lean"),
            root / "pyproject.toml",
            root / "uv.lock",
            root / "lean-toolchain",
            root / "configs/contracts/curb65-llama-v2.json",
        ]
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), len(paths)


def _artifact_entry(path: str, data: bytes) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": path,
        "sha256": _sha256(data),
        "bytes": len(data),
    }
    if path.endswith(".jsonl"):
        entry["records"] = len(data.splitlines())
    elif path.endswith((".json", ".yaml")):
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(parsed, dict) and "schema_version" in parsed:
                entry["schema_version"] = parsed["schema_version"]
    return entry


def _claims(
    goal05: dict[str, Any],
    goal07: dict[str, Any],
    goal08: dict[str, Any],
    goal09: dict[str, Any],
    goal11: dict[str, Any],
    goal12: dict[str, Any],
) -> list[dict[str, Any]]:
    proof_results = {item["id"]: item for item in goal11["results"]}
    return [
        {
            "id": "symbolic_scorer_correctness",
            "status": "proved",
            "statement": "the five-Boolean scorer equals its formal specification",
            "domain": "all 32 five-Boolean CURB-65 assignments",
            "evidence": "evidence/goal-11/certificate.json#/results/0",
            "certificate_id": goal11["certificate_id"],
        },
        {
            "id": "generator_coverage",
            "status": "proved",
            "statement": "the deterministic generator covers its declared finite domain",
            "domain": "262 prompts and 640 matched pairs under grammar v1",
            "evidence": "evidence/goal-11/certificate.json#/results/1",
            "dataset_sha256": proof_results["generator_coverage"]["certificate"][
                "dataset_sha256"
            ],
        },
        {
            "id": "symbolic_mechanism_equivalence",
            "status": "proved",
            "statement": "the Goal 10 five-input mechanism equals the Goal 03 scorer",
            "domain": "all 32 five-Boolean CURB-65 assignments",
            "evidence": "evidence/goal-11/certificate.json#/results/2",
            "certificate_id": goal11["certificate_id"],
        },
        {
            "id": "behavioural_translation",
            "status": "empirical",
            "statement": "model translation performance on the complete prompt set",
            "domain": "256 complete prompts for each pinned model",
            "evidence": "evidence/goal-05/metrics.json",
            "primary_exact_accuracy": goal05["models"]["primary"]["exact_record"][
                "accuracy"
            ],
            "control_exact_accuracy": goal05["models"]["control"]["exact_record"][
                "accuracy"
            ],
        },
        {
            "id": "feature_decodability",
            "status": "empirical",
            "statement": "candidate directions decode each fact on held-out templates",
            "domain": "64 held-out prompts on templates t07 and t08",
            "evidence": "evidence/goal-07/report.json",
            "criteria": sorted(
                {item["criterion"] for item in goal07["ranked_candidates"]}
            ),
        },
        {
            "id": "candidate_circuit_sufficiency",
            "status": "empirical",
            "statement": "retained last-token circuits preserve measured held-out choices",
            "domain": "64 held-out prompts on templates t07 and t08",
            "evidence": "evidence/goal-08/report.json",
            "keep_tolerance": goal08["scope"]["keep_only_tolerance"],
        },
        {
            "id": "causal_intervention_alignment",
            "status": "empirical",
            "statement": "matched residual interventions control each tested fact token",
            "domain": "160 held-out matched interventions",
            "evidence": "evidence/goal-09/report.json",
            "variable_statuses": {
                fact: goal09["variables"][fact]["status"] for fact in FACTS
            },
        },
        {
            "id": "neural_symbolic_alignment",
            "status": "unknown",
            "statement": "the five neural variables form a complete symbolic alignment",
            "domain": "Goal 09 tested synthetic interventions only",
            "evidence": "evidence/goal-10/report.json#/neural_alignment",
            "reason": "all five variable alignments are mixed; aligned_score is null",
        },
        {
            "id": "incomplete_scorer_equivalence",
            "status": "counterexample",
            "statement": "a scorer omitting confusion equals the complete scorer",
            "domain": "all 32 five-Boolean CURB-65 assignments",
            "evidence": "evidence/goal-11/certificate.json#/results/3",
            "witness": proof_results["incomplete_scorer_equivalence"]["counterexample"],
        },
        {
            "id": "local_neural_slice_certification",
            "status": "unknown",
            "statement": "the extracted neural slice is formally certified",
            "domain": "Goal 09 held-out intervention slice",
            "evidence": "evidence/goal-11/certificate.json#/results/4",
            "reason": proof_results["local_neural_slice_certification"]["reason"],
        },
        {
            "id": "recorded_failure_regressions",
            "status": "counterexample",
            "statement": "current behavioural and causal claims have reproducible failures",
            "domain": "Goal 01 grammar and Goal 09 intervention configurations",
            "evidence": "evidence/goal-12/regressions.jsonl",
            "active": goal12["corpus"]["active"],
            "fixed": goal12["corpus"]["fixed"],
        },
    ]


def _benchmark(
    contract_ref: str,
    goal05: dict[str, Any],
    goal07: dict[str, Any],
    goal08: dict[str, Any],
    goal09: dict[str, Any],
    goal12: dict[str, Any],
) -> dict[str, Any]:
    best_features = {}
    for fact in FACTS:
        candidate = next(
            item for item in goal07["ranked_candidates"] if item["criterion"] == fact
        )
        best_features[fact] = {
            "layer": candidate["layer"],
            "held_out_accuracy": candidate["held_out_accuracy"],
        }
    return {
        "schema_version": 1,
        "contract_ref": contract_ref,
        "behavioural_summary": {
            "source": "evidence/goal-05/metrics.json",
            "scope": goal05["scope"],
            "models": {
                role: {key: value for key, value in result.items() if key != "records"}
                for role, result in goal05["models"].items()
            },
            "counterfactuals": goal05["counterfactuals"],
        },
        "feature_summary": {
            "split": goal07["split"],
            "best_candidate_per_fact": best_features,
            "dictionary": {
                "source": goal07["dictionary"]["source"],
                "rank": goal07["dictionary"]["rank"],
                "active_features": goal07["dictionary"]["active_features"],
                "reconstruction": goal07["dictionary"]["reconstruction"],
            },
        },
        "circuit_summary": {
            fact: {
                "retained_components": goal08["models"]["primary"]["circuits"][fact][
                    "selection"
                ]["retained_components"],
                "keep_agreement": goal08["models"]["primary"]["circuits"][fact][
                    "keep_only"
                ]["agreement_with_full_model"],
                "remove_degradation": goal08["models"]["primary"]["circuits"][fact][
                    "remove_only"
                ]["degradation_pass"],
            }
            for fact in FACTS
        },
        "causal_summary": {
            fact: {
                "status": goal09["variables"][fact]["status"],
                "eligible_pairs": goal09["variables"][fact]["interchange"][
                    "eligible_pairs"
                ],
                "predicted_direction_rate": goal09["variables"][fact]["interchange"][
                    "predicted_direction_rate"
                ],
                "unrelated_output_stability": goal09["variables"][fact]["interchange"][
                    "unrelated_output_stability"
                ],
            }
            for fact in FACTS
        },
        "counterexample_summary": goal12["corpus"],
    }


def _technical_report(
    contract: dict[str, Any],
    contract_ref: str,
    claims: list[dict[str, Any]],
    goal05: dict[str, Any],
    goal09: dict[str, Any],
    goal11: dict[str, Any],
    goal12: dict[str, Any],
) -> str:
    lines = [
        "CERTIFIED TRANSLATION CONTRACTS FOR CLINICAL LANGUAGE MODELS",
        "Technical assurance report v1",
        "",
        f"Contract: {contract_ref}",
        f"Primary model: {contract['models']['primary']['repo_id']} @ {contract['models']['primary']['revision']}",
        f"Control model: {contract['models']['control']['repo_id']} @ {contract['models']['control']['revision']}",
        "",
        "SCOPE",
        "Five Boolean CURB-65 facts; 32 logical combinations; eight controlled",
        "templates; 256 complete prompts; 6 incomplete prompts; no patient data.",
        "",
        "RESULTS",
        f"Primary exact-record accuracy: {goal05['models']['primary']['exact_record']['accuracy']:.4f}",
        f"Control exact-record accuracy: {goal05['models']['control']['exact_record']['accuracy']:.4f}",
        "All eligible Goal 09 matched patches changed their tested fact token in",
        "the predicted direction, but every variable remains mixed.",
        "Unrelated-output stability: "
        + ", ".join(
            f"{fact}={goal09['variables'][fact]['interchange']['unrelated_output_stability']:.4f}"
            for fact in FACTS
        ),
        "",
        "CLAIMS",
        *[
            f"[{claim['status'].upper()}] {claim['id']}: {claim['statement']}"
            for claim in claims
        ],
        "",
        "CERTIFICATE",
        f"ID: {goal11['certificate_id']}",
        "Summary: "
        + ", ".join(
            f"{status}={count}" for status, count in sorted(goal11["summary"].items())
        ),
        "",
        "COUNTEREXAMPLES",
        f"Regression corpus: active={goal12['corpus']['active']}, fixed={goal12['corpus']['fixed']}.",
        "The omitted-confusion witness fails under incomplete_score and passes",
        "under the corrected five-fact score.",
        "",
        "LIMITATIONS",
        *[f"- {item}" for item in contract["claims"]["unsupported"]],
        "- Neural findings apply only to the tested synthetic prompts and interventions.",
        "- No whole-model, arbitrary-language, safety, or deployment proof is claimed.",
        "",
        "REPRODUCE",
        "uv sync --frozen",
        "uv run python -m clinical_translator.assurance.proof_engine",
        "uv run python -m clinical_translator.assurance.counterexamples --demo",
        "uv run python -m clinical_translator.assurance.manifest",
        "uv run python -m unittest discover -s tests -v",
        "",
    ]
    return "\n".join(lines)


def _notebook() -> dict[str, Any]:
    def code(source: str) -> dict[str, Any]:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.splitlines()],
        }

    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Clinical Translator assurance audit\n",
                    "Rebuild the bounded certificate and live counterexample using only repository inputs.\n",
                ],
            },
            code(
                "from pathlib import Path\n"
                "import json\n"
                "root = Path.cwd()\n"
                "manifest = json.loads((root / 'evidence/goal-13/assurance-manifest-v1.yaml').read_text())\n"
                "assert {claim['status'] for claim in manifest['claims']} <= {'proved', 'empirical', 'counterexample', 'unknown'}\n"
                "manifest['manifest_version'], manifest['contract_ref']"
            ),
            code(
                "from clinical_translator.assurance.proof_engine import build_report, validate_report\n"
                "certificate = build_report(root)\n"
                "validate_report(certificate)\n"
                "certificate['summary']"
            ),
            code(
                "from clinical_translator.assurance.counterexamples import omitted_confusion_demo\n"
                "demo = omitted_confusion_demo()\n"
                "assert demo['before']['status'] == 'counterexample'\n"
                "assert demo['after']['status'] == 'fixed'\n"
                "demo"
            ),
            code(
                "benchmark = json.loads((root / 'evidence/goal-13/benchmark.json').read_text())\n"
                "{role: result['exact_record']['accuracy'] for role, result in benchmark['behavioural_summary']['models'].items()}"
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_package(root: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    contract = load(root / "configs/contracts/curb65-llama-v2.json")
    contract_ref = reference(contract)
    goal05 = _load(root, "evidence/goal-05/metrics.json")
    goal07 = _load(root, "evidence/goal-07/report.json")
    goal08 = _load(root, "evidence/goal-08/report.json")
    goal09 = _load(root, "evidence/goal-09/report.json")
    goal11 = _load(root, "evidence/goal-11/certificate.json")
    goal12 = _load(root, "evidence/goal-12/report.json")
    claims = _claims(goal05, goal07, goal08, goal09, goal11, goal12)
    benchmark = _benchmark(contract_ref, goal05, goal07, goal08, goal09, goal12)
    report = _technical_report(
        contract,
        contract_ref,
        claims,
        goal05,
        goal09,
        goal11,
        goal12,
    )
    notebook = _notebook()
    generated = {
        OUTPUTS["benchmark"]: _json_bytes(benchmark),
        OUTPUTS["report"]: report.encode(),
        OUTPUTS["notebook"]: _json_bytes(notebook),
    }
    evidence_files = []
    for path in root.glob("evidence/goal-*"):
        if path.is_file():
            evidence_files.append(path)
        elif path.name != "goal-13":
            evidence_files.extend(item for item in path.rglob("*") if item.is_file())
    evidence_paths = sorted(str(path.relative_to(root)) for path in evidence_files)
    artifacts = [
        _artifact_entry(path, (root / path).read_bytes()) for path in evidence_paths
    ]
    artifacts.extend(
        _artifact_entry(path, data) for path, data in sorted(generated.items())
    )
    source_digest, source_files = _source_digest(root)
    manifest = {
        "manifest_version": "1.0.0",
        "schema_version": 1,
        "format": "YAML 1.2 (JSON-compatible)",
        "contract_ref": contract_ref,
        "models": contract["models"],
        "task": {
            "id": contract["contract_id"],
            "name": contract["task"]["name"],
            "output_facts": list(FACTS),
            "prompt_protocol": PROMPT_VERSION,
        },
        "input_domain": {
            "grammar_version": contract["input_domain"]["grammar_version"],
            "logical_combinations": 32,
            "templates": 8,
            "complete_prompts": 256,
            "incomplete_prompts": 6,
            "matched_pairs": 640,
            "patient_data": False,
        },
        "code": {
            "package": "clinical-translator",
            "version": "0.1.0",
            "source_tree_sha256": source_digest,
            "source_files": source_files,
            "uv_lock_sha256": _sha256((root / "uv.lock").read_bytes()),
            "lean_toolchain": (root / "lean-toolchain").read_text().strip(),
        },
        "claims": claims,
        "formal_results": {
            "certificate": "evidence/goal-11/certificate.json",
            "certificate_id": goal11["certificate_id"],
            "summary": goal11["summary"],
        },
        "behavioural_evaluation": {
            "benchmark": OUTPUTS["benchmark"],
            "complete_summary": "evidence/goal-05/metrics.json",
            "model_prompt_outcomes": goal05["scope"]["model_prompt_outcomes"],
        },
        "interpretability_evidence": {
            "features": "evidence/goal-07/report.json",
            "circuits": "evidence/goal-08/report.json",
            "causal": "evidence/goal-09/report.json",
        },
        "counterexamples": {
            "report": "evidence/goal-12/report.json",
            "corpus": "evidence/goal-12/regressions.jsonl",
            "active": goal12["corpus"]["active"],
            "fixed": goal12["corpus"]["fixed"],
        },
        "unknowns": [
            {
                "claim": claim["id"],
                "reason": claim["reason"],
                "evidence": claim["evidence"],
            }
            for claim in claims
            if claim["status"] == "unknown"
        ],
        "assumptions": [
            "inputs are generated by curb65-vignette-grammar-v1",
            "model revisions and inference settings equal the frozen contract",
            "feature and circuit claims concern selected last-token activations",
            "causal claims concern teacher-forced Boolean fact-token decisions",
        ],
        "limitations": {
            "unsupported_claims": contract["claims"]["unsupported"],
            "additional": [
                "neural results do not constitute formal proof",
                "mixed interventions show the five-variable neural alignment is incomplete",
                "no clinical deployment approval is implied",
            ],
        },
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
        "deliverables": {
            "benchmark": OUTPUTS["benchmark"],
            "technical_report": OUTPUTS["report"],
            "reproducible_notebook": OUTPUTS["notebook"],
            "live_counterexample_demo": (
                "uv run python -m clinical_translator.assurance.counterexamples --demo"
            ),
        },
        "reproduce": {
            "requirements": [
                "Python 3.12",
                "uv 0.11.23 or compatible",
                "Lean toolchain from lean-toolchain",
            ],
            "commands": [
                "uv sync --frozen",
                "uv run python -m clinical_translator.assurance.proof_engine",
                "uv run python -m clinical_translator.assurance.counterexamples",
                "uv run python -m clinical_translator.assurance.manifest",
                "uv run python -m unittest discover -s tests -v",
            ],
        },
    }
    generated[OUTPUTS["manifest"]] = _json_bytes(manifest)
    return manifest, generated


def validate_manifest(
    manifest: dict[str, Any],
    root: Path,
    generated: dict[str, bytes],
) -> None:
    if manifest.get("manifest_version") != "1.0.0":
        raise ValueError("assurance manifest is not versioned")
    if not manifest.get("contract_ref") or set(manifest.get("models", {})) != {
        "primary",
        "control",
    }:
        raise ValueError("manifest does not pin the contract and models")
    if manifest.get("task", {}).get("output_facts") != list(FACTS):
        raise ValueError("manifest task differs from the frozen schema")
    domain = manifest.get("input_domain", {})
    if (
        domain.get("grammar_version") != "curb65-vignette-grammar-v1"
        or domain.get("complete_prompts") != 256
        or domain.get("patient_data") is not False
    ):
        raise ValueError("manifest input domain is incomplete")

    claims = manifest.get("claims", [])
    if {claim.get("status") for claim in claims} != CLAIM_STATUSES:
        raise ValueError("manifest claim has an invalid status")
    if any(
        claim["status"] == "proved"
        and claim["id"]
        not in {
            "symbolic_scorer_correctness",
            "generator_coverage",
            "symbolic_mechanism_equivalence",
        }
        for claim in claims
    ):
        raise ValueError("empirical evidence was labelled as proof")
    unsupported = set(manifest.get("limitations", {}).get("unsupported_claims", []))
    if (
        not {
            "unrestricted natural-language equivalence",
            "diagnosis or treatment safety",
        }
        <= unsupported
    ):
        raise ValueError("mandatory unsupported claims are missing")
    if not manifest.get("unknowns") or not manifest.get("counterexamples"):
        raise ValueError("manifest omits unresolved or failing results")

    for artifact in manifest.get("artifacts", []):
        path = artifact["path"]
        data = generated.get(path)
        if data is None:
            data = (root / path).read_bytes()
        if artifact["sha256"] != _sha256(data) or artifact["bytes"] != len(data):
            raise ValueError(f"artifact digest mismatch: {path}")
    deliverables = manifest.get("deliverables", {})
    if set(deliverables) != {
        "benchmark",
        "technical_report",
        "reproducible_notebook",
        "live_counterexample_demo",
    }:
        raise ValueError("delivery package is incomplete")
    if not manifest.get("reproduce", {}).get("commands"):
        raise ValueError("new environments lack reproduction commands")


def main() -> None:
    root = Path(__file__).parents[3]
    manifest, generated = build_package(root)
    validate_manifest(manifest, root, generated)
    for path, data in generated.items():
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    print(
        f"assurance manifest v{manifest['manifest_version']} written with "
        f"{len(manifest['claims'])} labelled claims"
    )


if __name__ == "__main__":
    main()
