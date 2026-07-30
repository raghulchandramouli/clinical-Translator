import copy
import json
import unittest
from pathlib import Path

from clinical_translator.assurance.proof_engine import (
    build_report,
    symbolic_equivalence_claim,
    validate_report,
)

ROOT = Path(__file__).parents[1]


class ProofEngineTests(unittest.TestCase):
    def test_certificate_statuses_and_counterexample(self) -> None:
        report = build_report(ROOT)
        validate_report(report)
        self.assertEqual(
            report["summary"],
            {"proved": 3, "counterexample": 1, "unknown": 1},
        )

        symbolic = json.loads((ROOT / "evidence/goal-10/report.json").read_text())
        symbolic["oracle_alignment"]["truth_table"][0]["symbolic_score"] = 99
        claim = symbolic_equivalence_claim(symbolic, report["contract_ref"])
        self.assertEqual(claim["status"], "counterexample")
        self.assertIn("expected", claim["counterexample"])

        invalid = copy.deepcopy(report)
        result = next(
            item
            for item in invalid["results"]
            if item["id"] == "local_neural_slice_certification"
        )
        result["status"] = "proved"
        result["certificate"] = {"checker": "empirical"}
        with self.assertRaises(ValueError):
            validate_report(invalid)


if __name__ == "__main__":
    unittest.main()
