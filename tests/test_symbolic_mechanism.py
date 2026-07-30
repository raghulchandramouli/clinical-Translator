import copy
import json
import unittest
from pathlib import Path

from clinical_translator.contracts.validation import FACTS
from clinical_translator.assurance.symbolic import build_report, validate_report

ROOT = Path(__file__).parents[1]


class SymbolicMechanismTests(unittest.TestCase):
    def test_oracle_match_and_incomplete_alignment(self) -> None:
        report = build_report(
            json.loads((ROOT / "evidence/goal-09/report.json").read_text()),
            json.loads((ROOT / "evidence/goal-08/report.json").read_text()),
        )
        validate_report(report)
        self.assertEqual(report["candidate_mechanism"]["inputs"], list(FACTS))
        self.assertTrue(report["oracle_alignment"]["matches"])
        self.assertFalse(report["neural_alignment"]["complete"])
        self.assertEqual(
            {item["variable"] for item in report["counterexamples"]},
            set(FACTS),
        )

        invalid = copy.deepcopy(report)
        invalid["candidate_mechanism"]["hidden_variables"] = ["severity"]
        with self.assertRaises(ValueError):
            validate_report(invalid)


if __name__ == "__main__":
    unittest.main()
