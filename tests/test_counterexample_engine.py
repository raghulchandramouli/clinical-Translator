import json
import unittest
from pathlib import Path

from clinical_translator.assurance.counterexamples import (
    build_report,
    omitted_confusion_demo,
    validate_report,
)

ROOT = Path(__file__).parents[1]


class CounterexampleEngineTests(unittest.TestCase):
    def test_reduction_coverage_and_fixed_regression(self) -> None:
        report, corpus = build_report(ROOT)
        validate_report(report, corpus)
        self.assertEqual(report["source_coverage"]["goal05"]["mapped"], 961)
        self.assertEqual(report["source_coverage"]["goal09"]["mapped"], 15)
        self.assertTrue(
            all(item["reduction"]["preserves_failure"] for item in corpus)
        )
        self.assertTrue(
            all(
                item["reduction"]["valid_under_goal01_grammar"]
                for item in corpus
            )
        )

        demo = omitted_confusion_demo()
        self.assertEqual(demo["before"]["status"], "counterexample")
        self.assertEqual(demo["after"]["status"], "fixed")
        fixed = next(
            item for item in corpus if item["id"] == "ce-formal-omitted-confusion"
        )
        self.assertEqual(fixed["resolution"]["status"], "fixed")


if __name__ == "__main__":
    unittest.main()
