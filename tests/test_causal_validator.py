import copy
import unittest
from pathlib import Path

from clinical_translator.contracts.validation import FACTS, load
from clinical_translator.data.generator import generate
from clinical_translator.interpretability.causal import (
    HELD_OUT_TEMPLATES,
    causal_pairs,
    causal_records,
    validate_report,
)

CONTRACT = Path(__file__).parents[1] / "configs/contracts/curb65-llama-v2.json"


class CausalValidatorTests(unittest.TestCase):
    def test_matched_pair_acceptance_gate(self) -> None:
        records, pairs = generate(load(CONTRACT))
        self.assertEqual(len(causal_records(records)), 64)
        self.assertEqual(len(causal_pairs(pairs)), 160)
        experiment = {
            "effect_size": 1.0,
            "accuracy": 1.0,
            "pass": True,
        }
        variable = {
            "status": "pass",
            "interchange": {
                "eligible_pairs": 4,
                "predicted_direction_rate": 1.0,
                "effect_size": 2.0,
                "accuracy": 1.0,
                "unrelated_output_stability": 1.0,
            },
            "ablations": {
                name: copy.deepcopy(experiment)
                for name in ("feature", "attention_heads", "mlp", "path")
            },
            "controls": {
                "positive_counterfactual": copy.deepcopy(experiment),
                "negative_sham": {
                    "effect_size": 0.0,
                    "stability": 1.0,
                    "pass": True,
                },
                "negative_unrelated_feature": {
                    "effect_size": 0.0,
                    "stability": 1.0,
                    "pass": True,
                },
            },
        }
        report = {
            "schema_version": 1,
            "scope": {
                "templates": list(HELD_OUT_TEMPLATES),
                "matched_pairs": 160,
            },
            "variables": {
                fact: copy.deepcopy(variable) for fact in FACTS
            },
            "counterexamples": [],
        }
        validate_report(report)
        report["variables"]["confusion"]["interchange"][
            "predicted_direction_rate"
        ] = 0.5
        with self.assertRaises(ValueError):
            validate_report(report)


if __name__ == "__main__":
    unittest.main()
