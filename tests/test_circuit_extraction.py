import copy
import unittest
from pathlib import Path

from clinical_translator.contracts.validation import FACTS, load
from clinical_translator.data.generator import generate
from clinical_translator.interpretability.circuits import (
    HELD_OUT_TEMPLATES,
    KEEP_TOLERANCE,
    TRAIN_TEMPLATE,
    circuit_records,
    validate_report,
)

CONTRACT = Path(__file__).parents[1] / "configs/contracts/curb65-llama-v2.json"


class CircuitExtractionTests(unittest.TestCase):
    def test_bounded_circuit_acceptance_gate(self) -> None:
        records = circuit_records(generate(load(CONTRACT))[0])
        self.assertEqual(len(records), 96)

        circuit = {
            "selection": {
                "candidate_components": 1056,
                "retained_components": 64,
            },
            "keep_only": {"agreement_with_full_model": KEEP_TOLERANCE},
            "remove_only": {
                "full_choice_margin_drop": 0.25,
                "degradation_pass": True,
            },
            "causal_ablation": {"pass": True},
            "graph": {
                "nodes": [
                    {"type": "feature"},
                    {"type": "attention_head"},
                    {"type": "mlp"},
                    {"type": "fact_logit"},
                ],
                "edges": [{"source": "head", "target": "logit"}],
            },
        }
        report = {
            "schema_version": 1,
            "scope": {
                "train_templates": [TRAIN_TEMPLATE],
                "held_out_templates": list(HELD_OUT_TEMPLATES),
                "held_out_prompts": 64,
                "numeric_boundaries_covered": True,
            },
            "models": {
                role: {"circuits": {fact: copy.deepcopy(circuit) for fact in FACTS}}
                for role in ("primary", "control")
            },
            "comparison": {
                fact: {"component_jaccard": 0.5} for fact in FACTS
            },
        }
        validate_report(report)
        report["models"]["primary"]["circuits"]["confusion"]["keep_only"][
            "agreement_with_full_model"
        ] = KEEP_TOLERANCE - 0.01
        with self.assertRaises(ValueError):
            validate_report(report)


if __name__ == "__main__":
    unittest.main()
