import copy
import json
import unittest
from pathlib import Path

from clinical_translator.contracts.validation import FACTS, load
from clinical_translator.data.generator import generate
from clinical_translator.interpretability.features import (
    DICTIONARY_ACTIVE_FEATURES,
    DICTIONARY_RANK,
    HELD_OUT_TEMPLATES,
    TRAIN_TEMPLATES,
    feature_records,
    held_out_interventions,
    validate_report,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "configs/contracts/curb65-llama-v2.json"


class FeatureDiscoveryTests(unittest.TestCase):
    def test_template_split_and_acceptance_gate(self) -> None:
        records, pairs = generate(load(CONTRACT))
        selected = feature_records(records)
        predictions = [
            {
                "model": "primary",
                "kind": "complete",
                "status": "validated",
                "prompt_id": record["id"],
                "prediction": record["facts"],
            }
            for record in selected
        ]
        cases = held_out_interventions(pairs, predictions)
        self.assertEqual(len(selected), 256)
        self.assertEqual({case["criterion"] for case in cases}, set(FACTS))

        report = {
            "schema_version": 1,
            "split": {
                "train_templates": list(TRAIN_TEMPLATES),
                "held_out_templates": list(HELD_OUT_TEMPLATES),
                "train_prompts": 192,
                "held_out_prompts": 64,
            },
            "ranked_candidates": [
                {"criterion": fact, "held_out_accuracy": 0.75} for fact in FACTS
            ],
            "dictionary": {
                "source": "task_specific_topk_linear_autoencoder",
                "rank": DICTIONARY_RANK,
                "active_features": DICTIONARY_ACTIVE_FEATURES,
                "external_dictionary_used": False,
                "reconstruction": [
                    {
                        "train_explained_variance": 0.99,
                        "held_out_explained_variance": 0.8,
                        "adequate": True,
                    }
                ],
            },
            "causal_intervention": {
                "success": True,
                "template_id": HELD_OUT_TEMPLATES[0],
            },
        }
        validate_report(report)
        invalid = copy.deepcopy(report)
        invalid["causal_intervention"]["success"] = False
        with self.assertRaises(ValueError):
            validate_report(invalid)

        real_predictions = [
            json.loads(line)
            for line in (ROOT / "evidence/goal-05/predictions.jsonl")
            .read_text()
            .splitlines()
        ]
        self.assertEqual(
            {case["criterion"] for case in held_out_interventions(pairs, real_predictions)},
            set(FACTS),
        )


if __name__ == "__main__":
    unittest.main()
