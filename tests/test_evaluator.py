import unittest
from pathlib import Path

from clinical_translator.contracts.validation import load
from clinical_translator.data.generator import generate
from clinical_translator.evaluation.evaluator import evaluate

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "configs/contracts/curb65-medgemma-1.5-v1.json"


class EvaluatorTests(unittest.TestCase):
    def test_complete_metrics_and_failures(self) -> None:
        contract = load(CONTRACT)
        records, pairs = generate(contract)
        predictions = []
        for role in contract["models"]:
            for index, record in enumerate(records):
                prediction = {
                    fact: False if value is None else value
                    for fact, value in record["facts"].items()
                }
                item = {
                    "model": role,
                    "prompt_id": record["id"],
                    "status": "validated",
                    "prediction": prediction,
                }
                if role == "control" and index == 0:
                    prediction["confusion"] = not prediction["confusion"]
                if role == "control" and index == 1:
                    item = {
                        "model": role,
                        "prompt_id": record["id"],
                        "status": "parser_failure",
                        "error": "invalid JSON",
                    }
                if role == "control" and index == 2:
                    item = {
                        "model": role,
                        "prompt_id": record["id"],
                        "status": "runner_failure",
                        "error": "remote error",
                    }
                predictions.append(item)

        metrics, counterfactuals, failures = evaluate(
            contract,
            records,
            pairs,
            predictions,
        )
        self.assertEqual(metrics["scope"]["model_prompt_outcomes"], 524)
        self.assertEqual(metrics["models"]["primary"]["exact_record"]["accuracy"], 1)
        self.assertEqual(metrics["models"]["control"]["format_failures"], 1)
        self.assertEqual(metrics["models"]["control"]["runner_failures"], 1)
        self.assertEqual(
            set(metrics["by_template"]["primary"]),
            {f"t{i:02d}" for i in range(1, 9)},
        )
        self.assertEqual(len(metrics["by_combination"]["primary"]), 32)
        self.assertEqual(
            set(metrics["by_boundary"]["primary"]),
            set(contract["output_schema"]["required"]),
        )
        self.assertEqual(len(counterfactuals), 2 * len(pairs))
        self.assertTrue(
            {"fact_error", "parser_failure", "runner_failure", "incomplete_guess"}
            <= {failure["type"] for failure in failures}
        )
        with self.assertRaises(ValueError):
            evaluate(contract, records, pairs, predictions[:-1])


if __name__ == "__main__":
    unittest.main()
