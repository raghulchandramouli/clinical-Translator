from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clinical_translator.contracts.validation import FACTS, load, reference
from clinical_translator.data.generator import (
    REQUIRED_CONCEPTS,
    TEMPLATES,
    generate,
    write_jsonl,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "configs/contracts/curb65-llama-v2.json"


class GeneratorTest(unittest.TestCase):
    def test_complete_finite_dataset(self) -> None:
        contract = load(CONTRACT)
        records, pairs = generate(contract)
        self.assertEqual((records, pairs), generate(contract))
        self.assertEqual(
            tuple(contract["input_domain"]["record"]["required_concepts"]),
            REQUIRED_CONCEPTS,
        )
        template_range = contract["input_domain"]["paraphrases_per_combination"]
        self.assertLessEqual(template_range["minimum"], len(TEMPLATES))
        self.assertLessEqual(len(TEMPLATES), template_range["maximum"])

        complete = [record for record in records if record["kind"] == "complete"]
        incomplete = [record for record in records if record["kind"] == "incomplete"]
        self.assertEqual(len(complete), 32 * len(TEMPLATES))
        self.assertEqual(len(incomplete), len(REQUIRED_CONCEPTS))
        self.assertEqual(
            {record["provenance"]["combination"] for record in complete},
            {f"{combination:05b}" for combination in range(32)},
        )
        for combination in range(32):
            self.assertEqual(
                sum(
                    record["provenance"]["combination"] == f"{combination:05b}"
                    for record in complete
                ),
                len(TEMPLATES),
            )

        by_id = {record["id"]: record for record in complete}
        self.assertEqual(len(pairs), len(TEMPLATES) * 16 * len(FACTS))
        self.assertEqual(len({pair["id"] for pair in pairs}), len(pairs))
        fields = {
            "confusion": {"mental_state"},
            "elevated_urea": {"urea_mmol_l"},
            "high_respiratory_rate": {"respiratory_rate_per_minute"},
            "low_blood_pressure": {"systolic_bp_mmhg", "diastolic_bp_mmhg"},
            "age_at_least_65": {"age_years"},
        }
        for pair in pairs:
            source, target = by_id[pair["source_id"]], by_id[pair["target_id"]]
            changed = {
                fact for fact in FACTS if source["facts"][fact] != target["facts"][fact]
            }
            changed_values = {
                field
                for field in REQUIRED_CONCEPTS
                if source["values"][field] != target["values"][field]
            }
            self.assertEqual(changed, {pair["criterion"]})
            self.assertEqual(len(changed_values), 1)
            self.assertTrue(changed_values <= fields[pair["criterion"]])
            self.assertEqual(
                source["provenance"]["template_id"],
                target["provenance"]["template_id"],
            )

        values = [record["values"] for record in complete]
        self.assertTrue({6.9, 7.0, 7.1} <= {value["urea_mmol_l"] for value in values})
        self.assertTrue(
            {29, 30, 31}
            <= {value["respiratory_rate_per_minute"] for value in values}
        )
        self.assertTrue({64, 65, 66} <= {value["age_years"] for value in values})
        self.assertTrue(
            {89, 90, 91} <= {value["systolic_bp_mmhg"] for value in values}
        )
        self.assertTrue(
            {59, 60, 61} <= {value["diastolic_bp_mmhg"] for value in values}
        )

        contract_ref = reference(contract)
        self.assertEqual(len({record["id"] for record in records}), len(records))
        for record in records:
            self.assertEqual(set(record["values"]), set(REQUIRED_CONCEPTS))
            self.assertEqual(record["provenance"]["contract_ref"], contract_ref)
            self.assertIs(record["provenance"]["synthetic"], True)
            self.assertNotIn("patient", record["prompt"].lower())

        for record in complete:
            value, facts = record["values"], record["facts"]
            self.assertEqual(facts["confusion"], value["mental_state"].startswith("new"))
            self.assertEqual(facts["elevated_urea"], value["urea_mmol_l"] > 7)
            self.assertEqual(
                facts["high_respiratory_rate"],
                value["respiratory_rate_per_minute"] >= 30,
            )
            self.assertEqual(
                facts["low_blood_pressure"],
                value["systolic_bp_mmhg"] < 90
                or value["diastolic_bp_mmhg"] <= 60,
            )
            self.assertEqual(facts["age_at_least_65"], value["age_years"] >= 65)
        self.assertEqual(
            {record["provenance"]["missing_concept"] for record in incomplete},
            set(REQUIRED_CONCEPTS),
        )
        for record in incomplete:
            self.assertEqual(sum(value is None for value in record["values"].values()), 1)
            self.assertEqual(sum(value is None for value in record["facts"].values()), 1)
            self.assertIn("not reported", record["prompt"])

        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first", Path(directory) / "second"
            first.mkdir()
            second.mkdir()
            for path in (first, second):
                write_jsonl(records, path / "vignettes.jsonl")
                write_jsonl(pairs, path / "counterfactual_pairs.jsonl")
            self.assertEqual(
                (first / "vignettes.jsonl").read_bytes(),
                (second / "vignettes.jsonl").read_bytes(),
            )
            self.assertEqual(
                (first / "counterfactual_pairs.jsonl").read_bytes(),
                (second / "counterfactual_pairs.jsonl").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
