from __future__ import annotations

import copy
import unittest
from pathlib import Path

from clinical_translator.contracts.validation import FACTS, load, reference, validate

CONTRACT = Path(__file__).parents[1] / "configs/contracts/curb65-medgemma-1.5-v1.json"
REFERENCE = "curb65-medgemma-1.5-v1@sha256:1b6f0ca2d0d79276dfdfb3846c3d024a961b446c939c43a5eb00cbb79307f246"


class ContractTest(unittest.TestCase):
    def test_frozen_contract(self) -> None:
        contract = load(CONTRACT)
        self.assertEqual(tuple(contract["output_schema"]["properties"]), FACTS)
        self.assertEqual(reference(contract), REFERENCE)

        invalid = copy.deepcopy(contract)
        invalid["output_schema"]["properties"]["extra"] = {"type": "boolean"}
        self.assertTrue(validate(invalid))


if __name__ == "__main__":
    unittest.main()
