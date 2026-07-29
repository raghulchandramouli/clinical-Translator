from __future__ import annotations

import copy
import unittest
from pathlib import Path

from clinical_translator.contracts.validation import FACTS, load, reference, validate

CONTRACT = Path(__file__).parents[1] / "configs/contracts/curb65-llama-v2.json"
REFERENCE = "curb65-llama-v2@sha256:fc214c7d868fd0b1116c1587e2ffd872b87735b81f5c0d50a681b4d6bbe6a6de"


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
