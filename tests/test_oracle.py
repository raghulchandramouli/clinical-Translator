from __future__ import annotations

import ast
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from clinical_translator.assurance.oracle import (
    combinations,
    incomplete_score,
    score,
)
from clinical_translator.contracts.validation import load
from clinical_translator.data.generator import generate

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "configs/contracts/curb65-medgemma-1.5-v1.json"
PROOF = ROOT / "proofs/ClinicalTranslator/CURB65.lean"


class OracleTest(unittest.TestCase):
    def test_python_and_lean_oracles(self) -> None:
        facts = list(combinations())
        python_table = [score(record) for record in facts]
        self.assertEqual(len(facts), 32)
        self.assertEqual(python_table, [sum(record.values()) for record in facts])

        generated, _ = generate(load(CONTRACT))
        for record in generated:
            if record["kind"] == "complete":
                self.assertEqual(
                    score(record["facts"]),
                    record["provenance"]["combination"].count("1"),
                )

        confusion_only = dict.fromkeys(facts[0], False)
        confusion_only["confusion"] = True
        self.assertEqual(score(confusion_only), 1)
        self.assertEqual(incomplete_score(confusion_only), 0)
        with self.assertRaises(ValueError):
            score({**confusion_only, "extra": False})
        with self.assertRaises(TypeError):
            score({**confusion_only, "confusion": 1})

        lean = shutil.which("lean") or Path.home() / ".elan/bin/lean"
        self.assertTrue(Path(lean).exists(), "install Elan to check the Lean proof")
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [str(lean), "-o", f"{directory}/CURB65.olean", str(PROOF)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
        self.assertEqual(ast.literal_eval(result.stdout.strip()), python_table)


if __name__ == "__main__":
    unittest.main()
