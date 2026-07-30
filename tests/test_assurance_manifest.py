import json
import unittest
from pathlib import Path

from clinical_translator.assurance.manifest import (
    CLAIM_STATUSES,
    OUTPUTS,
    build_package,
    validate_manifest,
)

ROOT = Path(__file__).parents[1]


class AssuranceManifestTests(unittest.TestCase):
    def test_complete_labelled_reproducible_package(self) -> None:
        manifest, generated = build_package(ROOT)
        validate_manifest(manifest, ROOT, generated)
        self.assertEqual(
            {claim["status"] for claim in manifest["claims"]},
            CLAIM_STATUSES,
        )
        self.assertFalse(
            any(
                claim["status"] == "proved"
                for claim in manifest["claims"]
                if claim["id"]
                not in {
                    "symbolic_scorer_correctness",
                    "generator_coverage",
                    "symbolic_mechanism_equivalence",
                }
            )
        )
        notebook = json.loads(generated[OUTPUTS["notebook"]])
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "<notebook>", "exec")
        self.assertIn(
            "unrestricted natural-language equivalence",
            manifest["limitations"]["unsupported_claims"],
        )
        self.assertIn(
            "diagnosis or treatment safety",
            manifest["limitations"]["unsupported_claims"],
        )


if __name__ == "__main__":
    unittest.main()
