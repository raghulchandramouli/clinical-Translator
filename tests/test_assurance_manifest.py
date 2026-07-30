import json
import unittest
from pathlib import Path

from clinical_translator.assurance.manifest import (
    OUTPUTS,
    build_package,
    validate_manifest,
)

ROOT = Path(__file__).parents[1]


class AssuranceManifestTests(unittest.TestCase):
    def test_complete_labelled_reproducible_package(self) -> None:
        manifest, generated = build_package(ROOT)
        validate_manifest(manifest, ROOT, generated)
        notebook = json.loads(generated[OUTPUTS["notebook"]])
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "<notebook>", "exec")


if __name__ == "__main__":
    unittest.main()
