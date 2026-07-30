import copy
import unittest
from pathlib import Path

from clinical_translator.contracts.validation import load
from clinical_translator.data.generator import generate
from clinical_translator.interpretability.capture import (
    CAPTURE_LAYERS,
    SMOKE_COMBINATIONS,
    smoke_records,
    tensor_names,
    validate_manifest,
)

CONTRACT = Path(__file__).parents[1] / "configs/contracts/curb65-llama-v2.json"


class ActivationCaptureTests(unittest.TestCase):
    def test_fixed_bounded_bf16_capture(self) -> None:
        records = smoke_records(generate(load(CONTRACT))[0])
        self.assertEqual(
            [record["id"] for record in records],
            [f"complete-t01-{bits}" for bits in SMOKE_COMBINATIONS],
        )
        tensors = []
        for name in tensor_names():
            tensors.append(
                {
                    "name": name,
                    "dtype": "torch.bfloat16",
                    "token_position": 310,
                    "shape": [
                        1,
                        1,
                        128256 if name == "next_token_logits" else 4096,
                    ],
                }
            )
        manifest = {
            "schema_version": 1,
            "model": {"repo_id": "example/model", "revision": "a" * 40},
            "capture": {
                "layers": list(CAPTURE_LAYERS),
                "token_policy": "last_prompt_token",
                "dtype": "torch.bfloat16",
            },
            "parity": {"exact_match": True},
            "records": [
                {
                    "prompt": {"id": record["id"]},
                    "token": {"position": 310},
                    "tensors": tensors,
                }
                for record in records
            ],
        }
        validate_manifest(manifest)

        invalid = copy.deepcopy(manifest)
        invalid["records"][0]["tensors"][0]["dtype"] = "torch.float32"
        with self.assertRaises(ValueError):
            validate_manifest(invalid)


if __name__ == "__main__":
    unittest.main()
