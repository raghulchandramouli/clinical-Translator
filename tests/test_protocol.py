import json
import unittest

from clinical_translator.models.protocol import parse


class ProtocolTests(unittest.TestCase):
    def test_strict_json_schema(self) -> None:
        valid = {
            "confusion": False,
            "elevated_urea": True,
            "high_respiratory_rate": False,
            "low_blood_pressure": True,
            "age_at_least_65": False,
        }
        self.assertEqual(parse(json.dumps(valid)), valid)
        invalid = (
            "{}",
            json.dumps(valid | {"extra": False}),
            json.dumps(valid | {"confusion": None}),
            json.dumps(valid | {"confusion": 1}),
            '{"confusion":false,"confusion":true}',
            f"```json\n{json.dumps(valid)}\n```",
            json.dumps(valid) + "\nExplanation",
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises((TypeError, ValueError)):
                parse(raw)


if __name__ == "__main__":
    unittest.main()
