import unittest

from semantic_demo.cli import (
    FORBIDDEN_DETECTION_FIELDS,
    _load_entry,
    read_jsonl,
    validate_detection_manifest,
)


class IsolationTests(unittest.TestCase):
    def test_detection_manifest_contains_no_oracle_fields(self):
        samples = read_jsonl("data/detection_samples.jsonl")
        validate_detection_manifest(samples)
        for sample in samples:
            self.assertFalse(FORBIDDEN_DETECTION_FIELDS & set(sample))

    def test_all_vulnerable_entries_are_locally_available(self):
        samples = read_jsonl("data/detection_samples.jsonl")
        self.assertEqual(10, len(samples))
        for sample in samples:
            _, entry = _load_entry(sample)
            self.assertEqual(sample["entry_function"], entry.name)


if __name__ == "__main__":
    unittest.main()
