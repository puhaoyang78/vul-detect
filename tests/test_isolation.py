import unittest

from semantic_demo.cli import (
    FORBIDDEN_DETECTION_FIELDS,
    _load_entry,
    build_parser,
    read_jsonl,
    validate_detection_manifest,
)


class IsolationTests(unittest.TestCase):
    def test_detection_manifest_contains_no_oracle_fields(self):
        samples = read_jsonl("data/detection_samples.jsonl")
        validate_detection_manifest(samples)
        for sample in samples:
            self.assertFalse(FORBIDDEN_DETECTION_FIELDS & set(sample))

    def test_all_entries_are_locally_available(self):
        samples = read_jsonl("data/detection_samples.jsonl")
        self.assertEqual(60, len(samples))
        for sample in samples:
            _, entry = _load_entry(sample)
            self.assertEqual(sample["entry_function"], entry.name)

    def test_llm_normalization_defaults_to_local_qwen(self):
        args = build_parser().parse_args(["normalize"])
        self.assertEqual("local", args.llm_backend)
        self.assertTrue(args.llama_server.endswith("/llama-server"))
        self.assertTrue(args.local_model.endswith(".gguf"))

    def test_run_defaults_to_local_joern_and_jdk(self):
        args = build_parser().parse_args(["run"])
        self.assertEqual("/home/phy/joern", args.joern_dir)
        self.assertEqual("/home/phy/jdk21", args.java_home)
        self.assertFalse(args.refresh)
        self.assertEqual("/home/PublicData/PHY-data/resource/codebert-base", args.linevul_codebert_path)
        self.assertEqual("/home/PublicData/PHY-data/resource/linevul/12heads_linevul_model.bin", args.linevul_checkpoint)
        self.assertEqual(0.5, args.linevul_threshold)


if __name__ == "__main__":
    unittest.main()
