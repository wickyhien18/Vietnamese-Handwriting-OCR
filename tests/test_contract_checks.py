import json
import os
import unittest

from src.common.contract_checks import check_correct_output, check_dataset_index_records, check_splits
from src.common.index_io import read_index, read_splits
from src.stubs.identity_corrector import correct

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "..", "contracts", "examples")

A_DOT = "\u1ea1"  # a with dot below
A_BREVE = "\u0103"  # a with breve


class CorrectContractTest(unittest.TestCase):
    def test_identity_passes(self):
        for text in ("", "a", "nhieu ban doc da bay to", "doc, ban."):
            self.assertEqual(check_correct_output(text, correct(text)), [])

    def test_good_fix_passes(self):
        out = {"text": "b" + A_DOT + "n", "conf": [0.9]}
        self.assertEqual(check_correct_output("ban", out), [])

    def test_wrong_type_and_keys(self):
        self.assertTrue(check_correct_output("a", ["a"]))
        self.assertTrue(check_correct_output("a", {"text": "a"}))
        self.assertTrue(check_correct_output("a", {"text": "a", "conf": [1.0], "extra": 1}))

    def test_syllable_count_must_not_change(self):
        self.assertTrue(check_correct_output("a b", {"text": "a b c", "conf": [1.0, 1.0, 1.0]}))

    def test_conf_length_and_range(self):
        self.assertTrue(check_correct_output("a b", {"text": "a b", "conf": [1.0]}))
        self.assertTrue(check_correct_output("a b", {"text": "a b", "conf": [1.0, 1.5]}))
        self.assertTrue(check_correct_output("a b", {"text": "a b", "conf": [1.0, True]}))

    def test_only_diacritics_may_change(self):
        self.assertTrue(check_correct_output("ban", {"text": "bon", "conf": [0.5]}))
        self.assertTrue(check_correct_output("ban", {"text": "Ban", "conf": [0.5]}))
        self.assertEqual(check_correct_output("ban", {"text": "b" + A_BREVE + "n", "conf": [0.5]}), [])

    def test_punctuation_stays_attached(self):
        self.assertEqual(check_correct_output("doc,", {"text": "\u0111oc,".replace("\u0111", "d"), "conf": [1.0]}), [])
        self.assertTrue(check_correct_output("doc,", {"text": "doc", "conf": [1.0]}))

    def test_output_must_be_nfc(self):
        self.assertTrue(check_correct_output("ban", {"text": "ba\u0323n", "conf": [0.5]}))


class IndexContractTest(unittest.TestCase):
    def record(self, **extra):
        rec = {"id": "r1", "image_path": "x/1.png", "gt_text": "nhi\u1ec1u b\u1ea1n"}
        rec.update(extra)
        return rec

    def test_sample_files_are_valid(self):
        index = read_index(os.path.join(EXAMPLES, "dataset_index.sample.jsonl"))
        splits = read_splits(os.path.join(EXAMPLES, "splits.sample.json"))
        self.assertEqual(check_dataset_index_records(index), [])
        self.assertEqual(check_splits(splits, [rec["id"] for rec in index]), [])

    def test_good_record(self):
        self.assertEqual(check_dataset_index_records([self.record(split="train", writer_id="w1")]), [])

    def test_missing_and_unexpected_fields(self):
        self.assertTrue(check_dataset_index_records([{"id": "r1", "image_path": "x.png"}]))
        self.assertTrue(check_dataset_index_records([self.record(extra="x")]))

    def test_duplicate_ids(self):
        self.assertTrue(check_dataset_index_records([self.record(), self.record()]))

    def test_paths_must_be_relative(self):
        self.assertTrue(check_dataset_index_records([self.record(image_path="/abs/1.png")]))
        self.assertTrue(check_dataset_index_records([self.record(image_path="a\\1.png")]))
        self.assertTrue(check_dataset_index_records([self.record(image_path="../1.png")]))

    def test_text_rules(self):
        self.assertTrue(check_dataset_index_records([self.record(gt_text="ba\u0323n")]))
        self.assertTrue(check_dataset_index_records([self.record(gt_text="a  b")]))
        self.assertTrue(check_dataset_index_records([self.record(gt_text=" a")]))

    def test_partial_split_is_rejected(self):
        records = [self.record(id="a", split="train"), self.record(id="b")]
        self.assertTrue(check_dataset_index_records(records))

    def test_invalid_split_value(self):
        self.assertTrue(check_dataset_index_records([self.record(split="dev")]))


class SplitsContractTest(unittest.TestCase):
    def test_all_three_splits_needed(self):
        self.assertTrue(check_splits({"a": "train", "b": "val"}))
        self.assertEqual(check_splits({"a": "train", "b": "val", "c": "test"}), [])

    def test_coverage_against_index(self):
        splits = {"a": "train", "b": "val", "c": "test"}
        self.assertTrue(check_splits(splits, ["a", "b", "c", "d"]))
        self.assertTrue(check_splits(splits, ["a", "b"]))
        self.assertEqual(check_splits(splits, ["a", "b", "c"]), [])

    def test_invalid_value(self):
        self.assertTrue(check_splits({"a": "train", "b": "dev", "c": "test"}))


class SchemaFilesTest(unittest.TestCase):
    def test_schema_files_are_valid_json(self):
        contracts = os.path.join(HERE, "..", "contracts")
        for name in ("ocr_out.schema.json", "dataset_index.schema.json", "splits.schema.json"):
            with open(os.path.join(contracts, name), encoding="utf-8") as handle:
                self.assertIn("$schema", json.load(handle))


if __name__ == "__main__":
    unittest.main()
