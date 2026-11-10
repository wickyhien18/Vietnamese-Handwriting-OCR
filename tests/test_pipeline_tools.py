import json
import os
import random
import tempfile
import unittest

from scripts.make_splits import make_splits
from scripts.validate_ocr_out import validate_file
from src.stubs.stub_ocr import build_stub_records, corrupt_text, read_jsonl

LINE_1 = "nhi\u1ec1u b\u1ea1n \u0111\u1ecdc"
LINE_2 = "s\u1ef1 th\u1eadt \u0111\u01b0\u1ee3c n\u00f3i"


def make_records(n, writer_every=None):
    records = []
    for k in range(n):
        rec = {"id": "line_%06d" % k, "image_path": "x/%d.png" % k, "gt_text": LINE_1 if k % 2 else LINE_2}
        if writer_every:
            rec["writer_id"] = "w%03d" % (k // writer_every)
        records.append(rec)
    return records


class SplitsTest(unittest.TestCase):
    def test_by_id_is_deterministic_and_order_independent(self):
        records = make_records(500)
        first, mode = make_splits(records)
        shuffled = list(records)
        random.Random(1).shuffle(shuffled)
        second, _ = make_splits(shuffled)
        self.assertEqual(mode, "by_id")
        self.assertEqual(first, second)

    def test_by_id_ratios_are_close(self):
        result, _ = make_splits(make_records(2000))
        counts = {name: list(result.values()).count(name) for name in ("train", "val", "test")}
        self.assertTrue(1500 < counts["train"] < 1700)
        self.assertTrue(100 < counts["val"] < 300)
        self.assertTrue(100 < counts["test"] < 300)

    def test_writer_never_crosses_splits(self):
        records = make_records(600, writer_every=10)
        result, mode = make_splits(records)
        self.assertEqual(mode, "by_writer")
        by_writer = {}
        for rec in records:
            by_writer.setdefault(rec["writer_id"], set()).add(result[rec["id"]])
        self.assertTrue(all(len(splits) == 1 for splits in by_writer.values()))

    def test_official_split_is_kept(self):
        records = make_records(30)
        for k, rec in enumerate(records):
            rec["split"] = ("train", "val", "test")[k % 3]
        result, mode = make_splits(records)
        self.assertEqual(mode, "official")
        self.assertEqual(result["line_000001"], "val")

    def test_official_without_val_carves_val_from_train(self):
        records = make_records(1000)
        for k, rec in enumerate(records):
            rec["split"] = "train" if k % 5 else "test"
        result, mode = make_splits(records)
        self.assertEqual(mode, "official+carved_val")
        self.assertIn("val", result.values())
        self.assertTrue(all(result[rec["id"]] == "test" for rec in records if rec["split"] == "test"))

    def test_duplicate_ids_rejected(self):
        records = make_records(3)
        records.append(dict(records[0]))
        with self.assertRaises(ValueError):
            make_splits(records)

    def test_partial_writer_ids_rejected(self):
        records = make_records(4)
        records[0]["writer_id"] = "w1"
        with self.assertRaises(ValueError):
            make_splits(records)


class StubOcrTest(unittest.TestCase):
    def test_corrupt_text_keeps_syllable_count(self):
        rng = random.Random(0)
        out = corrupt_text(LINE_1, rng, 1.0)
        self.assertEqual(len(out.split(" ")), len(LINE_1.split(" ")))
        self.assertNotEqual(out, LINE_1)

    def test_probability_zero_keeps_text(self):
        self.assertEqual(corrupt_text(LINE_1, random.Random(0), 0.0), LINE_1)

    def test_stub_output_passes_validator(self):
        records = make_records(20)
        splits, _ = make_splits(records)
        stub, skipped = build_stub_records(records, splits, p=0.5)
        self.assertEqual(skipped, 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ocr_out.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                for rec in stub:
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.assertEqual(validate_file(path), [])
            self.assertEqual(len(read_jsonl(path)), 20)

    def test_stub_is_deterministic(self):
        records = make_records(10)
        splits, _ = make_splits(records)
        first, _ = build_stub_records(records, splits)
        second, _ = build_stub_records(records, splits)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
