import json
import os
import tempfile
import unittest

from src.common.index_io import read_index, read_splits, select_split


class IndexIoTest(unittest.TestCase):
    def test_round_trip_and_select(self):
        records = [
            {"id": "a", "image_path": "x/a.png", "gt_text": "one"},
            {"id": "b", "image_path": "x/b.png", "gt_text": "two"},
            {"id": "c", "image_path": "x/c.png", "gt_text": "three"},
        ]
        splits = {"a": "train", "b": "val", "c": "train"}
        with tempfile.TemporaryDirectory() as tmp:
            index_path = os.path.join(tmp, "index.jsonl")
            splits_path = os.path.join(tmp, "splits.json")
            with open(index_path, "w", encoding="utf-8") as handle:
                for rec in records:
                    handle.write(json.dumps(rec) + "\n")
                handle.write("\n")
            with open(splits_path, "w", encoding="utf-8") as handle:
                json.dump(splits, handle)
            index = read_index(index_path)
            loaded = read_splits(splits_path)
        self.assertEqual(index, records)
        self.assertEqual([r["id"] for r in select_split(index, loaded, "train")], ["a", "c"])
        self.assertEqual(select_split(index, loaded, "test"), [])


if __name__ == "__main__":
    unittest.main()
