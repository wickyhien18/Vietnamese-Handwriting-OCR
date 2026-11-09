"""Shared helpers to read the dataset index and the splits file.

dataset_index.jsonl : one JSON object per line with id, image_path, gt_text
                      (image_path is relative to the dataset root folder).
splits.json         : {"<id>": "train" | "val" | "test"}
"""

import json


def read_index(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_splits(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def select_split(index, splits, split):
    """Return the index records that belong to one split ('train', 'val' or 'test')."""
    return [rec for rec in index if splits.get(rec["id"]) == split]
