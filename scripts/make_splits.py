"""Create splits.json (id -> train/val/test) from a dataset index.

Input: a JSONL index, one record per line, with the fields
  id (required, unique), image_path, gt_text,
  writer_id (optional), split (optional, official split of the dataset).

Rules, in order:
  1. Every record has an official split -> use it (if there is no val, carve val out of train).
  2. Every record has writer_id         -> split by writer, so one person's handwriting never
                                           appears in two splits.
  3. Otherwise                          -> split by a hash of the id.
The result does not depend on record order, so re-running gives an identical file.

Usage (from the repository root):
  python scripts/make_splits.py --index outputs/dataset_index.jsonl --out outputs/splits.json
"""

import argparse
import hashlib
import json
from collections import Counter

SPLITS = ("train", "val", "test")


def bucket(key, salt):
    digest = hashlib.md5(("%s:%s" % (salt, key)).encode("utf-8")).hexdigest()
    return int(digest, 16) % 100


def split_from_bucket(value, train_pct, val_pct):
    if value < train_pct:
        return "train"
    if value < train_pct + val_pct:
        return "val"
    return "test"


def read_index(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def make_splits(records, salt="vho-splits-v1", train_pct=80, val_pct=10):
    ids = [rec["id"] for rec in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids in the index")

    official = [rec.get("split") for rec in records]
    has_writer = [rec.get("writer_id") is not None for rec in records]

    if all(s in SPLITS for s in official):
        mode = "official"
        result = {rec["id"]: rec["split"] for rec in records}
        if "val" not in set(official):
            mode = "official+carved_val"
            # Move about val/(train+val) of the train records to val, chosen by hash.
            threshold = round(100 * val_pct / (train_pct + val_pct))
            for rec in records:
                key = rec.get("writer_id") if rec.get("writer_id") is not None else rec["id"]
                if rec["split"] == "train" and bucket(key, salt) < threshold:
                    result[rec["id"]] = "val"
    elif any(s is not None for s in official):
        raise ValueError("official split present for only some records")
    elif all(has_writer):
        mode = "by_writer"
        result = {
            rec["id"]: split_from_bucket(bucket(rec["writer_id"], salt), train_pct, val_pct)
            for rec in records
        }
    elif any(has_writer):
        raise ValueError("writer_id present for only some records")
    else:
        mode = "by_id"
        result = {rec["id"]: split_from_bucket(bucket(rec["id"], salt), train_pct, val_pct) for rec in records}

    return result, mode


def report(result, mode):
    counts = Counter(result.values())
    total = len(result)
    lines = ["mode: %s" % mode, "total: %d" % total]
    for name in SPLITS:
        share = 100.0 * counts[name] / total if total else 0.0
        lines.append("%-5s %7d  (%.1f%%)" % (name, counts[name], share))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--salt", default="vho-splits-v1")
    parser.add_argument("--train", type=int, default=80)
    parser.add_argument("--val", type=int, default=10)
    args = parser.parse_args()

    records = read_index(args.index)
    result, mode = make_splits(records, args.salt, args.train, args.val)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True, indent=1)
    print(report(result, mode))


if __name__ == "__main__":
    main()
