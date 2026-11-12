"""Create splits.json (id -> train/val/test) from a dataset index.

Input: a JSONL index, one record per line, with the fields
  id (required, unique), image_path, gt_text,
  group_id (optional), writer_id (optional), split (optional, official split of the dataset).

group_id marks lines that belong together (for example one page) and must stay in one split.

Rules, in order:
  1. Every record has an official split -> use it. If there is no val, carve val out of train:
       with group_id or writer_id: move the LAST train groups (natural order of the group key)
       to val, whole groups only; neighbouring groups often share text, so a contiguous block
       leaks less than random groups; without groups: by a hash of the id.
  2. Every record has group_id or writer_id -> split by group, so one page or one person never
                                           appears in two splits (hash of the group key).
  3. Otherwise                          -> split by a hash of the id.
The result does not depend on record order, so re-running gives an identical file.

Usage (from the repository root):
  python scripts/make_splits.py --index outputs/dataset_index.jsonl --out outputs/splits.json
"""

import argparse
import hashlib
import json
import re
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


def group_of(rec):
    for key in ("group_id", "writer_id"):
        if rec.get(key) is not None:
            return rec[key]
    return None


def natural_key(value):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value))]


def carve_val_by_group(records, result, train_pct, val_pct):
    """Move the last train groups (natural order) to val until val has about val/(train+val) of train."""
    train_groups = {}
    for rec in records:
        if rec["split"] == "train":
            train_groups.setdefault(group_of(rec), []).append(rec["id"])
    total = sum(len(ids) for ids in train_groups.values())
    target = round(total * val_pct / (train_pct + val_pct))
    moved = 0
    for group in sorted(train_groups, key=natural_key, reverse=True):
        if moved >= target:
            break
        for rec_id in train_groups[group]:
            result[rec_id] = "val"
        moved += len(train_groups[group])


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
    has_group = [group_of(rec) is not None for rec in records]

    if all(s in SPLITS for s in official):
        mode = "official"
        result = {rec["id"]: rec["split"] for rec in records}
        if "val" not in set(official):
            if all(has_group):
                mode = "official+carved_val_by_group"
                carve_val_by_group(records, result, train_pct, val_pct)
            else:
                mode = "official+carved_val"
                # Move about val/(train+val) of the train records to val, chosen by hash of the id.
                threshold = round(100 * val_pct / (train_pct + val_pct))
                for rec in records:
                    if rec["split"] == "train" and bucket(rec["id"], salt) < threshold:
                        result[rec["id"]] = "val"
    elif any(s is not None for s in official):
        raise ValueError("official split present for only some records")
    elif all(has_group):
        mode = "by_group" if any(rec.get("group_id") is not None for rec in records) else "by_writer"
        result = {
            rec["id"]: split_from_bucket(bucket(group_of(rec), salt), train_pct, val_pct)
            for rec in records
        }
    elif any(has_group):
        raise ValueError("group_id or writer_id present for only some records")
    else:
        mode = "by_id"
        result = {rec["id"]: split_from_bucket(bucket(rec["id"], salt), train_pct, val_pct) for rec in records}

    return result, mode


def report(result, mode, records=None):
    counts = Counter(result.values())
    total = len(result)
    lines = ["mode: %s" % mode, "total: %d" % total]
    groups = {name: set() for name in SPLITS}
    if records:
        for rec in records:
            if group_of(rec) is not None and rec["id"] in result:
                groups[result[rec["id"]]].add(group_of(rec))
    for name in SPLITS:
        share = 100.0 * counts[name] / total if total else 0.0
        extra = "  groups: %d" % len(groups[name]) if groups[name] else ""
        lines.append("%-5s %7d  (%.1f%%)%s" % (name, counts[name], share, extra))
    shared = (groups["train"] & groups["val"]) | (groups["train"] & groups["test"]) | (groups["val"] & groups["test"])
    if any(groups.values()):
        lines.append("groups shared between splits: %d" % len(shared))
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
    print(report(result, mode, records))


if __name__ == "__main__":
    main()
