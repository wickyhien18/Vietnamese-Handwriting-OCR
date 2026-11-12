"""Validate dataset_index.jsonl and, optionally, splits.json.

Usage (from the repository root):
  python scripts/validate_dataset_index.py --index outputs/dataset_index.jsonl
  python scripts/validate_dataset_index.py --index outputs/dataset_index.jsonl --splits outputs/splits.json
"""

import argparse
import json
import os
import sys
from collections import Counter

# make the repository root importable when the file is run as a script
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.common.contract_checks import check_dataset_index_records, check_splits
from src.common.index_io import read_index, read_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True)
    parser.add_argument("--splits", default=None)
    args = parser.parse_args()

    try:
        records = read_index(args.index)
    except (OSError, json.JSONDecodeError) as error:
        print("ERROR: cannot read the index: %s" % error)
        sys.exit(1)
    problems = check_dataset_index_records(records)
    print("index records: %d" % len(records))
    if records and "split" in records[0]:
        print("official split counts: %s" % dict(Counter(rec.get("split") for rec in records)))
    if records and "writer_id" in records[0]:
        print("distinct writers: %d" % len({rec.get("writer_id") for rec in records}))

    if args.splits:
        splits = read_splits(args.splits)
        problems += check_splits(splits, [rec["id"] for rec in records if "id" in rec])
        print("split counts: %s" % dict(Counter(splits.values())))

    if problems:
        print("%d problems, first 20:" % len(problems))
        print("\n".join(problems[:20]))
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
