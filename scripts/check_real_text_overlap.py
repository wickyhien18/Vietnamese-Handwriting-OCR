"""Measure text overlap between the real handwriting test set and the training text.

check_text_overlap.py compares the val and test splits of an index with its train
split. This wrapper builds a temporary index in which every UIT-HWDB line is "train"
and the real handwriting lines are "test", then calls check_text_overlap.main(). No
file of the project is changed. Optional corpora (one sentence per line) are checked
in extra runs.

Usage (from the repository root):
  python scripts/check_real_text_overlap.py \
      --real-index data/handwriting-test/real_dataset_index.jsonl \
      --uit-index outputs/dataset_index.jsonl \
      --corpus data/corpus/viwiki_sentences.txt --corpus /path/to/train.txt
Read the "test" row of every table: exact, any and major, as in the report (section 4.5).
"""

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_text_overlap


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_temp_files(uit, real, tmp_dir):
    index_rows = [{"id": "uit_" + row["id"], "gt_text": row["gt_text"], "group_id": str(row.get("group_id") or "")}
                  for row in uit]
    index_rows += [{"id": row["id"], "gt_text": row["gt_text"], "group_id": str(row.get("group_id") or "")}
                   for row in real]
    splits = {"uit_" + row["id"]: "train" for row in uit}
    splits.update({row["id"]: "test" for row in real})
    index_path = os.path.join(tmp_dir, "index.jsonl")
    splits_path = os.path.join(tmp_dir, "splits.json")
    with open(index_path, "w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(splits_path, "w", encoding="utf-8") as handle:
        json.dump(splits, handle)
    return index_path, splits_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real-index", required=True)
    parser.add_argument("--uit-index", required=True, help="the full UIT-HWDB dataset_index.jsonl")
    parser.add_argument("--corpus", action="append", default=[], help="text file, one sentence per line; repeatable")
    parser.add_argument("--window", type=int, default=check_text_overlap.DEFAULT_WINDOW)
    parser.add_argument("--top", type=int, default=check_text_overlap.DEFAULT_TOP)
    parser.add_argument("--tmp-dir", default=None, help="default: a temporary folder")
    args = parser.parse_args()

    uit = read_jsonl(args.uit_index)
    real = read_jsonl(args.real_index)
    print("UIT lines: %d (all treated as train), real lines: %d (treated as test)" % (len(uit), len(real)))
    tmp_context = None
    tmp_dir = args.tmp_dir
    if tmp_dir is None:
        tmp_context = tempfile.TemporaryDirectory()
        tmp_dir = tmp_context.name
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        index_path, splits_path = write_temp_files(uit, real, tmp_dir)
        base = ["--index", index_path, "--splits", splits_path, "--window", str(args.window), "--top", str(args.top)]
        print("\n=== against UIT-HWDB ===")
        status = check_text_overlap.main(base)
        for corpus in args.corpus:
            print("\n=== against %s ===" % corpus)
            status = max(status, check_text_overlap.main(base + ["--corpus", corpus]))
    finally:
        if tmp_context is not None:
            tmp_context.cleanup()
    return status


if __name__ == "__main__":
    sys.exit(main())
