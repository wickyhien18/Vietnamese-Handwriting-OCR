"""Build real_dataset_index.jsonl for the hand-collected (real handwriting) test set.

Input: labels.json, a list of objects with the keys id, image, text, passage, writer.
The writer key in the file is NOT used: only two people wrote the lines, so the writer is
derived from the line number (lines 1..DUONG_LAST by Duong, the rest by Hien).

Output: one JSON object per line with the fields id, image_path, gt_text, split, writer_id,
group_id. These are exactly the fields of dataset_index.schema.json (no extra fields).
split is always "test" because the frozen contract only allows train/val/test. This file must
stay separate from outputs/dataset_index.jsonl and from splits.json.

Usage (from the repository root):
  python scripts/build_real_index.py --labels data/handwriting-test/labels.json \
      --image-dir /content/handwriting-test --out outputs/real_dataset_index.jsonl
  python scripts/validate_dataset_index.py --index outputs/real_dataset_index.jsonl

Also provides strip_punctuation(), used for the main metric on the real set (the labels have
no punctuation, so punctuation printed by the OCR model is removed before scoring).
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter

DUONG_LAST = 45
REQUIRED_KEYS = ("id", "image", "text", "passage")


def normalize_label(text):
    """NFC, single spaces, no leading or trailing space."""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_punctuation(text):
    """Replace every punctuation or symbol character by a space, then normalize spaces.

    A space (not an empty string) keeps word boundaries: "a,b" becomes "a b".
    """
    text = unicodedata.normalize("NFC", text)
    chars = [" " if unicodedata.category(ch)[0] in ("P", "S") else ch for ch in text]
    return normalize_label("".join(chars))


def line_number(record_id):
    match = re.search(r"(\d+)$", record_id)
    if match is None:
        raise ValueError("id has no trailing number: %r" % record_id)
    return int(match.group(1))


def build_records(labels, duong_last, image_dir=None):
    """Return (records, problems). records is empty when there are problems."""
    problems = []
    records = []
    seen = set()
    for position, item in enumerate(labels, start=1):
        missing = [key for key in REQUIRED_KEYS if key not in item]
        if missing:
            problems.append("entry %d: missing keys %s" % (position, missing))
            continue
        rec_id = item["id"]
        if rec_id in seen:
            problems.append("%s: duplicate id" % rec_id)
        seen.add(rec_id)

        text = normalize_label(item["text"])
        if not text:
            problems.append("%s: empty text" % rec_id)
        bad = sorted({ch for ch in text if not (ch.isalpha() or ch == " ")})
        if bad:
            problems.append("%s: unexpected characters %s (labels must be letters and spaces)" % (rec_id, bad))

        if image_dir is not None and not os.path.isfile(os.path.join(image_dir, item["image"])):
            problems.append("%s: image not found: %s" % (rec_id, item["image"]))

        try:
            number = line_number(rec_id)
        except ValueError as error:
            problems.append(str(error))
            continue
        records.append({
            "id": rec_id,
            "image_path": item["image"],
            "gt_text": text,
            "split": "test",
            "writer_id": "duong" if number <= duong_last else "hien",
            "group_id": item["passage"],
        })
    records.sort(key=lambda rec: rec["id"])
    return (records if not problems else []), problems


def summarize(records):
    lines = ["lines: %d" % len(records)]
    writers = Counter(rec["writer_id"] for rec in records)
    lines.append("lines per writer: %s" % dict(sorted(writers.items())))
    groups = Counter(rec["group_id"] for rec in records)
    lines.append("passages: %d, lines per passage: %s" % (len(groups), dict(sorted(groups.items()))))
    by_group = {}
    for rec in records:
        by_group.setdefault(rec["group_id"], set()).add(rec["writer_id"])
    mixed = sorted(group for group, who in by_group.items() if len(who) > 1)
    lines.append("passages written by both writers: %s" % (mixed if mixed else "none"))
    total_chars = sum(len(rec["gt_text"]) for rec in records)
    total_words = sum(len(rec["gt_text"].split()) for rec in records)
    lines.append("label characters: %d, label words: %d" % (total_chars, total_words))
    return "\n".join(lines)


def self_test():
    assert strip_punctuation("xa, rộng.") == "xa rộng"
    assert strip_punctuation("a,b") == "a b"
    assert strip_punctuation("85% (ở) \"x\"") == "85 ở x"
    assert normalize_label("  a   b ") == "a b"
    labels = [
        {"id": "real_001", "image": "a.jpg", "text": "có vẻ", "passage": "P01", "writer": "W1"},
        {"id": "real_046", "image": "b.jpg", "text": "bà ba", "passage": "P05", "writer": "W2"},
    ]
    records, problems = build_records(labels, 45)
    assert not problems and [r["writer_id"] for r in records] == ["duong", "hien"]
    _, problems = build_records([{"id": "real_001", "image": "a.jpg", "text": "a, b", "passage": "P01"}], 45)
    assert problems
    print("self-test OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels")
    parser.add_argument("--out")
    parser.add_argument("--image-dir", default=None, help="if given, every image file must exist there")
    parser.add_argument("--duong-last", type=int, default=DUONG_LAST,
                        help="last line number written by Duong (default %d)" % DUONG_LAST)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.labels or not args.out:
        parser.error("--labels and --out are required")

    with open(args.labels, "r", encoding="utf-8") as handle:
        labels = json.load(handle)
    records, problems = build_records(labels, args.duong_last, args.image_dir)
    if problems:
        print("%d problems, first 20:" % len(problems))
        print("\n".join(problems[:20]))
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(summarize(records))
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
