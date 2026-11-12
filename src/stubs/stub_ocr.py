"""Fake OCR: builds ocr_out.jsonl from ground-truth text plus diacritic noise.

Lets role C run the whole evaluation before the real OCR exists.

Usage (from the repository root):
  python -m src.stubs.stub_ocr --index outputs/dataset_index.jsonl \
      --splits outputs/splits.json --out outputs/ocr_out_stub.jsonl
"""

import argparse
import json
import random
import unicodedata

from src.common.text_utils import normalize_nfc, strip_diacritics

TONE_MARKS = ["\u0300", "\u0301", "\u0303", "\u0309", "\u0323"]


def has_tone(syllable):
    return any(ch in TONE_MARKS for ch in unicodedata.normalize("NFD", syllable))


def swap_tone(syllable, rng):
    swapped = False
    chars = []
    for ch in unicodedata.normalize("NFD", syllable):
        if ch in TONE_MARKS and not swapped:
            ch = rng.choice([mark for mark in TONE_MARKS if mark != ch])
            swapped = True
        chars.append(ch)
    return unicodedata.normalize("NFC", "".join(chars))


def corrupt_syllable(syllable, rng):
    if has_tone(syllable) and rng.random() < 0.5:
        return swap_tone(syllable, rng)
    return strip_diacritics(syllable)


def corrupt_text(text, rng, p):
    syllables = text.split(" ")
    out = [corrupt_syllable(s, rng) if rng.random() < p else s for s in syllables]
    return " ".join(out)


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_stub_records(index_records, splits, p=0.15, seed="stub-v1"):
    out = []
    skipped = 0
    for rec in index_records:
        rec_id = rec["id"]
        if rec_id not in splits:
            skipped += 1
            continue
        rng = random.Random("%s:%s" % (seed, rec_id))
        gt_text = normalize_nfc(rec["gt_text"])
        out.append({
            "id": rec_id,
            "split": splits[rec_id],
            "image_path": rec["image_path"],
            "gt_text": gt_text,
            "ocr_text": corrupt_text(gt_text, rng, p),
        })
    return out, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--p", type=float, default=0.15, help="probability that a syllable is corrupted")
    parser.add_argument("--seed", default="stub-v1")
    args = parser.parse_args()

    index_records = read_jsonl(args.index)
    with open(args.splits, "r", encoding="utf-8") as handle:
        splits = json.load(handle)
    records, skipped = build_stub_records(index_records, splits, args.p, args.seed)
    with open(args.out, "w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("wrote %d records to %s (%d skipped: id not in splits)" % (len(records), args.out, skipped))


if __name__ == "__main__":
    main()
