"""Score the OCR output on the clean subset of the real handwriting test set.

The clean subset is every line whose status in label_review.csv is "khop" or
"khac_hoa_thuong" (the line matches the label, or differs only in letter case).
The main number ignores letter case and punctuation; a case-sensitive number is
reported next to it. Every edit distance is computed twice (the project's
src.eval.align.edit_distance and the editdistance library) and the script stops
if the two disagree on any line.

Usage (from the repository root):
  pip install editdistance
  python scripts/eval_real_clean.py \
      --review-csv data/handwriting-test/label_review.csv \
      --ocr data/handwriting-test/real_ocr_out_beam_run1.jsonl \
      --out data/handwriting-test/real_eval_clean.json
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import unicodedata

import editdistance

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.eval.align import edit_distance

STATUSES = ("khop", "xuong_dong", "khac_chu", "khac_hoa_thuong")
KEEP = ("khop", "khac_hoa_thuong")


def strip_punctuation(text):
    """Replace punctuation and symbols by a space, then collapse spaces."""
    text = unicodedata.normalize("NFC", text)
    chars = [" " if unicodedata.category(ch)[0] in ("P", "S") else ch for ch in text]
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def primary(text):
    return strip_punctuation(text).lower()


def case_sensitive(text):
    return strip_punctuation(text)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rates(subset, norm):
    """Return (CER percent, WER percent, number of reference characters)."""
    char_err = char_len = word_err = word_len = 0
    for row in subset:
        ref, hyp = norm(row["gt_text"]), norm(row["ocr_text"])
        d_library = editdistance.eval(ref, hyp)
        d_project = edit_distance(ref, hyp)
        if d_library != d_project:
            raise SystemExit("edit distance mismatch on %s: library %d, project %d" % (row["id"], d_library, d_project))
        char_err += d_library
        char_len += len(ref)
        word_err += editdistance.eval(ref.split(), hyp.split())
        word_len += len(ref.split())
    return 100.0 * char_err / char_len, 100.0 * word_err / word_len, char_len


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--ocr", required=True, help="real_ocr_out_beam*.jsonl with id, gt_text, ocr_text")
    parser.add_argument("--out", required=True, help="where to write the JSON summary")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()

    with open(args.review_csv, "r", encoding="utf-8-sig", newline="") as handle:
        review = {row["id"]: row for row in csv.DictReader(handle)}
    bad = [rid for rid, row in review.items() if row["status"] not in STATUSES]
    if bad:
        raise SystemExit("invalid or empty status for: %s" % ", ".join(bad[:10]))
    rows = read_jsonl(args.ocr)
    missing = [row["id"] for row in rows if row["id"] not in review]
    if missing:
        raise SystemExit("ids missing from the review table: %s" % ", ".join(missing[:10]))

    clean = [row for row in rows if review[row["id"]]["status"] in KEEP]
    result = {"lines": len(rows), "clean_lines": len(clean), "status_counts": {}}
    for record in review.values():
        result["status_counts"][record["status"]] = result["status_counts"].get(record["status"], 0) + 1

    cer, wer, chars = rates(clean, primary)
    print("clean subset: %d of %d lines, %d reference characters" % (len(clean), len(rows), chars))
    print("PRIMARY (lowercase, no punctuation): CER %.1f  WER %.1f" % (cer, wer))
    result["primary"] = {"cer": cer, "wer": wer, "chars": chars}
    cer_cs, wer_cs, _ = rates(clean, case_sensitive)
    print("case-sensitive (no punctuation):     CER %.1f  WER %.1f" % (cer_cs, wer_cs))
    result["case_sensitive"] = {"cer": cer_cs, "wer": wer_cs}

    result["by_writer"] = {}
    for writer in sorted({review[row["id"]]["writer_id"] for row in clean}):
        subset = [row for row in clean if review[row["id"]]["writer_id"] == writer]
        c, w, _ = rates(subset, primary)
        print("writer %-6s lines %2d  CER %.1f  WER %.1f" % (writer, len(subset), c, w))
        result["by_writer"][writer] = {"lines": len(subset), "cer": c, "wer": w}

    groups = {}
    for row in clean:
        groups.setdefault(review[row["id"]]["passage"], []).append(row)
    parts = {}
    result["by_passage"] = {}
    for passage in sorted(groups):
        c, _, n = rates(groups[passage], primary)
        parts[passage] = (c * n / 100.0, n)
        print("%s lines %d  CER %.1f" % (passage, len(groups[passage]), c))
        result["by_passage"][passage] = {"lines": len(groups[passage]), "cer": c}

    random.seed(args.seed)
    keys = sorted(parts)
    values = []
    for _ in range(args.draws):
        pick = [random.choice(keys) for _ in keys]
        values.append(100.0 * sum(parts[k][0] for k in pick) / sum(parts[k][1] for k in pick))
    values.sort()
    low = values[max(int(0.025 * args.draws) - 1, 0)]
    high = values[min(int(0.975 * args.draws) - 1, args.draws - 1)]
    print("bootstrap by passage (%d clusters, %d draws): 95%% CI %.1f to %.1f" % (len(keys), args.draws, low, high))
    result["bootstrap_ci95"] = [low, high]

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=1)
    print("saved %s (editdistance and src.eval.align agreed on every line)" % args.out)


if __name__ == "__main__":
    main()
