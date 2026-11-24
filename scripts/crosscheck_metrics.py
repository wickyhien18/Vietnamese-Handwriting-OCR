"""Cross-check src.eval.align.edit_distance against the editdistance library.

Reads an OCR output file (ocr_out_beam.jsonl: id, split, gt_text, ocr_text), keeps one
split (default val) and compares the two edit distances on every line. It also prints
the pooled CER computed with the project function and a word-level error rate computed
with the library (the project's word-level function is not exercised here).

Usage (from the repository root):
  pip install editdistance
  python scripts/crosscheck_metrics.py --ocr-out outputs/ocr_out_beam.jsonl --split val
Exit code 1 if any line disagrees.
"""

import argparse
import json
import os
import sys

import editdistance

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.eval.align import edit_distance


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ocr-out", required=True)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    with open(args.ocr_out, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = [row for row in rows if row.get("split") == args.split]
    if not rows:
        raise SystemExit("no lines with split %r in %s" % (args.split, args.ocr_out))

    mismatches = 0
    char_edits = char_total = word_edits = word_total = 0
    for row in rows:
        ref, hyp = row["gt_text"], row["ocr_text"]
        d_project = edit_distance(ref, hyp)
        d_library = editdistance.eval(ref, hyp)
        if d_project != d_library:
            mismatches += 1
            print("mismatch %s: project %d, library %d" % (row["id"], d_project, d_library))
        char_edits += d_project
        char_total += len(ref)
        word_edits += editdistance.eval(ref.split(), hyp.split())
        word_total += len(ref.split())
    print("split %s: %d lines, mismatches: %d" % (args.split, len(rows), mismatches))
    print("pooled CER (project function): %.2f%% (%d edits over %d characters)" % (100.0 * char_edits / char_total, char_edits, char_total))
    print("pooled WER (editdistance on word lists): %.2f%%" % (100.0 * word_edits / word_total))
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
