"""Print the most frequent character-level OCR confusions (hand-written, no external library).

The script reads an ocr_out.jsonl file, aligns every label line with its OCR line at
character level (src.eval.align) and counts the edits by kind:
  substitution  label character -> OCR character, grouped by the error label of classify_errors
  deletion      a label character that is missing from the OCR output
  insertion     an extra character in the OCR output

The totals per error label are the same numbers that the hypothesis gate uses, so the
two outputs can be compared directly.

Usage (run from the repo root):
  python scripts/print_confusions.py --ocr-out outputs/ocr_out.jsonl --split val --top 15
  python scripts/print_confusions.py --ocr-out outputs/ocr_out.jsonl --out-json confusions_val.json
"""

import argparse
import glob
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.common.text_utils import normalize_nfc  # noqa: E402
from src.eval.align import DEL, INS, MATCH, SUB, align  # noqa: E402


def load_classifier():
    """Import classify_errors, or find it anywhere in the repo if the layout differs."""
    try:
        from src.eval import classify_errors
        return classify_errors
    except ImportError:
        hits = sorted(
            path for path in glob.glob(os.path.join(REPO_ROOT, "**", "classify_errors.py"), recursive=True)
            if os.sep + ".git" + os.sep not in path
        )
        if not hits:
            raise SystemExit("classify_errors.py was not found in the repo")
        spec = importlib.util.spec_from_file_location("classify_errors", hits[0])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


classify_errors = load_classifier()
SPACE_NAME = "<space>"
MISSING_NAME = "<missing>"


def show(char):
    return SPACE_NAME if char == " " else char


def word_around(text, index):
    """Return the whitespace-delimited word that contains position index."""
    if not text:
        return ""
    index = max(0, min(index, len(text) - 1))
    start = index
    while start > 0 and text[start - 1] != " ":
        start -= 1
    end = index
    while end < len(text) and text[end] != " ":
        end += 1
    return text[start:end]


def read_rows(path, split):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("id", "split", "gt_text", "ocr_text"):
                if key not in row:
                    raise SystemExit("line %d of %s has no '%s'" % (number, path, key))
            if split == "all" or row["split"] == split:
                rows.append(row)
    return rows


def count_confusions(rows, keep_raw_spacing):
    stats = {
        "lines": len(rows),
        "label_chars": 0,
        "errors": 0,
        "label_totals": Counter(),
        "subs": defaultdict(Counter),
        "dels": Counter(),
        "ins": Counter(),
        "examples": {},
        "ref_chars": Counter(),
        "ref_errors": Counter(),
        "ref_targets": defaultdict(Counter),
    }
    for row in rows:
        ref = " ".join(normalize_nfc(row["gt_text"]).split())
        hyp = normalize_nfc(row["ocr_text"])
        if not keep_raw_spacing:
            hyp = " ".join(hyp.split())
        stats["label_chars"] += len(ref)
        ref_pos = hyp_pos = 0
        for kind, ref_char, hyp_char in align(ref, hyp):
            if kind == MATCH:
                stats["ref_chars"][ref_char] += 1
                ref_pos += 1
                hyp_pos += 1
                continue
            label = classify_errors.classify_op(kind, ref_char, hyp_char)
            stats["errors"] += 1
            stats["label_totals"][label] += 1
            if kind == SUB:
                key = (ref_char, hyp_char)
                stats["subs"][label][key] += 1
                stats["examples"].setdefault(("sub", label, key), (word_around(ref, ref_pos), word_around(hyp, hyp_pos)))
                stats["ref_chars"][ref_char] += 1
                stats["ref_errors"][ref_char] += 1
                stats["ref_targets"][ref_char][hyp_char] += 1
                ref_pos += 1
                hyp_pos += 1
            elif kind == DEL:
                stats["dels"][ref_char] += 1
                stats["examples"].setdefault(("del", label, ref_char), (word_around(ref, ref_pos), ""))
                stats["ref_chars"][ref_char] += 1
                stats["ref_errors"][ref_char] += 1
                stats["ref_targets"][ref_char][MISSING_NAME] += 1
                ref_pos += 1
            else:
                stats["ins"][hyp_char] += 1
                stats["examples"].setdefault(("ins", label, hyp_char), ("", word_around(hyp, hyp_pos)))
                hyp_pos += 1
    return stats


def print_pair_table(title, counter, group_total, examples, example_key, top):
    print("\n%s" % title)
    if not counter:
        print("  (none)")
        return
    print("  %4s %6s %8s  %-22s %s" % ("rank", "count", "share", "label -> OCR", "example (label word -> OCR word)"))
    for rank, ((ref_char, hyp_char), count) in enumerate(counter.most_common(top), start=1):
        example = examples.get(example_key(ref_char, hyp_char), ("", ""))
        print("  %4d %6d %7.1f%%  %-22s %s -> %s" % (
            rank, count, 100.0 * count / group_total, "%s -> %s" % (show(ref_char), show(hyp_char)),
            example[0], example[1]))


def print_char_table(title, counter, group_total, examples, kind, top):
    print("\n%s" % title)
    if not counter:
        print("  (none)")
        return
    print("  %4s %6s %8s  %-12s %s" % ("rank", "count", "share", "character", "example"))
    for rank, (char, count) in enumerate(counter.most_common(top), start=1):
        label = classify_errors.classify_op(DEL if kind == "del" else INS, char if kind == "del" else None,
                                            char if kind == "ins" else None)
        example = examples.get((kind, label, char), ("", ""))
        shown = example[0] if kind == "del" else example[1]
        print("  %4d %6d %7.1f%%  %-12s %s" % (rank, count, 100.0 * count / group_total, show(char), shown))


def main():
    parser = argparse.ArgumentParser(description="Print the most frequent character-level OCR confusions.")
    parser.add_argument("--ocr-out", required=True, help="path to ocr_out.jsonl")
    parser.add_argument("--split", default="val", choices=["train", "val", "test", "all"], help="split to analyse (default: val)")
    parser.add_argument("--top", type=int, default=15, help="rows per table (default: 15)")
    parser.add_argument("--raw", action="store_true", help="keep the raw whitespace of ocr_text (default: collapse repeated spaces)")
    parser.add_argument("--min-occurrences", type=int, default=30, help="minimum label occurrences for the per-character table (default: 30)")
    parser.add_argument("--out-json", default=None, help="optional path to save the counts as JSON")
    args = parser.parse_args()

    rows = read_rows(args.ocr_out, args.split)
    if not rows:
        raise SystemExit("no rows for split '%s' in %s" % (args.split, args.ocr_out))
    stats = count_confusions(rows, args.raw)

    errors = stats["errors"]
    print("Character confusions: split=%s, lines=%d, whitespace=%s" % (args.split, stats["lines"], "raw" if args.raw else "normalized"))
    print("label characters: %d, character errors: %d, CER from these counts: %.2f%%" % (
        stats["label_chars"], errors, 100.0 * errors / stats["label_chars"]))
    print("\nerror labels (same classes as the hypothesis gate):")
    for label in classify_errors.ALL_LABELS:
        count = stats["label_totals"][label]
        print("  %-16s %6d  %5.1f%%" % (label, count, 100.0 * count / errors if errors else 0.0))
    diacritic = sum(stats["label_totals"][label] for label in classify_errors.DIACRITIC_LABELS)
    print("  diacritic share: %.1f%%" % (100.0 * diacritic / errors if errors else 0.0))

    for label in classify_errors.ALL_LABELS:
        counter = stats["subs"][label]
        if not counter:
            continue
        print_pair_table(
            "Substitutions, error label %s (%d edits, top %d)" % (label, sum(counter.values()), args.top),
            counter, sum(counter.values()), stats["examples"],
            lambda ref_char, hyp_char, label=label: ("sub", label, (ref_char, hyp_char)), args.top)

    print_char_table("Deletions: label characters missing from the OCR (%d edits)" % sum(stats["dels"].values()),
                     stats["dels"], max(sum(stats["dels"].values()), 1), stats["examples"], "del", args.top)
    print_char_table("Insertions: extra OCR characters (%d edits)" % sum(stats["ins"].values()),
                     stats["ins"], max(sum(stats["ins"].values()), 1), stats["examples"], "ins", args.top)

    rates = []
    for char, occurrences in stats["ref_chars"].items():
        if occurrences >= args.min_occurrences:
            wrong = stats["ref_errors"][char]
            rates.append((wrong / occurrences, char, occurrences, wrong))
    rates.sort(reverse=True)
    print("\nLeast reliable label characters (error rate, at least %d occurrences, top %d):" % (args.min_occurrences, args.top))
    print("  %4s %-10s %12s %8s %8s  %s" % ("rank", "character", "occurrences", "errors", "rate", "most common OCR output"))
    for rank, (rate, char, occurrences, wrong) in enumerate(rates[:args.top], start=1):
        target, target_count = stats["ref_targets"][char].most_common(1)[0]
        print("  %4d %-10s %12d %8d %7.1f%%  %s (%d times)" % (rank, show(char), occurrences, wrong, 100 * rate, show(target), target_count))

    if args.out_json:
        payload = {
            "split": args.split,
            "whitespace": "raw" if args.raw else "normalized",
            "lines": stats["lines"],
            "label_chars": stats["label_chars"],
            "errors": errors,
            "label_totals": dict(stats["label_totals"]),
            "substitutions": {
                label: [{"label_char": a, "ocr_char": b, "count": c} for (a, b), c in counter.most_common(200)]
                for label, counter in stats["subs"].items()
            },
            "deletions": [{"label_char": char, "count": c} for char, c in stats["dels"].most_common(200)],
            "insertions": [{"ocr_char": char, "count": c} for char, c in stats["ins"].most_common(200)],
        }
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
        print("\nwrote", args.out_json)


if __name__ == "__main__":
    main()
