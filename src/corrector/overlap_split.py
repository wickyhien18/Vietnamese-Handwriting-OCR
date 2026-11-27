"""Experiment E5: does the corrector do better on lines whose text it may have seen?

Each OCR val line is flagged when its label shares at least one run of 5 words with the
corrector's training text (the "any" column of scripts/check_text_overlap.py, same normalization,
same windows). The lines are then split into two groups, and for each group we report
CER before and after correction and the fixed / broken / net counts of evaluate_on_ocr
(syllables paired by alignment, as in Table 8).

    flags = overlap_flags([gt for gt, _ in lines], corpus_path, window=5, corpus_lines=300_000)
    rows = split_report(corrector, lines, flags)
"""

import os
import tempfile
from itertools import islice
from types import SimpleNamespace

from scripts.check_text_overlap import compare_with_corpus, normalize_tokens, summarize
from src.corrector.ocr_adapt import evaluate_on_ocr

MIN_LINES = 20  # below this, a group is too small to conclude anything


def overlap_flags(labels, corpus_path, window=5, corpus_lines=None):
    """For every label: (flag, windows, matched). flag = at least one window also occurs in the corpus.

    corpus_lines limits the corpus to its first N lines (for example the 300,000 sentences the
    tagger was trained on). Also returns the summary of check_text_overlap.py for these lines.
    """
    targets = []
    for text in labels:
        tokens = normalize_tokens(text)
        targets.append(SimpleNamespace(tokens=tokens, key=" ".join(tokens)))

    path = corpus_path
    temp = None
    if corpus_lines is not None:
        temp = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False)
        with open(corpus_path, encoding="utf-8-sig") as source:
            temp.writelines(islice(source, corpus_lines))
        temp.close()
        path = temp.name
    try:
        results, n_corpus = compare_with_corpus(targets, path, window)
    finally:
        if temp is not None:
            os.unlink(temp.name)
    flags = [(r.matched > 0, r.windows, r.matched) for r in results]
    stats = summarize(results)
    stats["corpus_lines"] = n_corpus
    return flags, stats


def split_report(corrector, lines, flags, tau=0.0):
    """One row per group ("overlap", "no overlap"): lines, CER before/after, fixed, broken, net.

    lines: (label, OCR text) pairs as from ocr_lines(); flags: from overlap_flags(), same order.
    A group with fewer than MIN_LINES lines gets a note instead of a conclusion.
    """
    groups = {"overlap": [], "no overlap": []}
    short = {"overlap": 0, "no overlap": 0}
    for line, (flag, windows, _) in zip(lines, flags):
        name = "overlap" if flag else "no overlap"
        groups[name].append(line)
        short[name] += windows == 0
    rows = {}
    for name, group in groups.items():
        if not group:
            rows[name] = {"lines": 0, "note": "no lines"}
            continue
        r = evaluate_on_ocr(corrector, group, tau=tau, n_examples=0)
        c = r["counts"]
        rows[name] = {
            "lines": len(group),
            "lines_shorter_than_window": short[name],
            "cer_before": r["cer_before"], "cer_after": r["cer_after"],
            "wer_before": r["wer_before"], "wer_after": r["wer_after"],
            "fixed": c.get("fixed", 0), "broken": c.get("broken", 0), "net": r["net_fixed"],
            "paired_syllables": sum(c.get(k, 0) for k in ("kept_right", "broken", "fixed", "still_wrong")),
            "note": "too few lines to conclude" if len(group) < MIN_LINES else "",
        }
    return rows


def print_split(rows):
    header = f"{'group':<12}{'lines':>6}{'CER no corr.':>14}{'CER after':>11}{'fixed':>7}{'broken':>8}{'net':>6}  note"
    print(header)
    print("-" * len(header))
    for name, r in rows.items():
        if r["lines"] == 0:
            print(f"{name:<12}{0:>6}  {r['note']}")
            continue
        print(f"{name:<12}{r['lines']:>6}{r['cer_before']:>14.2%}{r['cer_after']:>11.2%}"
              f"{r['fixed']:>7}{r['broken']:>8}{r['net']:>6}  {r['note']}")
