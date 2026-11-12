"""Measure how much of the val and test text already appears in the training text.

Why this matters: lines of one passage (for example two writers copying the same
article with different line breaks) can end up in different splits. The val or test
score is then too optimistic, because the text was seen during training.

Usage:
  python scripts/check_text_overlap.py --index outputs/dataset_index.jsonl --splits outputs/splits.json
  python scripts/check_text_overlap.py --index ... --splits ... --corpus data/corpus/viwiki_sentences.txt

Texts are normalized before they are compared: NFC, lower case, punctuation removed.
Punctuation must go because dataset labels may have spaces around punctuation
while ordinary text does not.

Three numbers are reported for every split (a "window" is WINDOW words in a row):
  exact  share of lines whose whole text also appears as a reference line
  any    share of lines with at least one shared window
  major  share of lines where at least half of the windows are shared

Lines shorter than WINDOW words have no window: they count only for "exact".

The three-level rule uses "major" (percent of val or test lines):
  below 10        keep the split and note the overlap in the report
  10 to 25        keep the split, report CER separately for overlapping lines
  above 25        group the duplicate pages and re-split (val), or drop/report (test)

With --corpus the same numbers are computed against a text file with one sentence
per line (for example the corpus built for the corrector). The file is streamed, so
a very large corpus does not need much memory.
"""

import argparse
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field

SPLITS = ("train", "val", "test")
DEFAULT_WINDOW = 5
DEFAULT_TOP = 10
KEEP_BELOW = 10.0
RESPLIT_ABOVE = 25.0


@dataclass
class Line:
    line_id: str
    split: str
    group: str
    tokens: list

    @property
    def key(self):
        return " ".join(self.tokens)


@dataclass
class LineResult:
    line: Line
    exact: bool
    windows: int
    matched: int
    sources: Counter = field(default_factory=Counter)


def normalize_tokens(text):
    """NFC, lower case, every character that is not a letter, digit or mark becomes a space."""
    text = unicodedata.normalize("NFC", text).lower()
    chars = [ch if unicodedata.category(ch)[0] in "LNM" else " " for ch in text]
    return "".join(chars).split()


def windows(tokens, size):
    """All runs of `size` consecutive tokens, each joined into one string."""
    return [" ".join(tokens[i:i + size]) for i in range(len(tokens) - size + 1)]


def percent(count, total):
    return 100.0 * count / total if total else 0.0


def load_lines(index_path, splits_path):
    """Return (lines, ids_missing_from_splits, ids_missing_from_index)."""
    with open(splits_path, encoding="utf-8") as handle:
        splits = json.load(handle)
    if not isinstance(splits, dict):
        raise ValueError("splits file must be a JSON object {id: split}")
    bad = sorted({str(v) for v in splits.values() if v not in SPLITS})
    if bad:
        raise ValueError("unknown split values in splits file: %s" % ", ".join(bad[:5]))

    lines = []
    missing_from_splits = []
    seen_ids = set()
    with open(index_path, encoding="utf-8") as handle:
        for number, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                raise ValueError("index line %d is not valid JSON" % number)
            line_id = record.get("id")
            text = record.get("gt_text")
            if line_id is None or text is None:
                raise ValueError("index line %d needs the fields 'id' and 'gt_text'" % number)
            seen_ids.add(line_id)
            split = splits.get(line_id)
            if split is None:
                missing_from_splits.append(line_id)
                continue
            group = str(record.get("group_id") or "")
            lines.append(Line(line_id, split, group, normalize_tokens(text)))
    missing_from_index = sorted(set(splits) - seen_ids)
    return lines, missing_from_splits, missing_from_index


def build_train_reference(train_lines, size):
    keys = set()
    gram_groups = defaultdict(set)
    for line in train_lines:
        keys.add(line.key)
        for gram in windows(line.tokens, size):
            gram_groups[gram].add(line.group)
    return keys, gram_groups


def compare_with_train(targets, keys, gram_groups, size):
    results = []
    for line in targets:
        grams = windows(line.tokens, size)
        matched = 0
        sources = Counter()
        for gram in grams:
            groups = gram_groups.get(gram)
            if groups:
                matched += 1
                for group in groups:
                    sources[group] += 1
        results.append(LineResult(line, line.key in keys, len(grams), matched, sources))
    return results


def compare_with_corpus(targets, corpus_path, size):
    """Stream the corpus and look only for the windows and lines that occur in `targets`."""
    target_keys = {line.key for line in targets}
    target_grams = set()
    for line in targets:
        target_grams.update(windows(line.tokens, size))
    found_keys = set()
    found_grams = set()
    corpus_lines = 0
    with open(corpus_path, encoding="utf-8-sig") as handle:
        for raw in handle:
            tokens = normalize_tokens(raw)
            if not tokens:
                continue
            corpus_lines += 1
            key = " ".join(tokens)
            if key in target_keys:
                found_keys.add(key)
            for gram in windows(tokens, size):
                if gram in target_grams:
                    found_grams.add(gram)
    results = []
    for line in targets:
        grams = windows(line.tokens, size)
        matched = sum(1 for gram in grams if gram in found_grams)
        results.append(LineResult(line, line.key in found_keys, len(grams), matched))
    return results, corpus_lines


def summarize(results):
    eligible = [r for r in results if r.windows > 0]
    stats = {
        "lines": len(results),
        "eligible": len(eligible),
        "exact": sum(1 for r in results if r.exact),
        "any": sum(1 for r in eligible if r.matched > 0),
        "major": sum(1 for r in eligible if r.matched * 2 >= r.windows),
    }
    stats["exact_pct"] = percent(stats["exact"], stats["lines"])
    stats["any_pct"] = percent(stats["any"], stats["eligible"])
    stats["major_pct"] = percent(stats["major"], stats["eligible"])
    return stats


def verdict(value, split):
    """Map a 'major' percentage to a level and the action text for the split."""
    if value < KEEP_BELOW:
        return "keep", "keep the split and note the overlap in the report"
    if value <= RESPLIT_ABOVE:
        return "report", "keep the split and report CER separately for overlapping and other lines"
    if split == "test":
        return "resplit", ("the official test split is fixed: drop the overlapping lines from "
                           "train or report CER separately for them")
    return "resplit", "group the duplicate pages into one group and re-split val, then tell the team"


def group_rows(results, top):
    by_group = defaultdict(list)
    for result in results:
        by_group[result.line.group].append(result)
    rows = []
    for group, items in by_group.items():
        sources = Counter()
        for item in items:
            sources.update(item.sources)
        best = sources.most_common(1)
        rows.append({"group": group, "stats": summarize(items),
                     "top_source": best[0] if best else None})
    rows.sort(key=lambda row: (-row["stats"]["major_pct"], -row["stats"]["lines"], row["group"]))
    return rows[:top]


def format_stats_table(title, rows):
    out = [title,
           "  %-6s %6s %9s %9s %9s %9s" % ("split", "lines", "exact", "any", "major", "eligible")]
    for name, stats in rows:
        out.append("  %-6s %6d %8.1f%% %8.1f%% %8.1f%% %9d" % (
            name, stats["lines"], stats["exact_pct"], stats["any_pct"],
            stats["major_pct"], stats["eligible"]))
    return out


def format_group_table(name, rows):
    out = ["  most overlapping %s pages:" % name,
           "    %-8s %6s %8s %8s %8s  %s" % ("group", "lines", "exact", "any", "major", "top train page")]
    for row in rows:
        stats = row["stats"]
        source = "-" if row["top_source"] is None else "%s (%d windows)" % row["top_source"]
        out.append("    %-8s %6d %7.1f%% %7.1f%% %7.1f%%  %s" % (
            row["group"] or "-", stats["lines"], stats["exact_pct"], stats["any_pct"],
            stats["major_pct"], source))
    return out


def run(index_path, splits_path, corpus_path, size, top):
    lines, missing_from_splits, missing_from_index = load_lines(index_path, splits_path)
    empty = sum(1 for line in lines if not line.tokens)
    by_split = {name: [l for l in lines if l.split == name and l.tokens] for name in SPLITS}
    if not by_split["train"]:
        raise ValueError("no training lines found: nothing to compare with")

    out = ["text overlap report (window = %d words)" % size,
           "lines per split: " + ", ".join("%s=%d" % (n, len(by_split[n])) for n in SPLITS)]
    if empty:
        out.append("warning: %d lines have no usable words and are skipped" % empty)
    if missing_from_splits:
        out.append("warning: %d index ids are not in the splits file (first: %s)"
                   % (len(missing_from_splits), missing_from_splits[0]))
    if missing_from_index:
        out.append("warning: %d split ids are not in the index (first: %s)"
                   % (len(missing_from_index), missing_from_index[0]))

    keys, gram_groups = build_train_reference(by_split["train"], size)
    train_results = {}
    rows = []
    for name in ("val", "test"):
        if not by_split[name]:
            out.append("note: no %s lines, skipped" % name)
            continue
        results = compare_with_train(by_split[name], keys, gram_groups, size)
        train_results[name] = results
        rows.append((name, summarize(results)))

    out.append("")
    out += format_stats_table("against the training text (%d lines)" % len(by_split["train"]), rows)
    for name, stats in rows:
        if stats["eligible"] == 0:
            out.append("verdict %s: no line has %d or more words, cannot judge" % (name, size))
            continue
        level, action = verdict(stats["major_pct"], name)
        out.append("verdict %s: %s (major = %.1f%%): %s" % (name, level.upper(), stats["major_pct"], action))
    for name, _ in rows:
        out.append("")
        out += format_group_table(name, group_rows(train_results[name], top))

    if corpus_path:
        targets = by_split["val"] + by_split["test"]
        results, corpus_lines = compare_with_corpus(targets, corpus_path, size)
        corpus_rows = []
        for name in ("val", "test"):
            subset = [r for r in results if r.line.split == name]
            if subset:
                corpus_rows.append((name, summarize(subset)))
        out.append("")
        out += format_stats_table("against the corpus %s (%d lines)" % (corpus_path, corpus_lines),
                                  corpus_rows)
        exact_total = sum(stats["exact"] for _, stats in corpus_rows)
        if exact_total:
            out.append("corpus: %d val/test lines occur word for word in the corpus; remove "
                       "those sentences before training or scoring the corrector" % exact_total)
        else:
            out.append("corpus: no val/test line occurs word for word in the corpus")
    return "\n".join(out)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True, help="dataset_index.jsonl")
    parser.add_argument("--splits", required=True, help="splits.json ({id: split})")
    parser.add_argument("--corpus", help="optional text file, one sentence per line")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="words per window")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP, help="pages listed per split")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.window < 1 or args.top < 1:
        print("error: --window and --top must be at least 1", file=sys.stderr)
        return 2
    try:
        print(run(args.index, args.splits, args.corpus, args.window, args.top))
    except (OSError, ValueError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
