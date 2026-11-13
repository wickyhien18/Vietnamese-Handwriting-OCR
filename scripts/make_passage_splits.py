"""Split the dataset by passage, so that no text is shared between train, val and test.

Why: in UIT-HWDB-line many pages are different writers copying the same passage.
Splitting by page stops one writer's handwriting from leaking, but the text still
leaks: a val or test line can be a copy of a training line. This script groups pages
that carry the same passage and keeps every passage inside one split.

Steps:
  1. Two pages belong to the same passage when the share of their word windows
     (WINDOW words in a row, normalized like check_text_overlap.py) that they have in
     common reaches --link, measured against the smaller page. Links are joined
     transitively.
  2. The official test pages stay the test split, unchanged.
  3. Pages of the other splits that share a passage with a test page are left out of
     every split ("excluded"), so the test text never appears in training.
  4. Whole passages are drawn at random (--seed) from the remaining ones into val until
     val holds about --val-fraction of the remaining lines. The rest is train.

Inputs and outputs (formats of the data contracts do not change):
  --index      all labeled lines (dataset_index_all.jsonl); every record needs the fields
               id, gt_text, group_id (the page) and split (the official split)
  --out-index  the same records without the excluded pages (dataset_index.jsonl)
  --out-splits {id: split} for exactly the ids in --out-index (splits.json)
  --report     passages, their pages and roles, and the excluded pages (passages.json)

Usage:
  python scripts/make_passage_splits.py --index outputs/dataset_index_all.jsonl \\
      --out-index outputs/dataset_index.jsonl --out-splits outputs/splits.json \\
      --report outputs/passages.json
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_text_overlap import group_sort_key, normalize_tokens, windows  # noqa: E402

DEFAULT_WINDOW = 5
DEFAULT_LINK = 0.4
DEFAULT_VAL_FRACTION = 0.10
DEFAULT_SEED = 42
VAL_OVERSHOOT = 1.25  # a passage is skipped if val would end above target * this value
ROLES = ("train", "val", "test")


def read_index(path):
    records = []
    seen = set()
    with open(path, encoding="utf-8") as handle:
        for number, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                raise ValueError("index line %d is not valid JSON" % number)
            for key in ("id", "gt_text", "group_id", "split"):
                if not isinstance(record.get(key), str) or not record[key]:
                    raise ValueError("index line %d needs a non-empty string field '%s'" % (number, key))
            if record["id"] in seen:
                raise ValueError("duplicate id %r in the index" % record["id"])
            seen.add(record["id"])
            records.append(record)
    if not records:
        raise ValueError("the index has no records")
    return records


def group_pages(records):
    """Return ({page: [records]}, {page: official split})."""
    pages = defaultdict(list)
    official = {}
    for record in records:
        page = record["group_id"]
        pages[page].append(record)
        if official.setdefault(page, record["split"]) != record["split"]:
            raise ValueError("page %s has lines in more than one official split" % page)
    return pages, official


def find_passages(pages, size, link):
    """Cluster pages into passages; returns lists of pages in natural page order."""
    page_grams = {page: set() for page in pages}
    for page, items in pages.items():
        for item in items:
            page_grams[page].update(windows(normalize_tokens(item["gt_text"]), size))
    owners = defaultdict(list)
    for page, grams in page_grams.items():
        for gram in grams:
            owners[gram].append(page)
    shared = defaultdict(Counter)
    for group in owners.values():
        if len(group) > 1:
            for first in group:
                for second in group:
                    if first != second:
                        shared[first][second] += 1

    parent = {page: page for page in pages}

    def find(page):
        while parent[page] != page:
            parent[page] = parent[parent[page]]
            page = parent[page]
        return page

    for first, others in shared.items():
        for second, count in others.items():
            if count / min(len(page_grams[first]), len(page_grams[second])) >= link:
                root_a, root_b = find(first), find(second)
                if root_a != root_b:
                    keep, drop = sorted((root_a, root_b), key=group_sort_key)
                    parent[drop] = keep
    clusters = defaultdict(list)
    for page in pages:
        clusters[find(page)].append(page)
    result = [sorted(members, key=group_sort_key) for members in clusters.values()]
    result.sort(key=lambda members: group_sort_key(members[0]))
    return result


def assign_roles(clusters, pages, official, val_fraction, seed):
    """Return {page: 'train' | 'val' | 'test' | 'excluded'}."""
    def line_count(cluster):
        return sum(len(pages[page]) for page in cluster)

    test_pages = {page for page, split in official.items() if split == "test"}
    if not test_pages:
        raise ValueError("the index has no official test pages")
    roles = {}
    pool = []
    for cluster in clusters:
        if any(page in test_pages for page in cluster):
            for page in cluster:
                roles[page] = "test" if page in test_pages else "excluded"
        else:
            pool.append(cluster)
    if not pool:
        raise ValueError("no pages are left outside the passages of the test pages")

    target = val_fraction * sum(line_count(cluster) for cluster in pool)
    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    val_indexes = set()
    val_lines = 0
    for index in order:
        if val_lines >= target:
            break
        size = line_count(pool[index])
        if val_lines + size <= target * VAL_OVERSHOOT:
            val_indexes.add(index)
            val_lines += size
    if not val_indexes:
        raise ValueError("no whole passage fits the val target of about %d lines" % round(target))
    if len(val_indexes) == len(pool):
        raise ValueError("no passage is left for train")
    for index, cluster in enumerate(pool):
        for page in cluster:
            roles[page] = "val" if index in val_indexes else "train"
    return roles


def build_outputs(records, roles):
    kept = [record for record in records if roles[record["group_id"]] != "excluded"]
    splits = {record["id"]: roles[record["group_id"]] for record in kept}
    return kept, splits


def check_result(records, roles, clusters, splits):
    """Return a list of problems; an empty list means the split is consistent."""
    problems = []
    kept_ids = {record["id"] for record in records if roles[record["group_id"]] != "excluded"}
    if set(splits) != kept_ids:
        problems.append("splits do not cover exactly the kept lines")
    official_test = {record["id"] for record in records if record["split"] == "test"}
    if {line_id for line_id, split in splits.items() if split == "test"} != official_test:
        problems.append("the test lines differ from the official test lines")
    for cluster in clusters:
        used = {roles[page] for page in cluster if roles[page] != "excluded"}
        if len(used) > 1:
            problems.append("passage with pages %s sits in several splits: %s" % (cluster[:3], sorted(used)))
    for name in ROLES:
        if name not in splits.values():
            problems.append("split '%s' is empty" % name)
    return problems


def build_report(clusters, pages, roles, params):
    passages = {}
    for number, cluster in enumerate(clusters, 1):
        kinds = {roles[page] for page in cluster}
        role = "test" if "test" in kinds else next(iter(kinds))
        passages["p%03d" % number] = {
            "role": role,
            "pages": cluster,
            "excluded_pages": [page for page in cluster if roles[page] == "excluded"],
            "lines": sum(len(pages[page]) for page in cluster),
        }
    counts = {"pages": Counter(roles.values()), "lines": Counter(), "passages": Counter()}
    for page, role in roles.items():
        counts["lines"][role] += len(pages[page])
    for info in passages.values():
        counts["passages"][info["role"]] += 1
    return {
        "parameters": params,
        "counts": {key: dict(value) for key, value in counts.items()},
        "excluded_pages": sorted((p for p, r in roles.items() if r == "excluded"), key=group_sort_key),
        "passages": passages,
    }


def format_summary(report):
    counts = report["counts"]
    out = ["passage-level split",
           "passages: %d (%d of them contain a test page)"
           % (len(report["passages"]), counts["passages"].get("test", 0)),
           "  %-9s %6s %7s %9s" % ("split", "pages", "lines", "passages")]
    for name in ROLES:
        out.append("  %-9s %6d %7d %9d" % (name, counts["pages"].get(name, 0),
                                           counts["lines"].get(name, 0),
                                           counts["passages"].get(name, 0)))
    out.append("excluded (same passage as a test page): %d pages, %d lines"
               % (counts["pages"].get("excluded", 0), counts["lines"].get("excluded", 0)))
    return "\n".join(out)


def write_json_lines(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path, data, **options):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, **options)
        handle.write("\n")


def run(args):
    records = read_index(args.index)
    pages, official = group_pages(records)
    clusters = find_passages(pages, args.window, args.link)
    roles = assign_roles(clusters, pages, official, args.val_fraction, args.seed)
    kept, splits = build_outputs(records, roles)
    problems = check_result(records, roles, clusters, splits)
    if problems:
        raise ValueError("internal check failed: " + "; ".join(problems[:3]))
    params = {"window": args.window, "link": args.link,
              "val_fraction": args.val_fraction, "seed": args.seed}
    report = build_report(clusters, pages, roles, params)
    for path in (args.out_index, args.out_splits, args.report):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_json_lines(args.out_index, kept)
    write_json(args.out_splits, splits, sort_keys=True)
    write_json(args.report, report, indent=1)
    return "%s\nwrote %d records to %s\nwrote %s and %s" % (
        format_summary(report), len(kept), args.out_index, args.out_splits, args.report)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True, help="all labeled lines (dataset_index_all.jsonl)")
    parser.add_argument("--out-index", required=True, help="index without the excluded pages")
    parser.add_argument("--out-splits", required=True, help="splits.json for the kept lines")
    parser.add_argument("--report", required=True, help="passages report (passages.json)")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="words per window")
    parser.add_argument("--link", type=float, default=DEFAULT_LINK,
                        help="shared share of windows that joins two pages (0 to 1)")
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION,
                        help="share of the remaining lines used for val (0 to 1)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.window < 1:
        print("error: --window must be at least 1", file=sys.stderr)
        return 2
    if not 0 < args.link <= 1:
        print("error: --link must be above 0 and at most 1", file=sys.stderr)
        return 2
    if not 0 < args.val_fraction < 1:
        print("error: --val-fraction must be between 0 and 1", file=sys.stderr)
        return 2
    outputs = {os.path.abspath(p) for p in (args.out_index, args.out_splits, args.report)}
    if len(outputs) < 3 or os.path.abspath(args.index) in outputs:
        print("error: the input and the three output files must all be different files", file=sys.stderr)
        return 2
    try:
        print(run(args))
    except (OSError, ValueError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
