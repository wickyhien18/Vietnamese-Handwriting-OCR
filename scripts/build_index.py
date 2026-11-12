"""Build dataset_index.jsonl from an extracted dataset folder (heuristic adapter).

It finds the images, finds the label files, matches them and writes one JSON record
per labeled image: id, image_path (relative to --root), gt_text, and when they can
be detected: split (from folder names such as train/val/test), writer_id and group_id.

Label formats understood:
  * JSON or JSONL: {"file.png": "text"}, {"file.png": {"text": ...}}, or a list of
    objects with a file field and a text field
  * line files (.txt .tsv .csv .lst .gt .label): "file.png<TAB>text", "file.png,text",
    or "file.png text"
  * one small .txt next to each image with the same name (sidecar label)

If fewer than 95 percent of the images get a label, it stops and prints what it saw,
so the adapter can be adjusted to the real layout.

Usage:
  python scripts/build_index.py --root /content/uit_hwdb --out outputs/dataset_index.jsonl
  optional: --writer-regex "writer(\\d+)"   (first group of the regex on the image path)
            --group-regex  "(?:train|test)_data/(\\d+)/"   lines of one group must stay in one split,
                           for example the page folder of the image
"""

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
LINE_EXT = {".txt", ".tsv", ".csv", ".lst", ".gt", ".label", ".labels"}
SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val", "dev": "val",
    "test": "test", "testing": "test",
}
KEY_FIELDS = ("image", "img", "file", "filename", "file_name", "path", "image_path", "name", "id")
TEXT_FIELDS = ("text", "label", "gt", "transcription", "transcript", "content", "annotation", "ground_truth")
HEADER_KEYS = {"filename", "file", "image", "img", "path", "name", "id", "file_name", "image_path"}
MIN_LABELED_SHARE = 0.95
AMBIGUOUS = object()


class IndexBuildError(Exception):
    pass


def normalize_text(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def ext_of(rel):
    return os.path.splitext(rel)[1].lower()


def strip_ext(rel):
    return os.path.splitext(rel)[0]


def list_files(root):
    found = []
    for base, _, names in os.walk(root):
        for name in names:
            found.append(os.path.relpath(os.path.join(base, name), root).replace(os.sep, "/"))
    return sorted(found)


def read_text_file(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


class ImageLookup:
    def __init__(self, images):
        self.by_key = defaultdict(list)
        for rel in images:
            stem_rel = strip_ext(rel)
            for key in {rel, stem_rel, os.path.basename(rel), os.path.basename(stem_rel)}:
                self.by_key[key].append(rel)

    def find(self, key, base_dir):
        key = key.strip().replace("\\", "/")
        while key.startswith("./"):
            key = key[2:]
        if not key:
            return None
        tries = []
        if base_dir:
            tries.append(base_dir + "/" + key)
        tries.append(key)
        tries.append(os.path.basename(key))
        for attempt in tries:
            candidates = self.by_key.get(attempt)
            if not candidates:
                continue
            if len(candidates) == 1:
                return candidates[0]
            if base_dir:
                inside = [c for c in candidates if c.startswith(base_dir + "/")]
                if len(inside) == 1:
                    return inside[0]
            return AMBIGUOUS
        return None


def first_field(item, names):
    lowered = {str(k).lower(): v for k, v in item.items()}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def parse_json_entries(data):
    entries = []
    if isinstance(data, dict):
        if data and all(isinstance(v, str) for v in data.values()):
            return list(data.items())
        for key, value in data.items():
            if isinstance(value, dict):
                text = first_field(value, TEXT_FIELDS)
                if isinstance(text, str):
                    entries.append((key, text))
        if entries:
            return entries
        for value in data.values():
            if isinstance(value, list):
                entries.extend(parse_json_entries(value))
        return entries
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                key = first_field(item, KEY_FIELDS)
                text = first_field(item, TEXT_FIELDS)
                if isinstance(key, str) and isinstance(text, str):
                    entries.append((key, text))
    return entries


def split_label_line(line):
    if "\t" in line:
        key, text = line.split("\t", 1)
    else:
        match = re.match(r"^([^\s,;|]+)\s*[,;|]\s*(.*)$", line)
        if not match:
            match = re.match(r"^(\S+)\s+(.*)$", line)
        if not match:
            return None
        key, text = match.group(1), match.group(2)
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    return key.strip().strip("\"'"), text


def parse_line_entries(content):
    entries = []
    for number, line in enumerate(content.splitlines()):
        if not line.strip():
            continue
        pair = split_label_line(line.strip())
        if pair is None:
            continue
        if number == 0 and pair[0].lower() in HEADER_KEYS:
            continue
        entries.append(pair)
    return entries


def detect_split(rel):
    for part in rel.lower().split("/")[:-1]:
        for token in re.split(r"[^a-z]+", part):
            if token in SPLIT_ALIASES:
                return SPLIT_ALIASES[token]
    return None


def collect_labels(root, files, lookup):
    """Return (labels, stats). labels maps image rel path -> text."""
    labels = {}
    stats = {"unmatched": [], "ambiguous": 0, "conflicts": 0, "label_files": []}

    def add(source, base_dir, key, text):
        found = lookup.find(key, base_dir)
        if found is None:
            stats["unmatched"].append((source, key))
        elif found is AMBIGUOUS:
            stats["ambiguous"] += 1
        elif found in labels and labels[found] != text:
            stats["conflicts"] += 1
        else:
            labels.setdefault(found, text)

    for rel in files:
        ext = ext_of(rel)
        base_dir = os.path.dirname(rel)
        full = os.path.join(root, rel.replace("/", os.sep))
        if ext in (".json", ".jsonl"):
            content = read_text_file(full)
            try:
                if ext == ".jsonl":
                    data = [json.loads(line) for line in content.splitlines() if line.strip()]
                else:
                    data = json.loads(content)
            except json.JSONDecodeError:
                continue
            entries = parse_json_entries(data)
            if entries:
                stats["label_files"].append((rel, entries[:3]))
                for key, text in entries:
                    add(rel, base_dir, key, text)
        elif ext in LINE_EXT:
            sidecar = lookup.by_key.get(strip_ext(rel), [])
            if len(sidecar) == 1 and sidecar[0] != rel and ext_of(sidecar[0]) in IMAGE_EXT:
                content = read_text_file(full).strip()
                if content and "\n" not in content:
                    stats["label_files"].append((rel, [(sidecar[0], content)]))
                    labels.setdefault(sidecar[0], content)
                    continue
            entries = parse_line_entries(read_text_file(full))
            if entries:
                stats["label_files"].append((rel, entries[:3]))
                for key, text in entries:
                    add(rel, base_dir, key, text)
    return labels, stats


def build_index(root, writer_regex=None, group_regex=None):
    files = list_files(root)
    images = [rel for rel in files if ext_of(rel) in IMAGE_EXT]
    if not images:
        raise IndexBuildError("no image files found under " + root)
    lookup = ImageLookup(images)
    labels, stats = collect_labels(root, files, lookup)

    records = []
    used_ids = set()
    missing_label = []
    empty_label = []
    for rel in images:
        raw = labels.get(rel)
        text = normalize_text(raw or "")
        if not text:
            (missing_label if raw is None else empty_label).append(rel)
            continue
        rec_id = re.sub(r"[^\w\-]+", "_", strip_ext(rel))
        if rec_id in used_ids:
            rec_id = re.sub(r"[^\w\-]+", "_", rel)
        used_ids.add(rec_id)
        rec = {"id": rec_id, "image_path": rel, "gt_text": text}
        split = detect_split(rel)
        if split:
            rec["split"] = split
        if writer_regex:
            match = re.search(writer_regex, rel)
            if match:
                rec["writer_id"] = match.group(1)
        if group_regex:
            match = re.search(group_regex, rel)
            if match:
                rec["group_id"] = match.group(1)
        records.append(rec)

    labeled_share = len(records) / len(images)
    report = {
        "images": len(images),
        "labeled": len(records),
        "unmatched_label_keys": len(stats["unmatched"]),
        "ambiguous": stats["ambiguous"],
        "conflicts": stats["conflicts"],
        "no_label_entry": len(missing_label),
        "empty_label": len(empty_label),
        "unlabeled_examples": (missing_label + empty_label)[:5],
    }
    if labeled_share < MIN_LABELED_SHARE:
        lines = ["only %d of %d images got a label (need at least %d%%)" % (
            len(records), len(images), int(MIN_LABELED_SHARE * 100))]
        lines.append("first image paths: " + ", ".join(images[:5]))
        for name, sample in stats["label_files"][:5]:
            lines.append("label file %s, first entries: %s" % (name, sample))
        if stats["unmatched"]:
            lines.append("unmatched label keys, first 5: %s" % stats["unmatched"][:5])
        raise IndexBuildError("\n".join(lines))

    with_split = [rec for rec in records if "split" in rec]
    if with_split and len(with_split) != len(records):
        for rec in records:
            rec.pop("split", None)
        report["split_note"] = "split folders found for only some images, official split ignored"
    with_writer = [rec for rec in records if "writer_id" in rec]
    if with_writer and len(with_writer) != len(records):
        for rec in records:
            rec.pop("writer_id", None)
        report["writer_note"] = "writer id found for only some images, ignored"
    with_group = [rec for rec in records if "group_id" in rec]
    if with_group and len(with_group) != len(records):
        for rec in records:
            rec.pop("group_id", None)
        report["group_note"] = "group id found for only some images, ignored"
    records.sort(key=lambda r: r["id"])
    report["split_counts"] = dict(Counter(rec.get("split", "none") for rec in records))
    report["writers"] = len({rec["writer_id"] for rec in records if "writer_id" in rec})
    report["groups"] = len({rec["group_id"] for rec in records if "group_id" in rec})
    return records, report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--writer-regex", default=None)
    parser.add_argument("--group-regex", default=None)
    args = parser.parse_args()
    try:
        records, report = build_index(args.root, args.writer_regex, args.group_regex)
    except IndexBuildError as error:
        print("ERROR: " + str(error))
        sys.exit(1)
    with open(args.out, "w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    for key, value in report.items():
        print("%-22s %s" % (key, value))
    print("wrote %d records to %s" % (len(records), args.out))
    for rec in records[:3]:
        print("  " + json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
