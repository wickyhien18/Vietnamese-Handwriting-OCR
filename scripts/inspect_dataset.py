"""Print a compact report about a dataset folder or .zip file, so its layout can be understood.

Usage:
  python scripts/inspect_dataset.py path/to/uit_hwdb_folder_or_zip [--depth 3] [--samples 5]

It does not change or extract anything. Copy the whole output into the group chat or the project notes.
"""

import argparse
import os
import re
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TEXT_EXT = {".txt", ".json", ".jsonl", ".csv", ".tsv", ".xml", ".label", ".gt", ".lst"}
SPLIT_NAMES = {"train", "training", "val", "valid", "validation", "dev", "test", "testing"}
MAX_TEXT_BYTES = 2 * 1024 * 1024


def list_entries(path):
    """Return a sorted list of (relative_path_with_forward_slashes, size_in_bytes)."""
    entries = []
    if os.path.isdir(path):
        for base, _, files in os.walk(path):
            for name in files:
                full = os.path.join(base, name)
                rel = os.path.relpath(full, path).replace(os.sep, "/")
                entries.append((rel, os.path.getsize(full)))
    elif path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    entries.append((info.filename, info.file_size))
    else:
        sys.exit("ERROR: give a folder or a .zip file (other archive types: extract them first)")
    return sorted(entries)


def read_bytes(path, rel):
    if os.path.isdir(path):
        with open(os.path.join(path, rel.replace("/", os.sep)), "rb") as handle:
            return handle.read(MAX_TEXT_BYTES)
    with zipfile.ZipFile(path) as archive:
        with archive.open(rel) as handle:
            return handle.read(MAX_TEXT_BYTES)


def ext_of(rel):
    return os.path.splitext(rel)[1].lower()


def top_exts(counter, n=3):
    return ", ".join("%s x%d" % (ext or "(none)", count) for ext, count in counter.most_common(n))


def print_layout(entries, depth):
    print("\n== Folder layout (depth %d) ==" % depth)
    per_dir = defaultdict(Counter)
    for rel, _ in entries:
        parts = rel.split("/")
        folder = "/".join(parts[:-1][:depth]) or "."
        per_dir[folder][ext_of(rel)] += 1
    for folder in sorted(per_dir):
        total = sum(per_dir[folder].values())
        print("%-50s files: %-7d %s" % (folder[:50], total, top_exts(per_dir[folder])))


def print_split_names(entries):
    found = set()
    for rel, _ in entries:
        for part in rel.lower().split("/")[:-1]:
            if part in SPLIT_NAMES:
                found.add(part)
    print("\n== Split-like folder names ==")
    print(", ".join(sorted(found)) if found else "none found (the dataset may ship without an official split)")


def print_images(entries, samples):
    images = [rel for rel, _ in entries if ext_of(rel) in IMAGE_EXT]
    print("\n== Images ==")
    print("total: %d" % len(images))
    for rel in images[:samples]:
        print("  " + rel)
    stems = [os.path.splitext(os.path.basename(rel))[0] for rel in images]
    if stems and all(re.fullmatch(r"\d+", stem) for stem in stems):
        print("hint: file names are plain numbers, so they carry no writer information")
    else:
        first_tokens = Counter()
        for stem in stems:
            tokens = [t for t in re.split(r"[_\-\s]+", stem) if t]
            if len(tokens) >= 2:
                first_tokens[tokens[0]] += 1
        if first_tokens:
            print("heuristic writer hint: first name token has %d distinct values over %d files; most common: %s" % (
                len(first_tokens), sum(first_tokens.values()),
                ", ".join("%s x%d" % kv for kv in first_tokens.most_common(5))))


def print_text_files(path, entries, samples):
    texts = [rel for rel, _ in entries if ext_of(rel) in TEXT_EXT]
    print("\n== Text-like files (labels, indexes) ==")
    print("total: %d" % len(texts))
    for rel in texts[:samples]:
        raw = read_bytes(path, rel)
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        not_nfc = sum(1 for line in lines if line != unicodedata.normalize("NFC", line))
        print("\n-- %s  (%d lines read, %d not NFC)" % (rel, len(lines), not_nfc))
        for line in lines[:3]:
            print("   " + line[:200])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    entries = list_entries(args.path)
    ext_count = Counter(ext_of(rel) for rel, _ in entries)
    print("== Summary ==")
    print("source: %s" % args.path)
    print("files: %d" % len(entries))
    print("extensions: %s" % top_exts(ext_count, 10))
    print_layout(entries, args.depth)
    print_split_names(entries)
    print_images(entries, args.samples)
    print_text_files(args.path, entries, args.samples)


if __name__ == "__main__":
    main()
