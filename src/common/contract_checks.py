"""Checks for the project data contracts. Pure Python, each function returns a list of problems."""

import unicodedata

from src.common.text_utils import strip_diacritics

SPLITS = ("train", "val", "test")
INDEX_REQUIRED = ("id", "image_path", "gt_text")
INDEX_OPTIONAL = ("split", "writer_id", "group_id")


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def check_correct_output(text_in, out):
    """Check the result of correct(text_in) against contracts/corrector_api.md."""
    if not isinstance(out, dict):
        return ["output must be a dict"]
    if set(out.keys()) != {"text", "conf"}:
        return ["output keys must be exactly 'text' and 'conf', got %s" % sorted(out.keys())]
    text_out = out["text"]
    conf = out["conf"]
    if not isinstance(text_out, str):
        return ["'text' must be a string"]
    if not isinstance(conf, list):
        return ["'conf' must be a list"]

    problems = []
    if text_out != unicodedata.normalize("NFC", text_out):
        problems.append("'text' is not NFC-normalized")
    tokens_in = unicodedata.normalize("NFC", text_in).split()
    tokens_out = text_out.split()
    if len(tokens_out) != len(tokens_in):
        problems.append("syllable count changed: %d in, %d out" % (len(tokens_in), len(tokens_out)))
    if len(conf) != len(tokens_out):
        problems.append("'conf' has %d values for %d syllables" % (len(conf), len(tokens_out)))
    for position, value in enumerate(conf):
        if not _is_number(value) or not 0.0 <= value <= 1.0:
            problems.append("conf[%d] must be a number between 0 and 1" % position)
            break
    if len(tokens_in) == len(tokens_out):
        for position, (token_in, token_out) in enumerate(zip(tokens_in, tokens_out)):
            if strip_diacritics(token_in) != strip_diacritics(token_out):
                problems.append("syllable %d changed more than its diacritics: %r -> %r" % (position, token_in, token_out))
                break
    return problems


def check_dataset_index_records(records):
    """Check a list of dataset_index records (contracts/dataset_index.schema.json)."""
    problems = []
    seen = set()
    with_split = 0
    with_writer = 0
    with_group = 0
    for number, rec in enumerate(records, start=1):
        label = "record %d" % number
        for key in INDEX_REQUIRED:
            if not isinstance(rec.get(key), str) or not rec.get(key):
                problems.append("%s: '%s' must be a non-empty string" % (label, key))
        for key in rec:
            if key not in INDEX_REQUIRED and key not in INDEX_OPTIONAL:
                problems.append("%s: unexpected field '%s'" % (label, key))
        rec_id = rec.get("id")
        if rec_id in seen:
            problems.append("%s: duplicate id %r" % (label, rec_id))
        seen.add(rec_id)
        path = rec.get("image_path")
        if isinstance(path, str) and (path.startswith("/") or "\\" in path or ".." in path.split("/")):
            problems.append("%s: image_path must be relative with forward slashes: %r" % (label, path))
        text = rec.get("gt_text")
        if isinstance(text, str) and text:
            if text != unicodedata.normalize("NFC", text):
                problems.append("%s: gt_text is not NFC" % label)
            if text != " ".join(text.split()):
                problems.append("%s: gt_text must use single spaces and no leading or trailing space" % label)
        if "split" in rec:
            with_split += 1
            if rec["split"] not in SPLITS:
                problems.append("%s: split must be one of %s" % (label, list(SPLITS)))
        if "writer_id" in rec:
            with_writer += 1
            if not isinstance(rec["writer_id"], str):
                problems.append("%s: writer_id must be a string" % label)
        if "group_id" in rec:
            with_group += 1
            if not isinstance(rec["group_id"], str):
                problems.append("%s: group_id must be a string" % label)
    if records and 0 < with_split < len(records):
        problems.append("split is present for only some records")
    if records and 0 < with_writer < len(records):
        problems.append("writer_id is present for only some records")
    if records and 0 < with_group < len(records):
        problems.append("group_id is present for only some records")
    return problems


def check_splits(splits, index_ids=None):
    """Check splits.json; with index_ids also check that it covers exactly those ids."""
    problems = []
    if not isinstance(splits, dict):
        return ["splits must be a JSON object"]
    for key, value in splits.items():
        if value not in SPLITS:
            problems.append("id %r has invalid split %r" % (key, value))
    present = set(splits.values())
    for name in SPLITS:
        if name not in present:
            problems.append("split '%s' is empty" % name)
    if index_ids is not None:
        ids = set(index_ids)
        missing = sorted(ids - set(splits))
        extra = sorted(set(splits) - ids)
        if missing:
            problems.append("%d index ids are missing from splits, for example %s" % (len(missing), missing[:3]))
        if extra:
            problems.append("%d split ids are not in the index, for example %s" % (len(extra), extra[:3]))
    return problems
