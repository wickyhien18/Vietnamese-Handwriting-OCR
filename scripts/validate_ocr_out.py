import json
import sys
import unicodedata

REQUIRED = ["id", "split", "image_path", "gt_text", "ocr_text"]
OPTIONAL = ["ocr_conf"]
SPLITS = {"train", "val", "test"}


def validate_record(rec: dict, line_no: int) -> list:
    errors = []
    for key in REQUIRED:
        if key not in rec:
            errors.append("line %d: missing field '%s'" % (line_no, key))
    for key in rec:
        if key not in REQUIRED and key not in OPTIONAL:
            errors.append("line %d: unexpected field '%s'" % (line_no, key))
    if rec.get("split") not in SPLITS:
        errors.append("line %d: split must be one of %s" % (line_no, sorted(SPLITS)))
    for key in ("gt_text", "ocr_text"):
        value = rec.get(key)
        if isinstance(value, str) and value != unicodedata.normalize("NFC", value):
            errors.append("line %d: '%s' is not NFC-normalized" % (line_no, key))
    conf = rec.get("ocr_conf")
    if conf is not None and not isinstance(conf, list):
        errors.append("line %d: ocr_conf must be a list" % line_no)
    return errors


def validate_file(path: str) -> list:
    errors = []
    seen_ids = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append("line %d: invalid JSON (%s)" % (line_no, exc))
                continue
            errors.extend(validate_record(rec, line_no))
            rec_id = rec.get("id")
            if rec_id in seen_ids:
                errors.append("line %d: duplicate id '%s'" % (line_no, rec_id))
            seen_ids.add(rec_id)
    return errors


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/validate_ocr_out.py <file.jsonl>")
        sys.exit(2)
    problems = validate_file(sys.argv[1])
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print("OK")
