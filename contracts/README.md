# Data contracts

Everything the three roles exchange goes through these files. A contract changes only when all three people agree.

| # | Contract | Written by | Read by | Format | Check |
|---|---|---|---|---|---|
| 1 | `dataset_index.jsonl` | Hiển (`scripts/build_index.py`) | A, B, C | `dataset_index.schema.json` | `python scripts/validate_dataset_index.py --index <file>` |
| 2 | `splits.json` | Hiển (`scripts/make_splits.py`) | A, B, C | `splits.schema.json` | `python scripts/validate_dataset_index.py --index <file> --splits <file>` |
| 3 | `ocr_out.jsonl` | Dương | Hiển | `ocr_out.schema.json` | `python scripts/validate_ocr_out.py <file>` |
| 4 | `correct(text)` | Khánh | Hiển, demo | `corrector_api.md` | `check_correct_output` in `src/common/contract_checks.py` |

All text is Unicode NFC. Examples are in `examples/`.

## What each file contains, with concrete data

**`dataset_index.jsonl`**, one record per labeled image:

```json
{"id": "train_lines_000001", "image_path": "train/lines/000001.png", "gt_text": "nhiều bạn đọc đã bày tỏ", "split": "train"}
```

`image_path` is relative to the dataset root. On Colab the zip is extracted to `/content/uit_hwdb`, so the image is `/content/uit_hwdb/train/lines/000001.png`. `split`, `writer_id` and `group_id` appear only when they are known for every record. `group_id` marks lines that belong together (for UIT-HWDB-line, the page folder) and that must stay in one split.

**`splits.json`**, the same split for everybody:

```json
{"test_lines_000001": "test", "train_lines_000001": "train", "train_lines_000002": "val"}
```

**`ocr_out.jsonl`**, one record per image the OCR has read (the `id`, `split`, `image_path` and `gt_text` are copied from the index and the splits):

```json
{"id": "line_000001", "split": "val", "image_path": "uit_hwdb/line/000001.png", "gt_text": "nhiều bạn đọc đã bày tỏ", "ocr_text": "nhiều ban đọc đã bày tỏ"}
```

**`correct(text)`:** see `corrector_api.md`.

## Frozen since 8 Nov

The fields and keys above do not change. Two rules in `corrector_api.md` (same number of syllables, only diacritics change) are clarifications added on 9 Nov and wait for Khánh's confirmation.
