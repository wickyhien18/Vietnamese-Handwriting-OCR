# Data contracts

Everything the three roles exchange goes through these files. A contract changes only when all three people agree.

| # | Contract | Written by | Read by | Format | Check |
|---|---|---|---|---|---|
| 1 | `dataset_index.jsonl` | Hiển (`scripts/make_passage_splits.py`, from `dataset_index_all.jsonl` written by `scripts/build_index.py`) | A, B, C | `dataset_index.schema.json` | `python scripts/validate_dataset_index.py --index <file>` |
| 2 | `splits.json` | Hiển (`scripts/make_passage_splits.py`) | A, B, C | `splits.schema.json` | `python scripts/validate_dataset_index.py --index <file> --splits <file>` |
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

## Which lines the two files contain (since 12 Nov)

In UIT-HWDB-line many pages are different writers copying the same passage, so splitting by page alone leaves the same text in train and in val or test. Since 12 Nov `dataset_index.jsonl` and `splits.json` are made by `scripts/make_passage_splits.py`:

- the official test (6 pages, 201 lines) is unchanged;
- pages that share a passage with a test page are left out of both files, so the test text never appears in training. All labeled lines stay in `outputs/dataset_index_all.jsonl`, and `outputs/passages.json` lists the passages and the pages that were left out;
- val is made of whole passages, so a passage never sits in two splits.

Counts at the time of writing: train 5,369, val 636, test 201 lines. The index has 6,206 lines; `dataset_index_all.jsonl` has 7,229, of which 1,023 are left out. Val has 9 passages and test has 5, so scores on them carry a wide uncertainty.

## Frozen since 8 Nov

The fields and keys above do not change, and the formats did not change on 12 Nov: only which lines the two files contain. Two rules in `corrector_api.md` (same number of syllables, only diacritics change) are clarifications added on 9 Nov and wait for Khánh's confirmation.
