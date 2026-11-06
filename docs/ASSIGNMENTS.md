# Team assignments

Project: Vietnamese handwriting OCR + diacritic correction.
Final deadline: **30 Nov 2026**. The course has an oral exam (vấn đáp), so every person must be able to explain their own code line by line.

| Role | Person | Owns these folders | Main deliverable |
|---|---|---|---|
| A: OCR | **Dương** | `src/ocr/`, `notebooks/ocr_*` | `ocr_out.jsonl` (the OCR predictions) |
| B: Diacritic corrector | **Khánh** | `src/corrector/`, `notebooks/corrector_*` | `correct(text)` function + trained weights |
| C: Evaluation, integration, demo | **Hiển** | `src/eval/`, `demo/`, `scripts/`, `contracts/`, `docs/` | Evaluation table, demo, report skeleton |

Course rule: algorithms must be implemented by hand (sklearn only as a baseline for comparison). Each person has a list of "hand-written parts" below. Those are the parts the teacher will ask about.

---

## Role A: OCR (Dương)

**Goal:** turn an image of one handwritten line into text.

**Input:** line images from UIT-HWDB (Drive: `data/uit_hwdb/`) and `splits.json` (made by Hiển).
**Output:** `outputs/ocr_out.jsonl` on Drive, following `contracts/ocr_out.schema.json`; CER numbers on the validation split.

| Dates | Work |
|---|---|
| 6-8 Nov | Download the dataset. Check and report to the group: how many line images, whether the text labels contain full diacritics, whether a writer ID exists (needed for a clean train/test split), whether labels are NFC. |
| 9-15 Nov | Preprocessing (fixed height, grayscale, normalization). CRNN baseline (CNN, BiLSTM, CTC loss). **First `ocr_out.jsonl` by 13 Nov** (at least the validation split). |
| 16-22 Nov | Improve the model. Implement CTC greedy decoding and beam search by hand (numpy). Re-export `ocr_out.jsonl`. |
| 23-27 Nov | Error analysis: read at least 50 wrong lines and write down what goes wrong. Write the part A section of the report. |

**Hand-written parts (for the oral exam):** image preprocessing, CTC greedy decoding, CTC beam search.
**Definition of done:** `python scripts/validate_ocr_out.py outputs/ocr_out.jsonl` prints `OK`; CER on validation is reported; code is merged into `main`; you can explain how CTC decoding turns frame probabilities into text.
**If blocked:** if CER is still very high at the end of week 1, tell the group immediately. Fallbacks to discuss: stitch word images into lines; use an existing OCR for the predictions (this needs the teacher's approval because the OCR part would no longer be self-made).

---

## Role B: Diacritic corrector (Khánh)

**Goal:** given a Vietnamese sentence with wrong or missing diacritics, return the fixed sentence and one confidence number per syllable.

**Input:** a Vietnamese text corpus (Drive: `data/corpus/`). You create the training pairs yourself by corrupting clean text: remove diacritics, swap tone marks, swap vowel shape marks, each with some probability.
**Output:** the function `correct(text)` in `src/corrector/` following the contract (`{"text": str, "conf": [float]}`, one confidence per space-separated syllable); trained weights in Drive `checkpoints/corrector/`.

| Dates | Work |
|---|---|
| 6-8 Nov | Choose and download a corpus (for example Vietnamese Wikipedia or news). Clean it, make sure it is NFC, split into sentences. Report its size to the group. |
| 9-15 Nov | Noise generator. Frequency-table baseline (most common accented form of each syllable). Bigram + Viterbi by hand. Measure syllable accuracy, and separately on ambiguous syllables. |
| 16-22 Nov | Small transformer tagger (one accented syllable per input syllable). Confidence calibration by hand (temperature scaling). Final `correct()`. After week 1, adapt the noise generator to the real OCR error patterns that Hiển reports. |
| 23-27 Nov | Error analysis. Write the part B section of the report. |

**Hand-written parts (for the oral exam):** the noise generator, the Viterbi algorithm, temperature scaling.
**Definition of done:** `correct()` passes `tests/test_contracts.py` style checks (output keys, `len(conf) == len(text.split())`); accuracy numbers on held-out text; you can explain why Viterbi finds the best sequence.
**If blocked:** B does not depend on A for training. If real OCR outputs are late, keep training on synthetic noise and tell Hiển.

---

## Role C: Evaluation, integration, demo (Hiển)

**Goal:** measure honestly whether the corrector helps, and connect all parts.

**Input:** `splits.json`, `ocr_out.jsonl` (stub first, real later), `correct()` (stub first, real later).
**Output:** evaluation tables, a small demo, the report skeleton, contract validators.

| Dates | Work |
|---|---|
| By 8 Nov | Contracts frozen. `splits.json` created from the real dataset. |
| By 10 Nov | Stubs: fake OCR (ground truth with noise) and the identity corrector, so the pipeline runs before A and B finish. |
| By 12 Nov | Levenshtein alignment with backtrace, and the error classifier (tone mark, shape mark, base letter, spacing). |
| 13-15 Nov | **Hypothesis gate:** run the classifier on A's first real `ocr_out.jsonl` and report what share of OCR errors are diacritic errors. The group sets the threshold before looking at the number. |
| 16-22 Nov | `evaluate.py` with the comparison table (no correction, frequency table, Viterbi, transformer, transformer with confidence threshold), threshold sweep, demo. |
| 23-27 Nov | Final numbers, report, slides, rehearsal where everyone is questioned on the other two roles. |

**Hand-written parts (for the oral exam):** Levenshtein alignment with backtrace, error classification, confidence-threshold policy, CER/WER.
**Definition of done:** one command produces the full table from `ocr_out.jsonl` and `correct()`; the table includes the "fixed / broken / unchanged-right / unchanged-wrong" counts; the demo runs end to end.

---

## Who waits for whom

| Needs | From | By when | What if late |
|---|---|---|---|
| `splits.json` | Hiển | 8 Nov | A and B can still start on their own data |
| First real `ocr_out.jsonl` | Dương | 13 Nov | Hiển keeps using the stub; the hypothesis gate moves later |
| `correct()` with the contract signature | Khánh | 20 Nov for the real model; the identity stub is available from day one | Hiển integrates the baseline version |
| Real OCR error patterns | Hiển | 15 Nov | Khánh keeps the first noise generator |

## Gates (decisions that can change the plan)

1. **Gate 0, before 8 Nov:** ask the teacher whether PyTorch autograd is allowed. If it is not, role A (CRNN) cannot be built and the topic must change. *Who asks: to be agreed in the group (proposed: Hiển).*
2. **Hypothesis gate, 13-15 Nov:** if diacritic errors are only a small share of OCR errors, the pipeline loses its point and the group decides whether to change direction.
3. **OCR gate, end of week 1:** if CER is too high, choose a fallback (see role A).

## Open decisions

- Add a small real test set (about 90 handwritten lines, 30 per person)? Not decided yet.
- Who creates the GitHub repository and the Drive folder (proposed: Hiển, then share with the other two).
- GitHub usernames of Dương and Khánh (needed to add them as collaborators).

## Working agreement

- One branch per role: `a/ocr`, `b/corrector`, `c/eval`. Merge into `main` by pull request.
- Changes to `contracts/` or `src/common/` need all three people to agree.
- Never commit data, weights or tokens.
- Code, comments and identifiers in English.
