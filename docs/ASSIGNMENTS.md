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

## Team decisions (8 Nov)

1. **PyTorch autograd is allowed** for training the CRNN and the transformer. The project is self-directed and the team decided this itself. The teacher's exact answer is not recorded here. The hand-written parts listed below stay hand-written.
2. **The shared data is prepared once by Hiển** in Colab (`notebooks/00_prepare_data.ipynb`). Nobody else downloads the dataset. Results go to the shared Drive folder.
3. **Colab only reads from GitHub** (the token is read-only). Code is pushed from laptops. Notebooks are downloaded from Colab and committed from the laptop.
4. **Checkpoints go to Drive after every epoch**, because Colab sessions get cut.
5. The text corpus for role B is **Vietnamese Wikipedia sentences** (built by `scripts/prepare_corpus.py`). Khánh can add a second corpus if the coverage number in `corrector_00_start.ipynb` is low.

---

## Role A: OCR (Dương)

**Goal:** turn an image of one handwritten line into text.

**Input:** `data/uit_hwdb/` zip, `outputs/dataset_index.jsonl` and `outputs/splits.json` on Drive (made by Hiển).
**Output:** `outputs/ocr_out.jsonl` on Drive, following `contracts/ocr_out.schema.json`; CER numbers on the validation split.

| Dates | Work |
|---|---|
| 9 Nov | Run `ocr_00_start.ipynb`: check that images match their text, look at image sizes, check the GPU. Report image count and sizes to the group. |
| 9-13 Nov | Preprocessing (fixed height, grayscale, normalization). CRNN baseline (CNN, BiLSTM, CTC loss). First make the model overfit 50 training lines. **First `ocr_out.jsonl` by 13 Nov** (at least the validation split). |
| 16-22 Nov | Improve the model. Implement CTC greedy decoding and beam search by hand (numpy). Re-export `ocr_out.jsonl`. |
| 23-27 Nov | Error analysis: read at least 50 wrong lines and write down what goes wrong. Write the part A section of the report. |

**Hand-written parts (for the oral exam):** image preprocessing, CTC greedy decoding, CTC beam search.
**Definition of done:** `python scripts/validate_ocr_out.py outputs/ocr_out.jsonl` prints `OK`; CER on validation is reported; code is merged into `main`; you can explain how CTC decoding turns frame probabilities into text.
**If blocked:** if CER is still very high at the end of week 1, tell the group immediately. Fallbacks to discuss: stitch word images into lines; use an existing OCR for the predictions (this needs the teacher's approval because the OCR part would no longer be self-made).

---

## Role B: Diacritic corrector (Khánh)

**Goal:** given a Vietnamese sentence with wrong or missing diacritics, return the fixed sentence and one confidence number per syllable.

**Input:** `data/corpus/viwiki_sentences.txt` on Drive. You create the training pairs yourself by corrupting clean text: remove diacritics, swap tone marks, swap vowel shape marks, each with some probability.
**Output:** the function `correct(text)` in `src/corrector/` following the contract (`{"text": str, "conf": [float]}`, one confidence per space-separated syllable); trained weights in Drive `checkpoints/corrector/`.

| Dates | Work |
|---|---|
| 9 Nov | Run `corrector_00_start.ipynb`: look at the corpus, the most ambiguous syllables, and how well the corpus covers the OCR training labels. Report the numbers to the group. |
| 9-15 Nov | Noise generator. Frequency-table baseline (most common accented form of each syllable). Bigram + Viterbi by hand. Measure syllable accuracy, and separately on ambiguous syllables. **A first `correct()` (frequency table) by 15 Nov** so Hiển can integrate early. *Proposed: confirm in the group.* |
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

**Already written (tested only on made-up data, never on the real dataset yet):** `src/eval/align.py`, `src/eval/classify_errors.py`, `src/stubs/stub_ocr.py`, `src/common/index_io.py`, `scripts/inspect_dataset.py`, `scripts/build_index.py`, `scripts/make_splits.py`, `scripts/prepare_corpus.py`, the three notebooks `00_prepare_data`, `ocr_00_start`, `corrector_00_start`, and 60 unit tests.

| Dates | Work |
|---|---|
| 8-9 Nov | Run `00_prepare_data.ipynb`: dataset, index, splits and corpus on Drive. Tell Dương and Khánh. Merge the tools and the notebooks into `main`. |
| By 12 Nov | Set the hypothesis-gate threshold **before looking at any number**. |
| 13-15 Nov | **Hypothesis gate:** run the classifier on A's first real `ocr_out.jsonl`. Report the share of OCR errors that are diacritic errors **and the upper bound** on the CER gain (CER multiplied by that share). |
| By 15 Nov | A script that prints the most frequent OCR confusion pairs, for Khánh's noise generator. |
| 16-22 Nov | `evaluate.py` with the comparison table (no correction, frequency table, Viterbi, transformer, transformer with confidence threshold), the "fixed / broken / unchanged-right / unchanged-wrong" counts, threshold sweep, demo. |
| 23-27 Nov | Final numbers, report, slides, rehearsal where everyone is questioned on the other two roles. |

**Hand-written parts (for the oral exam):** Levenshtein alignment with backtrace, error classification, confidence-threshold policy, CER/WER.
**Provenance:** the first versions of the evaluation and data tools were written with an AI assistant. Hiển must be able to explain and re-derive them. If the teacher requires them to be written without help, rewrite them using the tests as the specification.
**Definition of done:** one command produces the full table from `ocr_out.jsonl` and `correct()`; the demo runs end to end.

---

## Who waits for whom

| Needs | From | By when | What if late |
|---|---|---|---|
| `dataset_index.jsonl`, `splits.json`, corpus on Drive | Hiển | 9 Nov | Dương and Khánh cannot start the real work, so this is the first priority |
| First real `ocr_out.jsonl` | Dương | 13 Nov | Hiển keeps using the stub; the hypothesis gate moves later |
| First `correct()` (frequency table) | Khánh | 15 Nov (proposed) | Hiển keeps the identity stub |
| Real `correct()` model | Khánh | 20 Nov | Hiển integrates the baseline version |
| Real OCR error patterns | Hiển | 15 Nov | Khánh keeps the first noise generator |

## Gates (decisions that can change the plan)

1. **Hypothesis gate, 13-15 Nov:** if diacritic errors are only a small share of OCR errors, or the upper bound on the gain is tiny, the pipeline loses its point and the group decides whether to change direction.
2. **OCR gate, end of week 1:** if CER is too high, choose a fallback (see role A).
3. **Data gate, 9 Nov:** if `build_index.py` cannot read the real dataset layout, the script is adjusted before anything else starts.

## Open decisions

- Add a small real test set (about 90 handwritten lines, 30 per person)? Not decided yet. If yes, it should be written and photographed by about 15 Nov so Dương can run the OCR on it.

## Working agreement

- One branch per role: `a/ocr`, `b/corrector`, `c/eval`. Merge into `main` by pull request.
- Changes to `contracts/` or `src/common/` need all three people to agree.
- Never commit data, weights or tokens.
- Code, comments and identifiers in English.
