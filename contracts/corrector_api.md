# Contract: `correct(text)` (role B produces, role C and the demo consume)

```python
def correct(text: str) -> dict:
    ...
```

## Concrete example

Input (the OCR output, NFC, syllables separated by single spaces):

```
"nhieu ban doc da bay to"
```

Output:

```python
{"text": "nhiều bạn đọc đã bày tỏ", "conf": [0.97, 0.91, 0.99, 0.88, 0.95, 0.90]}
```

(The numbers are illustrative.) Six input syllables, six output syllables, six confidence values.

## Rules

1. The output is a dict with **exactly** the keys `text` and `conf`.
2. `text` is NFC.
3. **The number of syllables does not change:** `len(out["text"].split()) == len(text.split())`. The evaluation compares input and output syllable by syllable. *(Clarification of 9 Nov, confirmed by Khánh on 22 Nov.)*
4. **Only diacritics change.** For every syllable, removing the diacritics from the input syllable and from the output syllable gives the same string, and letter case does not change. A different base letter is an OCR error that this corrector must leave alone. Punctuation attached to a syllable (`doc,`) stays attached. *(Clarification of 9 Nov, confirmed by Khánh on 22 Nov.)*
5. `conf` has one number per output syllable, each a float between 0 and 1.
6. `correct("")` returns `{"text": "", "conf": []}`.
7. The same input always gives the same output (no randomness at inference).

## One entry point, three levels (since 22 Nov)

```python
from src.corrector.api import correct          # the default level, loaded on the first call
correct("ban")                                  # {"text": "bạn", "conf": [...]}

from src.corrector.api import get_corrector
frequency = get_corrector("frequency")          # each one is a function that follows this contract
bigram = get_corrector("bigram")
tagger = get_corrector("tagger")                # needs PyTorch
```

| Level | What it does | Files in the model folder |
|---|---|---|
| `frequency` | most frequent accented form of each syllable (baseline) | `ngram.pkl.gz`, `viterbi.json` |
| `bigram` | bigram hidden Markov model decoded with Viterbi | `ngram.pkl.gz`, `viterbi.json` |
| `tagger` | small transformer, one label per syllable | `tagger.pt` |

**Model folder.** The argument `ckpt_dir`, else the environment variable `CORRECTOR_CKPT_DIR`, else
the shared Drive folder `/content/drive/MyDrive/Vietnamese-Handwriting-OCR/checkpoints/corrector`.
No personal path is written in the code.

**Default level.** `corrector_06_final.ipynb` runs the three levels on the 636 OCR val lines of
UIT-HWDB and writes the one with the lowest CER (every suggestion accepted, tau = 0) into
`default_level.json` in the model folder, with the numbers it was chosen on. `correct()` uses that
level. Until the file exists the default is `bigram`, because it was the best level on the corpus
val in the first runs (syllable accuracy 95.8 % against 95.3 % for the tagger, mixed noise) and it
needs no PyTorch. The choice is made on val only, never on test. On 22 Nov the choice was `tagger` (CER 13.72 % against 15.00 % for `bigram` and 17.85 % for `frequency`); note that all three are above the 12.27 % of no correction when every suggestion is accepted, so tau matters (`docs/corrector_model_card.md`).

**Why val and not the corpus.** The corpus accuracy measures synthetic noise; the project question
is about real OCR errors, so the default is chosen on the OCR val lines. Role C still chooses tau on
the same val lines, and "no correction" can win that sweep.

**Confidence.** Every level gives a calibrated confidence (temperature scaling, `calibration.py`,
fitted on the corpus val): softmax over the candidates of one syllable of score / T. Scores:
log-frequency (`frequency`), the bigram score with both neighbours fixed (`bigram`), the logits
(`tagger`). ECE uses 10 equal-width bins weighted by the share of syllables in each bin.

## Check your own function

```python
from src.common.contract_checks import check_correct_output

problems = check_correct_output("nhieu ban doc", correct("nhieu ban doc"))
assert problems == [], problems
```

A fake version that returns the input unchanged lives in `src/stubs/identity_corrector.py`.
