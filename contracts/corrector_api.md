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
3. **The number of syllables does not change:** `len(out["text"].split()) == len(text.split())`. The evaluation compares input and output syllable by syllable. *(New clarification, consistent with the one-label-per-syllable tagger. Needs Khánh's confirmation.)*
4. **Only diacritics change.** For every syllable, removing the diacritics from the input syllable and from the output syllable gives the same string, and letter case does not change. A different base letter is an OCR error that this corrector must leave alone. Punctuation attached to a syllable (`doc,`) stays attached. *(New clarification. Needs Khánh's confirmation.)*
5. `conf` has one number per output syllable, each a float between 0 and 1.
6. `correct("")` returns `{"text": "", "conf": []}`.
7. The same input always gives the same output (no randomness at inference).

## Check your own function

```python
from src.common.contract_checks import check_correct_output

problems = check_correct_output("nhieu ban doc", correct("nhieu ban doc"))
assert problems == [], problems
```

A fake version that returns the input unchanged lives in `src/stubs/identity_corrector.py`.
