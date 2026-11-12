"""Classify character-level OCR errors (hand-written rules).

Categories:
  tone_mark       only the tone differs (grave, acute, tilde, hook above, dot below)
  shape_mark      only the letter shape differs (circumflex, breve, horn, d with stroke)
  tone_and_shape  both differ
  base_letter     a different letter, or a letter inserted or deleted
  spacing         a space inserted, deleted or confused with another character

Inputs must be NFC-normalized. classify_pair normalizes for you.
"""

import unicodedata
from collections import Counter

from src.common.text_utils import normalize_nfc, strip_diacritics
from src.eval.align import DEL, INS, MATCH, align

TONE_MARKS = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}
STROKE = "stroke"
SHAPE_MARKS = {"\u0302", "\u0306", "\u031b", STROKE}

TONE = "tone_mark"
SHAPE = "shape_mark"
TONE_AND_SHAPE = "tone_and_shape"
BASE = "base_letter"
SPACING = "spacing"

DIACRITIC_LABELS = (TONE, SHAPE, TONE_AND_SHAPE)
ALL_LABELS = (TONE, SHAPE, TONE_AND_SHAPE, BASE, SPACING)


def split_marks(char):
    """Return (base letter, set of combining marks). d with stroke gets a 'stroke' mark."""
    decomposed = unicodedata.normalize("NFD", char)
    base = decomposed[0]
    marks = set(decomposed[1:])
    if char in ("\u0111", "\u0110"):
        marks.add(STROKE)
    return base, marks


def classify_op(kind, ref_char, hyp_char):
    """Return the error label of one alignment operation, or None for a match."""
    if kind == MATCH:
        return None
    if kind in (INS, DEL):
        char = hyp_char if kind == INS else ref_char
        return SPACING if char.isspace() else BASE
    if ref_char.isspace() or hyp_char.isspace():
        return SPACING
    if strip_diacritics(ref_char) != strip_diacritics(hyp_char):
        return BASE
    _, ref_marks = split_marks(ref_char)
    _, hyp_marks = split_marks(hyp_char)
    tone_differs = (ref_marks & TONE_MARKS) != (hyp_marks & TONE_MARKS)
    shape_differs = (ref_marks & SHAPE_MARKS) != (hyp_marks & SHAPE_MARKS)
    if tone_differs and shape_differs:
        return TONE_AND_SHAPE
    if tone_differs:
        return TONE
    if shape_differs:
        return SHAPE
    return BASE  # should not happen for NFC input


def classify_pair(ref, hyp):
    """Count error labels for one (ground truth, OCR output) pair."""
    ref = normalize_nfc(ref)
    hyp = normalize_nfc(hyp)
    counts = Counter()
    for kind, ref_char, hyp_char in align(ref, hyp):
        label = classify_op(kind, ref_char, hyp_char)
        if label is not None:
            counts[label] += 1
    return counts


def diacritic_share(counts):
    """Share of all errors that are diacritic errors (the hypothesis gate number)."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return sum(counts[label] for label in DIACRITIC_LABELS) / total
