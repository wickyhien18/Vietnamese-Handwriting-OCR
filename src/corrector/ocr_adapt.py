"""Adapt the corrector to the real OCR errors and measure it on real OCR output.

Input: rows of ocr_out.jsonl (fields gt_text, ocr_text, split), written by role A.
Only the val split is used here. The test split must never guide any choice.

1. ocr_error_stats  : align label and OCR syllables (Levenshtein, src/eval/align.py) and count,
                      for every aligned pair, what kind of difference it has.
2. noise_config_from_stats / p_same_from_stats : turn those counts into the probabilities of the
                      noise generator and into the emission probability of Viterbi.
3. evaluate_on_ocr  : run a corrector on the real OCR text and count fixed / broken syllables
                      and CER / WER before and after (same definitions as the evaluation notebook).
"""

from collections import Counter

from src.common.text_utils import normalize_nfc, strip_diacritics
from src.corrector.noise import NoiseConfig
from src.corrector.syllable import decompose, normalize_tone_placement, split_token
from src.eval.align import DEL, INS, MATCH, SUB, align, corpus_error_rate

# Kinds of difference between a label syllable and an OCR syllable.
SAME = "same"                      # identical
STRIPPED = "stripped"              # OCR lost all marks: "người" -> "nguoi"
TONE = "tone"                      # only the tone mark differs: "người" -> "ngưới"
SHAPE = "shape"                    # only a shape mark differs: "người" -> "nguời"
TONE_AND_SHAPE = "tone_and_shape"  # both differ
PLACEMENT = "tone_placement"       # same marks, tone on another letter: "hòa" -> "hoà"
BASE = "base_letter"               # a base letter differs: a diacritic corrector cannot fix it
DIACRITIC_KINDS = (STRIPPED, TONE, SHAPE, TONE_AND_SHAPE, PLACEMENT)


def ocr_lines(rows, split="val"):
    """(label, OCR text) per line, NFC, OCR whitespace collapsed (as in the evaluation notebook)."""
    lines = []
    for row in rows:
        if row["split"] == split:
            lines.append((normalize_nfc(row["gt_text"]), " ".join(normalize_nfc(row["ocr_text"]).split())))
    return lines


def aligned_syllables(gt_text, ocr_text):
    """Syllable pairs (label syllable, OCR syllable, position in the OCR line) that the alignment
    matches or substitutes. Missing and extra syllables have no partner and are skipped."""
    pairs = []
    position = 0
    for kind, ref, hyp in align(gt_text.split(), ocr_text.split()):
        if kind in (MATCH, SUB):
            pairs.append((ref, hyp, position))
            position += 1
        elif kind == INS:
            position += 1
    return pairs


def _tone(chars):
    """The tone mark of a decomposed syllable (None if it has none), wherever it is written."""
    for _, _, tone in chars:
        if tone is not None:
            return tone
    return None


def difference_kind(gt_word, ocr_word):
    """Kind of difference between two words without punctuation (see the constants above)."""
    if gt_word == ocr_word:
        return SAME
    if strip_diacritics(gt_word) != strip_diacritics(ocr_word):
        return BASE
    g, o = decompose(gt_word), decompose(ocr_word)
    if all(shape is None and tone is None for _, shape, tone in o):
        return STRIPPED
    tone_differs = _tone(g) != _tone(o)
    shape_differs = [shape for _, shape, _ in g] != [shape for _, shape, _ in o]
    if tone_differs and shape_differs:
        return TONE_AND_SHAPE
    if tone_differs:
        return TONE
    if shape_differs:
        return SHAPE
    return PLACEMENT


def ocr_error_stats(lines):
    """Count the kinds of difference over all aligned syllable pairs (lower-case words).

    Also counts label syllables whose tone is not in our standard position ("hoà" instead of
    "hòa"): the corrector always writes the standard position, so these would count as errors.
    """
    counts = Counter()
    for gt_text, ocr_text in lines:
        counts["label_syllables"] += len(gt_text.split())
        for gt_token, ocr_token, _ in aligned_syllables(gt_text, ocr_text):
            gt_word, ocr_word = split_token(gt_token)[1], split_token(ocr_token)[1]
            if not gt_word or not ocr_word:
                counts["no_letters"] += 1
                continue
            counts["pairs"] += 1
            counts[difference_kind(gt_word.lower(), ocr_word.lower())] += 1
            if normalize_tone_placement(gt_word) != gt_word:
                counts["label_other_tone_style"] += 1
    counts["same_base"] = counts["pairs"] - counts[BASE]
    return counts


def noise_config_from_stats(counts):
    """Noise probabilities that copy the real OCR: measured on pairs with the same base letters.

    OCR never strips a whole line, so p_strip_sentence = 0. A "tone_and_shape" error counts for
    both the tone and the shape probability, because the noise generator applies them independently.
    """
    n = counts["same_base"]
    return NoiseConfig(
        p_strip_sentence=0.0,
        p_strip_syllable=counts[STRIPPED] / n,
        p_change_tone=(counts[TONE] + counts[TONE_AND_SHAPE]) / n,
        p_change_shape=(counts[SHAPE] + counts[TONE_AND_SHAPE]) / n,
    )


def p_same_from_stats(counts):
    """Emission probability for Viterbi: how often the OCR syllable is exactly the label syllable,
    among pairs with the same base letters (the only ones a diacritic corrector can choose for)."""
    return counts[SAME] / counts["same_base"]


def evaluate_on_ocr(corrector, lines, tau=0.0, n_examples=15):
    """Correct every OCR line and compare with the label.

    A suggestion is accepted only when its confidence is at least tau (tau = 0 accepts all).
    Syllables are compared on the aligned pairs (option 2 of the evaluation notebook):
        kept_right   OCR right, output right       broken       OCR right, output wrong
        fixed        OCR wrong, output right       still_wrong  OCR wrong, output wrong
    "fixable" = OCR wrong but the same base letters, so a perfect diacritic corrector could fix it.
    """
    counts = Counter()
    examples = {"fixed": Counter(), "broken": Counter(), "missed": Counter()}
    before, after = [], []
    for gt_text, ocr_text in lines:
        result = corrector.correct(ocr_text)
        ocr_tokens = ocr_text.split()
        out_tokens = [new if c >= tau else old
                      for old, new, c in zip(ocr_tokens, result["text"].split(), result["conf"])]
        out_text = " ".join(out_tokens)
        before.append((gt_text, ocr_text))
        after.append((gt_text, out_text))
        for gt_token, ocr_token, position in aligned_syllables(gt_text, ocr_text):
            out_token = out_tokens[position]
            ocr_right, out_right = ocr_token == gt_token, out_token == gt_token
            fixable = not ocr_right and strip_diacritics(ocr_token) == strip_diacritics(gt_token)
            counts["fixable"] += fixable
            if ocr_right and out_right:
                counts["kept_right"] += 1
            elif ocr_right:
                counts["broken"] += 1
                examples["broken"][(gt_token, out_token)] += 1
            elif out_right:
                counts["fixed"] += 1
                examples["fixed"][(ocr_token, gt_token)] += 1
            else:
                counts["still_wrong"] += 1
                if fixable:
                    examples["missed"][(ocr_token, out_token, gt_token)] += 1

    right_before = counts["kept_right"] + counts["broken"]
    words = lambda pairs: [(tuple(g.split()), tuple(h.split())) for g, h in pairs]
    return {
        "tau": tau,
        "cer_before": corpus_error_rate(before),
        "cer_after": corpus_error_rate(after),
        "wer_before": corpus_error_rate(words(before)),
        "wer_after": corpus_error_rate(words(after)),
        "counts": dict(counts),
        "broken_rate": counts["broken"] / right_before if right_before else 0.0,
        "fixed_of_fixable": counts["fixed"] / counts["fixable"] if counts["fixable"] else 0.0,
        "net_fixed": counts["fixed"] - counts["broken"],
        "examples": {k: v.most_common(n_examples) for k, v in examples.items()},
    }


def print_ocr_table(rows):
    """rows: {method name: result of evaluate_on_ocr()}"""
    header = (f"{'method':<30}{'CER':>14}{'WER':>16}{'fixed':>7}{'broken':>8}"
              f"{'net':>6}{'broken rate':>13}{'fixed/fixable':>15}")
    print(header)
    print("-" * len(header))
    for name, r in rows.items():
        c = r["counts"]
        print(f"{name:<30}{r['cer_before']:>6.2%} ->{r['cer_after']:>6.2%}"
              f"{r['wer_before']:>7.2%} ->{r['wer_after']:>6.2%}"
              f"{c.get('fixed', 0):>7}{c.get('broken', 0):>8}{r['net_fixed']:>6}"
              f"{r['broken_rate']:>13.2%}{r['fixed_of_fixable']:>15.2%}")


def confidence_pairs_ocr(corrector, lines):
    """(confidence, 1 if right else 0) for every aligned syllable of the real OCR lines.
    The OCR syllable is matched with its label syllable by alignment (option 2 of the
    evaluation notebook), so the ECE is measured on exactly the syllables tau acts on."""
    conf, right = [], []
    for gt_text, ocr_text in lines:
        result = corrector.correct(ocr_text)
        out_tokens = result["text"].split()
        for gt_token, _, position in aligned_syllables(gt_text, ocr_text):
            conf.append(result["conf"][position])
            right.append(int(out_tokens[position] == gt_token))
    return conf, right
