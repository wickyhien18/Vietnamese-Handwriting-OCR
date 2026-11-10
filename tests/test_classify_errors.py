import unittest

from src.eval.align import DEL, INS, MATCH, SUB
from src.eval.classify_errors import (
    BASE, SHAPE, SPACING, TONE, TONE_AND_SHAPE,
    classify_op, classify_pair, diacritic_share, split_marks,
)

A_DOT = "\u1ea1"       # a with dot below (tone: heavy)
A_BREVE = "\u0103"     # a with breve (shape)
A_CIRC_DOT = "\u1ead"  # a with circumflex and dot below (shape + tone)
A_CIRC = "\u00e2"      # a with circumflex
D_STROKE = "\u0111"    # d with stroke


class ClassifyTest(unittest.TestCase):
    def test_match_has_no_label(self):
        self.assertIsNone(classify_op(MATCH, "a", "a"))

    def test_tone_only(self):
        self.assertEqual(classify_op(SUB, A_DOT, "a"), TONE)

    def test_shape_only(self):
        self.assertEqual(classify_op(SUB, A_BREVE, "a"), SHAPE)

    def test_d_stroke_is_shape(self):
        self.assertEqual(classify_op(SUB, D_STROKE, "d"), SHAPE)

    def test_tone_missing_on_shaped_letter(self):
        self.assertEqual(classify_op(SUB, A_CIRC_DOT, A_CIRC), TONE)

    def test_tone_and_shape(self):
        self.assertEqual(classify_op(SUB, A_CIRC_DOT, "a"), TONE_AND_SHAPE)

    def test_different_base_letter(self):
        self.assertEqual(classify_op(SUB, "b", "p"), BASE)

    def test_insert_and_delete(self):
        self.assertEqual(classify_op(INS, None, "x"), BASE)
        self.assertEqual(classify_op(DEL, "x", None), BASE)
        self.assertEqual(classify_op(INS, None, " "), SPACING)
        self.assertEqual(classify_op(DEL, " ", None), SPACING)

    def test_space_confused_with_letter(self):
        self.assertEqual(classify_op(SUB, " ", "x"), SPACING)

    def test_split_marks(self):
        base, marks = split_marks(A_DOT)
        self.assertEqual(base, "a")
        self.assertEqual(marks, {"\u0323"})

    def test_pair_worked_example(self):
        counts = classify_pair("b" + A_DOT + "n", "ban")
        self.assertEqual(dict(counts), {TONE: 1})
        self.assertEqual(diacritic_share(counts), 1.0)

    def test_pair_mixed_errors(self):
        # ground truth "b<dot-a>n d<stroke>" vs OCR "bin d": tone error is a base error here
        ref = "b" + A_DOT + "n " + D_STROKE
        hyp = "bxn d"
        counts = classify_pair(ref, hyp)
        self.assertEqual(counts[BASE], 1)
        self.assertEqual(counts[SHAPE], 1)
        self.assertAlmostEqual(diacritic_share(counts), 0.5)

    def test_unnormalized_input_is_handled(self):
        decomposed = "ba\u0323n"  # a followed by a combining dot below
        counts = classify_pair(decomposed, "ban")
        self.assertEqual(dict(counts), {TONE: 1})

    def test_share_of_no_errors(self):
        self.assertEqual(diacritic_share(classify_pair("abc", "abc")), 0.0)


if __name__ == "__main__":
    unittest.main()
