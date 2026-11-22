import unittest

from src.corrector.correctors import IdentityCorrector, ViterbiCorrector
from src.corrector.ngram import NgramModel
from src.corrector.ocr_adapt import (
    BASE,
    PLACEMENT,
    SAME,
    SHAPE,
    STRIPPED,
    TONE,
    TONE_AND_SHAPE,
    aligned_syllables,
    difference_kind,
    evaluate_on_ocr,
    noise_config_from_stats,
    ocr_error_stats,
    ocr_lines,
    p_same_from_stats,
)
from src.corrector.viterbi import Emission


class DifferenceKindTest(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(difference_kind("người", "người"), SAME)
        self.assertEqual(difference_kind("người", "nguoi"), STRIPPED)
        self.assertEqual(difference_kind("người", "ngưới"), TONE)
        self.assertEqual(difference_kind("người", "nguời"), SHAPE)
        self.assertEqual(difference_kind("người", "nguội"), TONE_AND_SHAPE)
        self.assertEqual(difference_kind("hòa", "hoà"), PLACEMENT)
        self.assertEqual(difference_kind("người", "ngưài"), BASE)


class StatsTest(unittest.TestCase):
    LINES = [
        ("nhiều bạn đọc đã bày tỏ", "nhiều ban đọc đả bày tó"),  # stripped, tone, tone
        ("sự thật được nói lên", "sự thật đuợc nói lên ."),     # shape + an extra token
        ("hòa bình", "hòa binh"),                               # stripped
        ("khoa học", "khoa hục"),                               # base letter
    ]

    def test_aligned_syllables_skip_extra_tokens(self):
        pairs = aligned_syllables("sự thật được nói lên", "sự thật đuợc nói lên .")
        self.assertEqual([p[0] for p in pairs], ["sự", "thật", "được", "nói", "lên"])

    def test_stats_and_config(self):
        counts = ocr_error_stats(self.LINES)
        self.assertEqual(counts["pairs"], 15)
        self.assertEqual(counts[BASE], 1)
        self.assertEqual(counts[STRIPPED], 2)
        self.assertEqual(counts[TONE], 2)
        self.assertEqual(counts[SHAPE], 1)
        self.assertEqual(counts["same_base"], 14)
        config = noise_config_from_stats(counts)
        self.assertEqual(config.p_strip_sentence, 0.0)
        self.assertAlmostEqual(config.p_strip_syllable, 2 / 14)
        self.assertAlmostEqual(config.p_change_tone, 2 / 14)
        self.assertAlmostEqual(p_same_from_stats(counts), 9 / 14)

    def test_ocr_lines_only_val_and_whitespace(self):
        rows = [{"split": "val", "gt_text": "a b", "ocr_text": " a   b "},
                {"split": "test", "gt_text": "c", "ocr_text": "c"}]
        self.assertEqual(ocr_lines(rows), [("a b", "a b")])


class EvaluateOnOcrTest(unittest.TestCase):
    def test_identity_changes_nothing(self):
        lines = [("nhiều bạn đọc", "nhiều ban đọc")]
        r = evaluate_on_ocr(IdentityCorrector(), lines)
        self.assertEqual(r["cer_before"], r["cer_after"])
        self.assertEqual(r["counts"]["broken"] if "broken" in r["counts"] else 0, 0)
        self.assertEqual(r["counts"]["fixable"], 1)

    def test_viterbi_fixes_and_tau_blocks(self):
        model = NgramModel()
        model.train(["Nhiều bạn đọc đã bày tỏ."] * 10 + ["Bàn ghế mới."] * 3)
        corrector = ViterbiCorrector(model, Emission(0.7))
        lines = [("nhiều bạn đọc đã bày tỏ", "nhiều ban đọc đả bày tỏ")]
        r = evaluate_on_ocr(corrector, lines)
        self.assertEqual(r["counts"]["fixed"], 2)
        self.assertLess(r["cer_after"], r["cer_before"])
        blocked = evaluate_on_ocr(corrector, lines, tau=1.01)  # nothing is confident enough
        self.assertEqual(blocked["cer_after"], blocked["cer_before"])


if __name__ == "__main__":
    unittest.main()
