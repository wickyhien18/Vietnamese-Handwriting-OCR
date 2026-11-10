import unittest

from src.eval.align import (
    DEL, INS, MATCH, SUB, align, corpus_error_rate, count_ops,
    edit_distance, edit_distance_table, error_rate,
)

A_DOT = "\u1ea1"  # a with dot below


class AlignTest(unittest.TestCase):
    def test_table_matches_worked_example(self):
        # ref is the ground truth, hyp is the OCR output; rows follow hyp, columns follow ref.
        table = edit_distance_table("b" + A_DOT + "n", "ban")
        self.assertEqual(table, [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 1, 2], [3, 2, 2, 1]])

    def test_backtrace_one_substitution(self):
        ops = align("b" + A_DOT + "n", "ban")
        self.assertEqual(ops, [(MATCH, "b", "b"), (SUB, A_DOT, "a"), (MATCH, "n", "n")])

    def test_identical(self):
        self.assertEqual(edit_distance("abc", "abc"), 0)
        self.assertTrue(all(op[0] == MATCH for op in align("abc", "abc")))

    def test_extra_hyp_character(self):
        self.assertEqual(align("ab", "axb"), [(MATCH, "a", "a"), (INS, None, "x"), (MATCH, "b", "b")])

    def test_missing_hyp_character(self):
        self.assertEqual(align("axb", "ab"), [(MATCH, "a", "a"), (DEL, "x", None), (MATCH, "b", "b")])

    def test_empty_strings(self):
        self.assertEqual(align("", ""), [])
        self.assertEqual(align("ab", ""), [(DEL, "a", None), (DEL, "b", None)])
        self.assertEqual(align("", "ab"), [(INS, None, "a"), (INS, None, "b")])

    def test_distance_is_symmetric(self):
        self.assertEqual(edit_distance("kitten", "sitting"), edit_distance("sitting", "kitten"))
        self.assertEqual(edit_distance("kitten", "sitting"), 3)

    def test_ops_account_for_distance(self):
        ref, hyp = "nhieu ban doc", "nhi ban dooc"
        counts = count_ops(align(ref, hyp))
        self.assertEqual(counts[SUB] + counts[INS] + counts[DEL], edit_distance(ref, hyp))

    def test_error_rates(self):
        self.assertAlmostEqual(error_rate("abcd", "abxd"), 0.25)
        self.assertEqual(error_rate("", ""), 0.0)
        self.assertEqual(error_rate("", "a"), float("inf"))
        self.assertAlmostEqual(corpus_error_rate([("abcd", "abxd"), ("ab", "ab")]), 1 / 6)

    def test_works_on_word_lists(self):
        ref = ["a", "b", "c"]
        hyp = ["a", "x", "c"]
        self.assertEqual(edit_distance(ref, hyp), 1)
        self.assertAlmostEqual(error_rate(ref, hyp), 1 / 3)


if __name__ == "__main__":
    unittest.main()
