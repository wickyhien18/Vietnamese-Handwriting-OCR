import importlib.util
import unittest

from src.corrector.ngram import NgramModel
from src.corrector.reference_check import (
    TAGGER_DECLARATION,
    compare_ngram_with_nltk,
    compare_viterbi_brute_force,
    compare_viterbi_hmmlearn,
)
from src.corrector.viterbi import Emission

HAS_NLTK = importlib.util.find_spec("nltk") is not None
HAS_HMMLEARN = importlib.util.find_spec("hmmlearn") is not None

CORPUS = (["Tôi mua bàn ghế mới."] * 5 + ["Cửa hàng bán hàng rất đông."] * 3
          + ["Bạn tôi là bạn tốt."] * 8 + ["Chúng tôi bàn bạc với bạn."] * 2)


class ReferenceCheckTest(unittest.TestCase):
    def test_viterbi_equals_brute_force_on_short_sentences(self):
        model = NgramModel()
        model.train(CORPUS)
        result = compare_viterbi_brute_force(model, Emission(0.7), CORPUS)
        self.assertGreater(result["sentences_checked"], 0)
        self.assertEqual(result["same_log_prob"], result["sentences_checked"])
        self.assertLess(result["max_abs_log_prob_diff"], 1e-9)

    def test_declaration_lists_attention_as_hand_written(self):
        self.assertIn("nn.MultiheadAttention", TAGGER_DECLARATION["not_used"])
        self.assertTrue(any("attention" in part for part in TAGGER_DECLARATION["hand_written"]))

    @unittest.skipUnless(HAS_NLTK, "nltk not installed")
    def test_ngram_matches_nltk(self):
        result = compare_ngram_with_nltk(CORPUS)
        self.assertTrue(result["same"], result)

    @unittest.skipUnless(HAS_HMMLEARN, "hmmlearn not installed")
    def test_viterbi_matches_hmmlearn(self):
        result = compare_viterbi_hmmlearn(trials=10)
        self.assertEqual(result["same_path"], 10)
        self.assertLess(result["max_abs_log_prob_diff"], 1e-8)


if __name__ == "__main__":
    unittest.main()
