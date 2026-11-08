import itertools
import os
import random
import tempfile
import unittest

from src.corrector.correctors import FrequencyCorrector, IdentityCorrector, ViterbiCorrector
from src.corrector.evaluate import evaluate
from src.corrector.ngram import NgramModel, sentence_words
from src.corrector.noise import make_pairs
from src.corrector.viterbi import Emission, estimate_p_same, path_log_prob, viterbi

CORPUS = (
    ["Tôi mua bàn ghế mới."] * 5
    + ["Cửa hàng bán hàng rất đông."] * 3
    + ["Bạn tôi là bạn tốt."] * 8
    + ["Chúng tôi bàn bạc với bạn."] * 2
)


def small_model():
    model = NgramModel()
    model.train(CORPUS)
    return model


class NgramTest(unittest.TestCase):
    def test_sentence_words(self):
        self.assertEqual(sentence_words("Hà Nội, năm 2020."), ["hà", "nội", "năm"])

    def test_counts(self):
        model = small_model()
        self.assertEqual(model.forms["ban"]["bạn"], 8 * 2 + 2)
        self.assertEqual(model.bigram[("bàn", "ghế")], 5)

    def test_probabilities_sum_to_one(self):
        model = small_model()
        words = [w for w in model.unigram if w != "<s>"] + ["</s>"]
        total = sum(model.p_bigram("tôi", w) for w in words)
        self.assertLess(total, 1.0 + 1e-9)  # the rest of the mass goes to unseen words

    def test_save_load(self):
        model = small_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.pkl.gz")
            model.save(path)
            loaded = NgramModel.load(path)
        self.assertEqual(loaded.bigram, model.bigram)
        self.assertEqual(loaded.forms, model.forms)


class ViterbiTest(unittest.TestCase):
    def test_viterbi_equals_brute_force(self):
        model = small_model()
        emission = Emission(0.7)
        observed = ["ban", "toi", "ban", "hang", "ban", "ghe"]
        best = viterbi(observed, model, emission)
        cands = [model.candidates(o) for o in observed]
        brute = max(itertools.product(*cands),
                    key=lambda p: path_log_prob(p, observed, model, emission))
        self.assertAlmostEqual(path_log_prob(best, observed, model, emission),
                               path_log_prob(brute, observed, model, emission))

    def test_context_decides(self):
        model = small_model()
        corrector = ViterbiCorrector(model, Emission(0.7))
        self.assertEqual(corrector.correct("Tôi mua ban ghe")["text"], "Tôi mua bàn ghế")
        self.assertEqual(corrector.correct("cua hang ban hang")["text"], "cửa hàng bán hàng")

    def test_frequency_ignores_context(self):
        corrector = FrequencyCorrector(small_model())
        self.assertEqual(corrector.correct("mua ban ghe")["text"], "mua bạn ghế")

    def test_contract_shape(self):
        model = small_model()
        for corrector in [IdentityCorrector(), FrequencyCorrector(model),
                          ViterbiCorrector(model, Emission(0.7))]:
            out = corrector.correct('"BAN toi", 2020 , Stockholm.')
            self.assertEqual(set(out), {"text", "conf"})
            self.assertEqual(len(out["conf"]), len(out["text"].split()))
            self.assertTrue(all(0.0 <= c <= 1.0 for c in out["conf"]))
            self.assertTrue(out["text"].startswith('"B'))  # quotes and capitals kept

    def test_estimate_p_same(self):
        self.assertEqual(estimate_p_same([("a b c d", "a b x d")]), 0.75)


class EvaluateTest(unittest.TestCase):
    def test_viterbi_beats_identity_on_stripped_text(self):
        model = small_model()
        pairs = list(make_pairs(CORPUS, seed=0))
        identity = evaluate(IdentityCorrector(), pairs, model)
        vit = evaluate(ViterbiCorrector(model, Emission(0.7)), pairs, model)
        self.assertGreater(vit["all"], identity["all"])
        self.assertEqual(identity["broken"], 0.0)


if __name__ == "__main__":
    unittest.main()
