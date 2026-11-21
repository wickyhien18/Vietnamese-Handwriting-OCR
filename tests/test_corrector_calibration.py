import unittest

import numpy as np

from src.corrector.calibration import (
    calibration_report,
    expected_calibration_error,
    fit_temperature,
    nll,
    softmax,
)
from src.corrector.ngram import NgramModel
from src.corrector.tagger_data import IGNORE, UNK, TaggerVocab


class CalibrationTest(unittest.TestCase):
    def test_softmax_with_missing_candidates(self):
        p = softmax(np.array([[1.0, 1.0, -np.inf]]))
        np.testing.assert_allclose(p, [[0.5, 0.5, 0.0]])

    def test_overconfident_model_gets_t_above_one(self):
        # Logits say "class 0" very strongly, but class 0 is right only 70 % of the time.
        rng = np.random.default_rng(0)
        n = 4000
        logits = np.tile([6.0, 0.0, 0.0], (n, 1))
        targets = np.where(rng.random(n) < 0.7, 0, rng.integers(1, 3, n))
        t = fit_temperature(logits, targets)
        self.assertGreater(t, 1.0)
        # At the best T the confidence of class 0 should be close to 0.7.
        self.assertAlmostEqual(softmax(logits / t)[0, 0], 0.7, delta=0.02)
        self.assertLess(nll(logits, targets, t), nll(logits, targets, 1.0))
        report = calibration_report(logits, targets, t)
        self.assertLess(report["ece_scaled"], report["ece_raw"])

    def test_ece(self):
        self.assertAlmostEqual(expected_calibration_error([0.8] * 10, [1] * 8 + [0] * 2), 0.0)
        self.assertAlmostEqual(expected_calibration_error([1.0] * 10, [1] * 5 + [0] * 5), 0.5)


class TaggerVocabTest(unittest.TestCase):
    def setUp(self):
        model = NgramModel()
        model.train(["Bạn tôi là bạn tốt."] * 5 + ["Tôi mua bàn ghế."] * 3 + ["Stockholm đẹp."])
        self.vocab = TaggerVocab.from_ngram(model, min_count=1)

    def test_encode(self):
        form_ids, key_ids, cand_ids, targets = self.vocab.encode(
            ["ban", "toi", "xyz"], ["bàn", "tôi", "xyz"])
        cands0 = [self.vocab.forms[i] for i in cand_ids[0]]
        self.assertEqual(cands0[0], "bạn")             # most frequent first
        self.assertEqual(cands0[targets[0]], "bàn")     # target points at the clean form
        self.assertEqual(form_ids[0], UNK)              # "ban" itself never seen in the corpus
        self.assertEqual(cand_ids[2], [UNK])            # unknown word: no candidates
        self.assertEqual(targets[2], IGNORE)

    def test_round_trip(self):
        copy = TaggerVocab.from_dict(self.vocab.to_dict())
        self.assertEqual(copy.encode(["ban", "toi"]), self.vocab.encode(["ban", "toi"]))


if __name__ == "__main__":
    unittest.main()
