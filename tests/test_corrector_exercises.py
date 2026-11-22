"""The worked exercises of docs/corrector_proofs.md (Part C), checked with the project code."""

import itertools
import math
import unittest

import numpy as np

from src.corrector.calibration import calibration_report, fit_temperature
from src.corrector.ngram import END, START, NgramModel
from src.corrector.viterbi import path_log_prob, viterbi

START_P = {"bạn": 0.4, "bàn": 0.1}
TRANS = {("bạn", "ghế"): 0.05, ("bạn", "ghe"): 0.10, ("bàn", "ghế"): 0.60, ("bàn", "ghe"): 0.05,
         ("ghế", "mới"): 0.50, ("ghế", "mời"): 0.10, ("ghe", "mới"): 0.20, ("ghe", "mời"): 0.30}
END_P = {"mới": 0.5, "mời": 0.4}
CANDS = {"ban": ["bạn", "bàn"], "ghe": ["ghế", "ghe"], "moi": ["mới", "mời"]}


class ExerciseModel:
    def candidates(self, observed, k=8):
        return CANDS[observed]

    def log_p(self, prev, c):
        if prev == START:
            return math.log(START_P[c])
        if c == END:
            return math.log(END_P[prev])
        return math.log(TRANS[(prev, c)])


def emission(observed, clean):
    return math.log(0.7 if observed == clean else 0.3)


class ExercisesTest(unittest.TestCase):
    def test_c1_viterbi_equals_brute_force(self):
        model, observed = ExerciseModel(), ["ban", "ghe", "moi"]
        best = viterbi(observed, model, emission)
        self.assertEqual(best, ["bàn", "ghế", "mới"])
        self.assertAlmostEqual(math.exp(path_log_prob(best, observed, model, emission)), 0.000405)
        brute = max(itertools.product(*[CANDS[o] for o in observed]),
                    key=lambda p: path_log_prob(p, observed, model, emission))
        self.assertEqual(list(brute), best)

    def test_c2_smoothing_sums_to_one(self):
        model = NgramModel(interpolation=0.8)
        model.train(["ta đi", "ta về"])
        self.assertAlmostEqual(model.p_unigram("ta"), 3 / 9)
        self.assertAlmostEqual(model.p_unigram(END), 1 / 9)
        self.assertAlmostEqual(model.p_bigram("ta", "đi"), 0.8 * 0.5 + 0.2 * 2 / 9)
        vocabulary = ["ta", "đi", "về", END, "<unk>"]
        self.assertAlmostEqual(sum(model.p_bigram("ta", w) for w in vocabulary), 1.0)
        self.assertAlmostEqual(sum(model.p_unigram(w) for w in vocabulary), 1.0)

    def test_c3_temperature_and_ece(self):
        logits, targets = np.tile([3.0, 0.0], (4, 1)), np.array([0, 0, 0, 1])
        t = fit_temperature(logits, targets)
        self.assertAlmostEqual(t, 3 / math.log(3), places=4)
        report = calibration_report(logits, targets, t)
        self.assertAlmostEqual(report["ece_raw"], 0.2026, places=4)
        self.assertAlmostEqual(report["ece_scaled"], 0.0, places=6)
        self.assertEqual(report["accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
