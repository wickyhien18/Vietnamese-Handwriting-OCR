import json
import os
import subprocess
import sys
import tempfile
import unittest

from src.corrector.calibration import fit_temperature
from src.corrector.correctors import FrequencyCorrector, ViterbiCorrector
from src.corrector.experiments import (
    collect_level_logits,
    confidence_pairs,
    contiguous_folds,
    contract_violations,
    mean_std,
    ngram_folds,
    spread_sample,
)
from src.corrector.ngram import NgramModel
from src.corrector.noise import make_pairs
from src.corrector.ocr_adapt import confidence_pairs_ocr
from src.corrector.viterbi import Emission

SENTENCES = (["Tôi mua bàn ghế mới cho nhà."] * 30 + ["Cửa hàng bán hàng rất đông khách."] * 30
             + ["Bạn tôi là người bạn tốt."] * 30 + ["Chúng tôi bàn bạc với bạn về việc học."] * 30)


class HelpersTest(unittest.TestCase):
    def test_contiguous_folds_cover_everything(self):
        folds = contiguous_folds(103, 5)
        self.assertEqual(folds[0][0], 0)
        self.assertEqual(folds[-1][1], 103)
        self.assertTrue(all(a[1] == b[0] for a, b in zip(folds, folds[1:])))
        self.assertLessEqual(max(e - s for s, e in folds) - min(e - s for s, e in folds), 1)

    def test_mean_std_is_sample_std(self):
        mean, std = mean_std([1, 2, 3, 4])
        self.assertAlmostEqual(mean, 2.5)
        self.assertAlmostEqual(std, (5 / 3) ** 0.5)  # population std would be 1.118
        self.assertEqual(mean_std([7]), (7.0, 0.0))

    def test_spread_sample(self):
        self.assertEqual(spread_sample(list(range(10)), 5), [0, 2, 4, 6, 8])


class ExperimentsTest(unittest.TestCase):
    def setUp(self):
        self.model = NgramModel()
        self.model.train(SENTENCES)
        self.pairs = list(make_pairs(SENTENCES[::7], seed=3))

    def test_ngram_folds_report_mean_std(self):
        result = ngram_folds(SENTENCES, k=4, n_eval=20, log=lambda *a: None)
        for level in ("frequency", "bigram"):
            self.assertEqual(result[level]["n_runs"], 4)
            self.assertEqual(result[level]["seeds"], [0, 1, 2, 3])
            self.assertIn("mean", result[level]["all"])
            self.assertIn("ambiguous", result[level])
        self.assertGreaterEqual(result["bigram"]["all"]["mean"], result["frequency"]["all"]["mean"])

    def test_temperature_for_frequency_and_bigram(self):
        for corrector in (FrequencyCorrector(self.model), ViterbiCorrector(self.model, Emission(0.7))):
            logits, targets = collect_level_logits(corrector, self.pairs, k_max=9)
            self.assertEqual(logits.shape[1], 9)
            self.assertEqual(len(logits), len(targets))
            self.assertGreater(len(targets), 0)
            self.assertGreater(fit_temperature(logits, targets), 0)

    def test_confidence_pairs_and_contract(self):
        corrector = ViterbiCorrector(self.model, Emission(0.7))
        conf, right = confidence_pairs(corrector, self.pairs)
        self.assertEqual(len(conf), len(right))
        self.assertTrue(all(0 <= c <= 1 for c in conf))
        lines = [("tôi mua bàn ghế", "toi mua ban ghe"), ("bạn tôi", "ban toi xx")]
        conf2, right2 = confidence_pairs_ocr(corrector, lines)
        self.assertEqual(len(conf2), 6)
        result = contract_violations(corrector.correct, ["toi mua ban ghe", "", "BAN, toi"])
        self.assertEqual(result["violations"], 0)

    def test_command_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "train.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(SENTENCES) + "\n")
            out = os.path.join(tmp, "eval.json")
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            subprocess.run([sys.executable, "-m", "src.corrector.experiments", "--corpus-dir", tmp,
                            "--out", out, "--folds", "3", "--n-eval", "10"],
                           check=True, cwd=root, capture_output=True)
            with open(out, encoding="utf-8") as f:
                result = json.load(f)
        self.assertEqual(result["levels"]["bigram"]["n_runs"], 3)


if __name__ == "__main__":
    unittest.main()
