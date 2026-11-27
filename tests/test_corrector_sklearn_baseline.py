import importlib.util
import os
import tempfile
import unittest

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None

CORPUS = (["Tôi mua bàn ghế mới."] * 20 + ["Cửa hàng bán hàng rất đông."] * 20
          + ["Bạn tôi là bạn tốt."] * 20 + ["Chúng tôi bàn bạc với bạn."] * 10)


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn not installed")
class SklearnBaselineTest(unittest.TestCase):
    def setUp(self):
        from src.corrector.ngram import NgramModel
        from src.corrector.noise import NoiseConfig, make_pairs
        from src.corrector.sklearn_baseline import train_sklearn_baseline
        self.model = NgramModel()
        self.model.train(CORPUS)
        pairs = list(make_pairs(CORPUS * 5, seed=0, config=NoiseConfig(p_strip_sentence=1.0)))
        self.corrector, self.rows = train_sklearn_baseline(self.model, pairs)

    def test_learns_context(self):
        self.assertGreater(self.rows, 0)
        self.assertEqual(self.corrector.correct("toi mua ban ghe")["text"], "tôi mua bàn ghế")
        self.assertEqual(self.corrector.correct("cua hang ban hang")["text"], "cửa hàng bán hàng")

    def test_contract_and_edge_cases(self):
        from src.common.contract_checks import check_correct_output
        from tests.test_corrector_api import EDGE_CASES
        for text in EDGE_CASES:
            out = self.corrector.correct(text)
            self.assertEqual(check_correct_output(text, out), [], (text, out))

    def test_temperature_and_save_load(self):
        from src.corrector.calibration import fit_temperature
        from src.corrector.experiments import collect_level_logits
        from src.corrector.noise import make_pairs
        from src.corrector.sklearn_baseline import load, save
        logits, targets = collect_level_logits(self.corrector, list(make_pairs(CORPUS, seed=3)), 9)
        self.assertGreater(fit_temperature(logits, targets), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sk.pkl")
            self.corrector.temperature = 1.3
            save(self.corrector, path)
            loaded = load(path, self.model)
        self.assertEqual(loaded.temperature, 1.3)
        self.assertEqual(loaded.correct("ban toi la ban tot"), self.corrector.correct("ban toi la ban tot"))


if __name__ == "__main__":
    unittest.main()
