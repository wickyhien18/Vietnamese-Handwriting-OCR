import json
import os
import tempfile
import unittest

from src.corrector import api
from src.corrector.ngram import NgramModel

CORPUS = ["Tôi mua bàn ghế mới."] * 5 + ["Bạn tôi là bạn tốt."] * 8


class ApiTest(unittest.TestCase):
    def setUp(self):
        api._corrector = None

    def test_correct_before_load_fails(self):
        with self.assertRaises(RuntimeError):
            api.correct("ban toi")

    def test_identity(self):
        api.load(method="identity")
        self.assertEqual(api.correct("ban toi"), {"text": "ban toi", "conf": [1.0, 1.0]})

    def test_viterbi_and_frequency_from_files(self):
        model = NgramModel()
        model.train(CORPUS)
        with tempfile.TemporaryDirectory() as tmp:
            model.save(os.path.join(tmp, "ngram.pkl.gz"))
            with open(os.path.join(tmp, "viterbi.json"), "w") as f:
                json.dump({"p_same": 0.7, "k": 8}, f)
            api.load(tmp, method="viterbi")
            out = api.correct("Toi mua ban ghe.")
            self.assertEqual(out["text"], "Tôi mua bàn ghế.")
            self.assertEqual(len(out["conf"]), 4)
            api.load(tmp, method="frequency")
            self.assertEqual(api.correct("ban toi")["text"], "bạn tôi")

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            api.load(method="magic")


if __name__ == "__main__":
    unittest.main()
