import json
import os
import tempfile
import unittest
from unittest import mock

from src.common.contract_checks import check_correct_output
from src.corrector import api
from src.corrector.ngram import NgramModel

CORPUS = ["Tôi mua bàn ghế mới."] * 5 + ["Bạn tôi là bạn tốt."] * 8 + ["Nhiều bạn đọc đã bày tỏ."] * 4

# Edge cases of contract rule checks (contracts/corrector_api.md), item 4 of the task list.
EDGE_CASES = [
    "",                                  # empty string
    "   ",                               # only spaces
    "ban",                               # one syllable
    "nhieu ban doc da bay to",           # no diacritics at all
    "năm 2004 , ngày 5 - 6",             # digits and loose punctuation
    '" ban " : ( toi ) ; ban !',         # quotes and brackets as their own tokens
    "doc,ban.toi",                       # punctuation inside a token
    "BAN TOI LA BAN TOT",                # all capitals
    "Ban Toi iPhone McDonald",           # mixed case
    "xyzq Stockholm Försvarsmakten",     # syllables the corpus has never seen
    "Đ đ ĐƯỜNG đuờng",                   # đ and shape marks
    "  ban   toi  ",                     # repeated spaces
]


def write_models(folder, default=None):
    model = NgramModel()
    model.train(CORPUS)
    model.save(os.path.join(folder, "ngram.pkl.gz"))
    with open(os.path.join(folder, "viterbi.json"), "w") as f:
        json.dump({"p_same": 0.7, "k": 8, "temperature": 1.3, "temperature_frequency": 0.9}, f)
    if default:
        with open(os.path.join(folder, "default_level.json"), "w") as f:
            json.dump({"level": default}, f)


class ApiTest(unittest.TestCase):
    def setUp(self):
        api._corrector = None
        api._cache.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = self.tmp.name
        write_models(self.folder)

    def tearDown(self):
        self.tmp.cleanup()
        api._corrector = None
        api._cache.clear()

    def test_empty_input_needs_no_model(self):
        with mock.patch.dict(os.environ, {api.CKPT_ENV: "/does/not/exist"}):
            self.assertEqual(api.correct(""), {"text": "", "conf": []})

    def test_missing_model_folder_fails_clearly(self):
        with mock.patch.dict(os.environ, {api.CKPT_ENV: "/does/not/exist"}):
            with self.assertRaises(FileNotFoundError):
                api.correct("ban toi")

    def test_folder_from_environment_variable(self):
        with mock.patch.dict(os.environ, {api.CKPT_ENV: self.folder}):
            self.assertEqual(api.checkpoint_dir(), self.folder)
            self.assertEqual(api.correct("Toi mua ban ghe.")["text"], "Tôi mua bàn ghế.")

    def test_default_level_file(self):
        self.assertEqual(api.default_level(self.folder), api.FALLBACK_LEVEL)
        write_models(self.folder, default="frequency")
        self.assertEqual(api.default_level(self.folder), "frequency")
        self.assertEqual(api.get_corrector(ckpt_dir=self.folder)("mua ban ghe")["text"], "mua bạn ghế")

    def test_levels_and_temperatures(self):
        bigram = api.build_corrector("bigram", self.folder)
        self.assertEqual(bigram.temperature, 1.3)
        self.assertEqual(api.build_corrector("frequency", self.folder).temperature, 0.9)
        self.assertIs(api.get_corrector("bigram", self.folder).__self__,
                      api.get_corrector("bigram", self.folder).__self__)  # loaded once
        with self.assertRaises(ValueError):
            api.get_corrector("magic", self.folder)

    def test_every_level_passes_the_contract_on_edge_cases(self):
        for level in ("identity", "frequency", "bigram"):
            fn = api.get_corrector(level, self.folder)
            for text in EDGE_CASES:
                out = fn(text)
                self.assertEqual(check_correct_output(text, out), [], (level, text, out))
                in_tokens = text.split()
                for a, b in zip(in_tokens, out["text"].split()):
                    self.assertEqual([c.isupper() for c in a], [c.isupper() for c in b], (level, a, b))
            self.assertEqual(fn("") , {"text": "", "conf": []})

    def test_old_load_interface(self):
        api.load(self.folder, method="viterbi")
        self.assertEqual(api.correct("Toi mua ban ghe.")["text"], "Tôi mua bàn ghế.")
        api.load(method="identity")
        self.assertEqual(api.correct("ban toi"), {"text": "ban toi", "conf": [1.0, 1.0]})
        with self.assertRaises(ValueError):
            api.load(method="magic")


if __name__ == "__main__":
    unittest.main()
