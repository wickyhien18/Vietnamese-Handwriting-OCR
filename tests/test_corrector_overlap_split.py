import os
import tempfile
import unittest

from src.corrector.correctors import IdentityCorrector, ViterbiCorrector
from src.corrector.ngram import NgramModel
from src.corrector.overlap_split import MIN_LINES, overlap_flags, split_report
from src.corrector.viterbi import Emission

CORPUS = ["nhiều bạn đọc đã bày tỏ sự quan tâm đến vấn đề này",
          "tôi mua bàn ghế mới cho nhà",
          "câu này không liên quan gì cả"]


class OverlapSplitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "train.txt")
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(CORPUS) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_flags_use_five_word_windows(self):
        labels = ["Nhiều bạn đọc đã bày tỏ ý kiến",      # shares "nhiều bạn đọc đã bày"
                  "hôm nay trời rất đẹp và mát",          # shares nothing
                  "bàn ghế mới"]                           # shorter than 5 words: no window
        flags, stats = overlap_flags(labels, self.path, window=5)
        self.assertEqual([f[0] for f in flags], [True, False, False])
        self.assertEqual(flags[2][1], 0)
        self.assertEqual(stats["corpus_lines"], 3)
        limited, stats1 = overlap_flags(labels, self.path, window=5, corpus_lines=1)
        self.assertEqual(stats1["corpus_lines"], 1)
        self.assertTrue(limited[0][0])

    def test_split_report_counts_and_small_groups(self):
        model = NgramModel()
        model.train(CORPUS * 5)
        corrector = ViterbiCorrector(model, Emission(0.7))
        lines = [("nhiều bạn đọc đã bày tỏ", "nhiều ban đọc đã bày tỏ"),
                 ("tôi mua bàn ghế mới", "toi mua ban ghe moi")]
        rows = split_report(corrector, lines, [(True, 2, 1), (False, 1, 0)])
        self.assertEqual(rows["overlap"]["lines"], 1)
        self.assertEqual(rows["overlap"]["fixed"], 1)
        self.assertEqual(rows["overlap"]["note"], "too few lines to conclude")
        self.assertLess(rows["no overlap"]["cer_after"], rows["no overlap"]["cer_before"])
        same = split_report(IdentityCorrector(), lines, [(True, 2, 1), (False, 1, 0)])
        self.assertEqual(same["overlap"]["net"], 0)
        self.assertGreater(MIN_LINES, 1)


if __name__ == "__main__":
    unittest.main()
