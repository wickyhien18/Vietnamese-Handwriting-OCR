import os
import tempfile
import unittest

from src.corrector.corpus import (
    assign_split,
    build_corpus,
    corpus_overlap,
    ends_with_initial,
    is_good_sentence,
    overlap_key,
    remove_lines,
    split_sentences,
)


class SplitRulesTest(unittest.TestCase):
    def test_initials_do_not_end_a_sentence(self):
        self.assertTrue(ends_with_initial("Ulysses S."))
        self.assertTrue(ends_with_initial("A.G."))
        self.assertFalse(ends_with_initial("Hà Nội."))
        text = "Tổng thống Ulysses S. Grant đến thăm. Ông ở lại ba ngày."
        self.assertEqual(split_sentences(text),
                         ["Tổng thống Ulysses S. Grant đến thăm.", "Ông ở lại ba ngày."])

    def test_quote_starts_a_sentence(self):
        self.assertEqual(split_sentences('Ông nói. "Tôi đi."'), ["Ông nói.", '"Tôi đi."'])

    def test_max_40_syllables(self):
        self.assertTrue(is_good_sentence(" ".join(["người"] * 40)))
        self.assertFalse(is_good_sentence(" ".join(["người"] * 41)))

    def test_near_duplicates_share_key_and_split(self):
        a, b = "Hà Nội là thủ đô.", "hà nội , là thủ đô"
        self.assertEqual(overlap_key(a), overlap_key(b))
        self.assertEqual(assign_split(a), assign_split(b))


class BuildAndOverlapTest(unittest.TestCase):
    def test_build_has_no_exact_overlap_and_overlap_tool_finds_planted_lines(self):
        base = ["Câu số %d nói về thành phố Hà Nội và người dân." % i for i in range(3000)]
        articles = [" ".join(base[i:i + 10]) for i in range(0, 3000, 10)]
        articles.append("CÂU SỐ 1 NÓI VỀ THÀNH PHỐ HÀ NỘI VÀ NGƯỜI DÂN")  # duplicate after normalization
        with tempfile.TemporaryDirectory() as tmp:
            stats = build_corpus(articles, tmp, max_sentences=10_000, log_every=0)
            self.assertEqual(stats["duplicates_skipped"], 1)  # the upper-case copy of sentence 1
            paths = {name: os.path.join(tmp, f"{name}.txt") for name in ("val", "test")}
            result = corpus_overlap(os.path.join(tmp, "train.txt"), paths)
            self.assertEqual(result["val"]["exact"], 0)
            self.assertEqual(result["test"]["exact"], 0)
            self.assertEqual(result["train_lines_to_remove"], [])

            # plant a val sentence (other case and punctuation) into train, then find and remove it
            with open(paths["val"], encoding="utf-8") as f:
                val_sentence = f.readline().strip()
            self.assertTrue(val_sentence)
            train = os.path.join(tmp, "train.txt")
            with open(train, "a", encoding="utf-8") as f:
                f.write(val_sentence.upper().rstrip(".") + " !\n")
            result = corpus_overlap(train, paths)
            self.assertEqual(result["val"]["exact"], 1)
            self.assertEqual(len(result["train_lines_to_remove"]), 1)
            clean = os.path.join(tmp, "train_clean.txt")
            remove_lines(train, clean, result["train_lines_to_remove"])
            self.assertEqual(corpus_overlap(clean, paths)["val"]["exact"], 0)

    def test_duplicates_are_skipped(self):
        articles = ["Hà Nội là thủ đô của Việt Nam.", "hà nội , là thủ đô của việt nam ."]
        with tempfile.TemporaryDirectory() as tmp:
            stats = build_corpus(articles, tmp, max_sentences=100, log_every=0)
        self.assertEqual(sum(stats["sentences"].values()), 1)
        self.assertEqual(stats["duplicates_skipped"], 1)


if __name__ == "__main__":
    unittest.main()
