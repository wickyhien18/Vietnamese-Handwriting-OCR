import unittest

from scripts.prepare_corpus import ALLOWED_CHARS, extract_sentences, iter_sentences

GOOD = "H\u00e0 N\u1ed9i l\u00e0 th\u1ee7 \u0111\u00f4 c\u1ee7a Vi\u1ec7t Nam."


class CorpusTest(unittest.TestCase):
    def test_vietnamese_letters_are_allowed(self):
        for ch in "\u0111\u0110\u1ea1\u1ead\u01a1\u01b0\u00e9\u00d4":
            self.assertIn(ch, ALLOWED_CHARS)

    def test_foreign_letters_are_not_allowed(self):
        self.assertNotIn("\u00f1", ALLOWED_CHARS)
        self.assertNotIn("\u4e2d", ALLOWED_CHARS)

    def test_good_sentence_is_kept(self):
        self.assertEqual(extract_sentences(GOOD), [GOOD])

    def test_rejections(self):
        self.assertEqual(extract_sentences("H\u00e0 N\u1ed9i l\u00e0 th\u1ee7 \u0111\u00f4 c\u1ee7a Vi\u1ec7t Nam"), [])  # no end mark
        self.assertEqual(extract_sentences("Xin ch\u00e0o."), [])  # too short
        self.assertEqual(extract_sentences("H\u00e0 N\u1ed9i l\u00e0 th\u1ee7 \u0111\u00f4 \u4e2d Nam."), [])  # foreign char
        self.assertEqual(extract_sentences("Se\u00f1or l\u00e0 m\u1ed9t t\u1eeb l\u1ea1."), [])  # foreign letter

    def test_non_nfc_input_is_normalized(self):
        decomposed = "Ba\u0323n l\u00e0 ng\u01b0\u1eddi b\u1ea1n t\u1ed1t c\u1ee7a t\u00f4i."
        out = extract_sentences(decomposed)
        self.assertEqual(len(out), 1)
        self.assertIn("B\u1ea1n", out[0])

    def test_splits_on_sentence_ends_and_paragraphs(self):
        text = GOOD + " " + GOOD.replace("H\u00e0 N\u1ed9i", "Hu\u1ebf") + "\nTi\u00eau \u0111\u1ec1 ng\u1eafn\n" + GOOD
        self.assertEqual(len(extract_sentences(text)), 3)

    def test_iter_sentences_deduplicates(self):
        documents = [{"text": GOOD}, {"text": GOOD + " " + GOOD}]
        self.assertEqual(list(iter_sentences(documents)), [GOOD])


if __name__ == "__main__":
    unittest.main()
