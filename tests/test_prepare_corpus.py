import unittest

from scripts.prepare_corpus import ALLOWED_CHARS, extract_sentences, iter_sentences, split_sentences

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

    def test_initials_do_not_split_a_sentence(self):
        text = "Day la noi sinh cua cac Tong thong: Ulysses S. Grant (tai Point Pleasant), Rutherford B. Hayes (tai Delaware), James A. Garfield."
        self.assertEqual(extract_sentences(text), [text])

    def test_dotted_initials_do_not_split(self):
        text = "Vi du cua A.G. Haudricourt ve viec do rat noi tieng."
        self.assertEqual(extract_sentences(text), [text])

    def test_lowercase_after_ellipsis_is_a_continuation(self):
        text = "Day la mot danh sach dai nhu Argentina,... hay Nha Trang va nhieu noi khac."
        self.assertEqual(extract_sentences(text), [text])

    def test_real_boundaries_still_split(self):
        text = "Hom nay troi rat dep o Ha Noi. Ngay mai troi se mua o Sai Gon."
        self.assertEqual(len(extract_sentences(text)), 2)
        self.assertEqual(split_sentences(text), ["Hom nay troi rat dep o Ha Noi.", "Ngay mai troi se mua o Sai Gon."])

    def test_fragment_starting_with_lowercase_is_dropped(self):
        self.assertEqual(extract_sentences("hay Toa Bach Oc nam o thu do cua nuoc nay."), [])

    def test_sentence_starting_with_a_digit_is_kept(self):
        text = "18 tuoi la tuoi duoc di bau cu o Hoa Ky."
        self.assertEqual(extract_sentences(text), [text])

    def test_numbers_with_dots_are_not_boundaries(self):
        text = "Dan so thanh pho nay la 8.990.000 nguoi vao nam 2019."
        self.assertEqual(extract_sentences(text), [text])


if __name__ == "__main__":
    unittest.main()
