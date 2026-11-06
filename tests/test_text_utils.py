import unittest

from src.common.text_utils import normalize_nfc, strip_diacritics


class TextUtilsTest(unittest.TestCase):
    def test_strip_basic(self):
        self.assertEqual(strip_diacritics("b\u1ea1n"), "ban")

    def test_strip_d_bar(self):
        self.assertEqual(strip_diacritics("\u0111\u1ecdc"), "doc")

    def test_nfc_equivalence(self):
        precomposed = "\u1ea1"
        decomposed = "a\u0323"
        self.assertNotEqual(precomposed, decomposed)
        self.assertEqual(normalize_nfc(decomposed), precomposed)


if __name__ == "__main__":
    unittest.main()
