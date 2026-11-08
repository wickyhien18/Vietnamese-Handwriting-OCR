import random
import unicodedata
import unittest

from src.common.text_utils import strip_diacritics
from src.corrector.corpus import (
    assign_split,
    clean_text,
    is_good_sentence,
    normalize_sentence,
    sentences_from_article,
    split_sentences,
)
from src.corrector.noise import NoiseConfig, add_noise, make_pairs
from src.corrector.syllable import (
    compose,
    decompose,
    normalize_tone_placement,
    split_token,
    tone_position,
)

ALL_LETTERS = (
    "aàáảãạăằắẳẵặâầấẩẫậbcdđeèéẻẽẹêềếểễệghiìíỉĩịklmn"
    "oòóỏõọôồốổỗộơờớởỡợpqrstuùúủũụưừứửữựvxyỳýỷỹỵ"
)

SENTENCE = "Người Việt Nam đã bày tỏ sự quan tâm, theo báo Hà Nội."


class SyllableTest(unittest.TestCase):
    def test_round_trip_every_letter(self):
        for ch in ALL_LETTERS + ALL_LETTERS.upper():
            self.assertEqual(compose(decompose(ch)), ch)

    def test_tone_position(self):
        # word -> letter that should carry the tone
        cases = {
            "hoa": "o", "hoan": "a", "thuy": "u", "gi": "i", "gia": "a", "quy": "y",
            "nguoi": "o", "ngoai": "a", "khuyu": "y", "huynh": "y", "mua": "u",
        }
        for word, letter in cases.items():
            chars = decompose(word)
            if word == "nguoi":
                chars = decompose("ngươi")
            self.assertEqual(chars[tone_position(chars)][0], letter, word)

    def test_normalize_tone_placement(self):
        self.assertEqual(normalize_tone_placement("hoà"), "hòa")
        self.assertEqual(normalize_tone_placement("thuỷ"), "thủy")
        self.assertEqual(normalize_tone_placement("người"), "người")
        self.assertEqual(normalize_tone_placement("quý"), "quý")

    def test_split_token(self):
        self.assertEqual(split_token('"Hà,'), ('"', "Hà", ","))
        self.assertEqual(split_token("1990."), ("1990.", "", ""))


class NoiseTest(unittest.TestCase):
    def test_same_number_of_tokens(self):
        rng = random.Random(1)
        config = NoiseConfig(0.0, 0.3, 0.5, 0.5)
        for _ in range(200):
            noisy = add_noise(SENTENCE, rng, config)
            self.assertEqual(len(noisy.split()), len(SENTENCE.split()))
            self.assertEqual(noisy, unicodedata.normalize("NFC", noisy))

    def test_no_noise_keeps_sentence(self):
        rng = random.Random(1)
        self.assertEqual(add_noise(SENTENCE, rng, NoiseConfig(0, 0, 0, 0)), SENTENCE)

    def test_strip_whole_sentence(self):
        rng = random.Random(1)
        out = add_noise(SENTENCE, rng, NoiseConfig(1.0, 0, 0, 0))
        self.assertEqual(out, strip_diacritics(SENTENCE))

    def test_tone_noise_changes_text(self):
        rng = random.Random(1)
        out = add_noise("bạn", rng, NoiseConfig(0, 0, 1.0, 0))
        self.assertNotEqual(out, "bạn")
        self.assertEqual(strip_diacritics(out), "ban")  # only the tone changed

    def test_same_seed_same_noise(self):
        a = list(make_pairs([SENTENCE] * 20, seed=7))
        b = list(make_pairs([SENTENCE] * 20, seed=7))
        self.assertEqual(a, b)


class CorpusTest(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(clean_text("Hà Nội (tiếng Anh: Hanoi)  là   thủ đô"), "Hà Nội là thủ đô")

    def test_split_sentences(self):
        self.assertEqual(
            split_sentences("Trời mưa. Tôi ở nhà. 1990 là năm đó."),
            ["Trời mưa.", "Tôi ở nhà.", "1990 là năm đó."],
        )

    def test_good_and_bad_sentences(self):
        self.assertTrue(is_good_sentence("Hà Nội là thủ đô của nước Việt Nam."))
        self.assertFalse(is_good_sentence("This is an English sentence for testing."))
        self.assertFalse(is_good_sentence("Quá ngắn."))

    def test_normalize_sentence(self):
        self.assertEqual(normalize_sentence("Hoà bình, thuỷ lợi."), "Hòa bình, thủy lợi.")

    def test_article(self):
        article = "Hà Nội là thủ đô của Việt Nam. Short.\n\nThe city is old and very big."
        self.assertEqual(sentences_from_article(article), ["Hà Nội là thủ đô của Việt Nam."])

    def test_assign_split_is_stable(self):
        self.assertEqual(assign_split("abc"), assign_split("abc"))
        self.assertIn(assign_split("abc"), {"train", "val", "test"})


if __name__ == "__main__":
    unittest.main()
