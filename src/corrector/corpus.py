"""Turn raw Vietnamese text (for example Wikipedia articles) into clean sentences, one per line.

Steps for each article:
    1. NFC-normalize, remove text in brackets, collapse whitespace.
    2. Split into sentences.
    3. Keep only sentences that look like normal Vietnamese (right length, mostly Vietnamese words).
    4. Put every tone mark in the standard position ("hoà" -> "hòa").
"""

import hashlib
import re

from src.common.text_utils import normalize_nfc
from src.corrector.syllable import normalize_tone_placement, split_token

VIETNAMESE_LETTERS = set(
    "aàáảãạăằắẳẵặâầấẩẫậbcdđeèéẻẽẹêềếểễệghiìíỉĩịklmn"
    "oòóỏõọôồốổỗộơờớởỡợpqrstuùúủũụưừứửữựvxyỳýỷỹỵ"
)
PLAIN_LETTERS = set("abcdeghiklmnopqrstuvxy")  # letters without any mark

MIN_SYLLABLES = 4
MAX_SYLLABLES = 50

BRACKETS = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")
SPACES = re.compile(r"\s+")
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def clean_text(text):
    """NFC, remove "(...)" and "[...]" (mostly foreign names, dates, references), one space only."""
    text = normalize_nfc(text)
    text = BRACKETS.sub(" ", text)
    text = SPACES.sub(" ", text)
    return text.strip()


def split_sentences(paragraph):
    """Split after . ! ? … when the next piece starts with a capital letter or a digit."""
    pieces = SENTENCE_END.split(paragraph)
    sentences = []
    for piece in pieces:
        starts_new = piece[:1].isupper() or piece[:1].isdigit()
        if sentences and not starts_new:
            sentences[-1] += " " + piece  # e.g. "v.v. và" is not a new sentence
        else:
            sentences.append(piece)
    return sentences


def is_vietnamese_word(word):
    return all(ch in VIETNAMESE_LETTERS for ch in word.lower())


def has_diacritic(word):
    return any(ch not in PLAIN_LETTERS for ch in word.lower())


def is_good_sentence(sentence):
    """Keep sentences of normal length where almost every word is Vietnamese."""
    tokens = sentence.split()
    if not MIN_SYLLABLES <= len(tokens) <= MAX_SYLLABLES:
        return False
    words = [split_token(t)[1] for t in tokens]
    words = [w for w in words if w]  # ignore tokens that are only numbers or punctuation
    if len(words) < MIN_SYLLABLES:
        return False
    vietnamese = sum(is_vietnamese_word(w) for w in words)
    if vietnamese / len(words) < 0.9:   # too many foreign words or symbols
        return False
    accented = sum(has_diacritic(w) for w in words)
    if accented / len(words) < 0.3:     # English or other text without Vietnamese marks
        return False
    return True


def normalize_sentence(sentence):
    """Standard tone placement for every word, punctuation unchanged."""
    out = []
    for token in sentence.split():
        prefix, core, suffix = split_token(token)
        out.append(prefix + normalize_tone_placement(core) + suffix)
    return " ".join(out)


def sentences_from_article(text):
    """All good, normalized sentences of one article."""
    result = []
    for paragraph in text.split("\n"):
        paragraph = clean_text(paragraph)
        if not paragraph:
            continue
        for sentence in split_sentences(paragraph):
            if is_good_sentence(sentence):
                result.append(normalize_sentence(sentence))
    return result


def assign_split(sentence, val_percent=1, test_percent=1):
    """Choose train / val / test from a hash of the sentence.

    The same sentence always goes to the same split, even if the corpus is rebuilt,
    and nothing has to be kept in memory.
    """
    bucket = int(hashlib.md5(sentence.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < test_percent:
        return "test"
    if bucket < test_percent + val_percent:
        return "val"
    return "train"
