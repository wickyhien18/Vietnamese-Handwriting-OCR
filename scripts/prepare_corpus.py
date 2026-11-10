"""Build a clean Vietnamese sentence corpus from Vietnamese Wikipedia (Hugging Face).

Source: dataset wikimedia/wikipedia, config 20231101.vi (text under CC BY-SA 3.0 / GFDL,
as stated on the dataset card). Sentences are NFC-normalized, one per line, and only
sentences made of Vietnamese letters, digits and basic punctuation are kept.

Usage (in Colab, needs the datasets package):
  python scripts/prepare_corpus.py --out data/corpus/viwiki_sentences.txt --max-sentences 300000
"""

import argparse
import re
import unicodedata

BASES = "aeiouy" + "\u0103\u00e2\u00ea\u00f4\u01a1\u01b0"  # plus a-breve, a/e/o-circumflex, o/u-horn
TONES = ["", "\u0300", "\u0301", "\u0303", "\u0309", "\u0323"]
PUNCTUATION = set(" ,.;:!?-()'\"%/")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
# a single capital letter right before the period, such as "Ulysses S." or "A.G."
INITIAL = re.compile(r"(?:^|[\s.])[A-Z\u0110]\.$")
OPENERS = "\"'("


def starts_like_sentence(text):
    """A real sentence starts with a capital letter, a digit, a quote or an opening bracket."""
    if not text:
        return False
    first = text[0]
    return first.isupper() or first.isdigit() or first in OPENERS


def split_sentences(paragraph):
    """Split a paragraph at . ! ? followed by a space, but not inside initials or continuations.

    Not a boundary:
      * the text before the mark ends with a single capital letter ("Ulysses S. Grant")
      * the text after the space starts with a lowercase letter (for example after "...")
    """
    pieces = []
    start = 0
    for match in SENTENCE_BOUNDARY.finditer(paragraph):
        before = paragraph[start:match.start()]
        after = paragraph[match.end():match.end() + 1]
        if INITIAL.search(before):
            continue
        if after and not starts_like_sentence(after):
            continue
        pieces.append(before)
        start = match.end()
    pieces.append(paragraph[start:])
    return pieces


def build_allowed_chars():
    allowed = set("\u0111\u0110")
    for base in BASES:
        for tone in TONES:
            for letter in (base, base.upper()):
                allowed.add(unicodedata.normalize("NFC", letter + tone))
    for code in range(ord("a"), ord("z") + 1):
        allowed.add(chr(code))
        allowed.add(chr(code).upper())
    allowed.update("0123456789")
    allowed.update(PUNCTUATION)
    return allowed


ALLOWED_CHARS = build_allowed_chars()


def extract_sentences(text, min_syllables=4, max_syllables=40):
    out = []
    for paragraph in text.split("\n"):
        paragraph = unicodedata.normalize("NFC", paragraph)
        for sentence in split_sentences(paragraph):
            sentence = " ".join(sentence.split())
            if not sentence or sentence[-1] not in ".!?":
                continue
            if not starts_like_sentence(sentence):
                continue
            count = len(sentence.split(" "))
            if count < min_syllables or count > max_syllables:
                continue
            if not all(ch in ALLOWED_CHARS for ch in sentence):
                continue
            out.append(sentence)
    return out


def iter_sentences(documents, min_syllables=4, max_syllables=40):
    """Yield unique clean sentences from an iterable of {'text': ...} documents."""
    seen = set()
    for doc in documents:
        for sentence in extract_sentences(doc["text"], min_syllables, max_syllables):
            if sentence not in seen:
                seen.add(sentence)
                yield sentence


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-sentences", type=int, default=300000)
    parser.add_argument("--min-syllables", type=int, default=4)
    parser.add_argument("--max-syllables", type=int, default=40)
    args = parser.parse_args()

    from datasets import load_dataset

    documents = load_dataset("wikimedia/wikipedia", "20231101.vi", split="train", streaming=True)
    written = 0
    with open(args.out, "w", encoding="utf-8") as handle:
        for sentence in iter_sentences(documents, args.min_syllables, args.max_syllables):
            handle.write(sentence + "\n")
            written += 1
            if written % 50000 == 0:
                print("written %d sentences" % written)
            if written >= args.max_sentences:
                break
    print("wrote %d sentences to %s" % (written, args.out))


if __name__ == "__main__":
    main()
