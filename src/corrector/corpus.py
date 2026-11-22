"""Turn raw Vietnamese text (for example Wikipedia articles) into clean sentences, one per line.

Steps for each article:
    1. NFC-normalize, remove text in brackets, collapse whitespace.
    2. Split into sentences (not after an initial such as "Ulysses S. Grant").
    3. Keep only sentences that look like normal Vietnamese (4 to 40 syllables, mostly Vietnamese words).
    4. Put every tone mark in the standard position ("hoà" -> "hòa").

Then for the whole corpus (build_corpus):
    5. Drop duplicates, compared after the normalization of scripts/check_text_overlap.py
       (lower case, punctuation removed), so "Hà Nội." and "hà nội" count as the same sentence.
    6. Choose train / val / test from a hash of that normalized form, so near-duplicates always
       land in the same split and the exact overlap between splits is 0 by construction.
"""

import hashlib
import json
import os
import re
import time

from scripts.check_text_overlap import normalize_tokens, windows
from src.common.text_utils import normalize_nfc
from src.corrector.syllable import normalize_tone_placement, split_token

VIETNAMESE_LETTERS = set(
    "aàáảãạăằắẳẵặâầấẩẫậbcdđeèéẻẽẹêềếểễệghiìíỉĩịklmn"
    "oòóỏõọôồốổỗộơờớởỡợpqrstuùúủũụưừứửữựvxyỳýỷỹỵ"
)
PLAIN_LETTERS = set("abcdeghiklmnopqrstuvxy")  # letters without any mark

MIN_SYLLABLES = 4
MAX_SYLLABLES = 40

BRACKETS = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")
SPACES = re.compile(r"\s+")
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
OPENERS = "\"'(\u201c"


def clean_text(text):
    """NFC, remove "(...)" and "[...]" (mostly foreign names, dates, references), one space only."""
    text = normalize_nfc(text)
    text = BRACKETS.sub(" ", text)
    text = SPACES.sub(" ", text)
    return text.strip()


def ends_with_initial(text):
    """True if text ends with a single capital letter and a period: "Ulysses S.", "A.G.", "Đ."."""
    if len(text) < 2 or text[-1] != "." or not (text[-2].isalpha() and text[-2].isupper()):
        return False
    return len(text) == 2 or text[-3] in " ."


def split_sentences(paragraph):
    """Split after . ! ? … followed by a space, except:
    - after an initial ("Ulysses S. Grant" stays one sentence);
    - when the next character does not start a sentence (lower case, e.g. "v.v. và").
    A sentence starts with a capital letter, a digit, a quote or an opening bracket."""
    sentences, start = [], 0
    for match in SENTENCE_END.finditer(paragraph):
        before = paragraph[start:match.start()]
        following = paragraph[match.end():match.end() + 1]
        if ends_with_initial(before):
            continue
        if not (following.isupper() or following.isdigit() or following in OPENERS):
            continue
        sentences.append(before)
        start = match.end()
    sentences.append(paragraph[start:])
    return [sentence for sentence in sentences if sentence]


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


def overlap_key(sentence):
    """Normalized form used to find duplicates: the same as scripts/check_text_overlap.py
    (NFC, lower case, every character that is not a letter, digit or mark removed)."""
    return " ".join(normalize_tokens(sentence))


def assign_split(sentence, val_percent=1, test_percent=1):
    """Choose train / val / test from a hash of the normalized sentence (overlap_key).

    Sentences that differ only in case or punctuation get the same key, so they always go to
    the same split. The choice does not change when the corpus is rebuilt, and nothing has to
    be kept in memory.
    """
    key = overlap_key(sentence)
    bucket = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < test_percent:
        return "test"
    if bucket < test_percent + val_percent:
        return "val"
    return "train"


def build_corpus(articles, out_dir, max_sentences, log_every=20_000):
    """Write train.txt, val.txt, test.txt and stats.json into out_dir from an iterable of article texts.

    Duplicates (same overlap_key) are written only once. Returns the stats dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    names = ("train", "val", "test")
    files = {name: open(os.path.join(out_dir, f"{name}.txt"), "w", encoding="utf-8") for name in names}
    counts = {name: 0 for name in names}
    seen = set()
    n_articles = duplicates = 0
    start = time.time()
    for text in articles:
        n_articles += 1
        for sentence in sentences_from_article(text):
            key = overlap_key(sentence)
            digest = hashlib.md5(key.encode("utf-8")).digest()
            if digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            split = assign_split(sentence)
            files[split].write(sentence + "\n")
            counts[split] += 1
        total = sum(counts.values())
        if log_every and n_articles % log_every == 0:
            print(f"{n_articles:>8} articles  {total:>9} sentences  {time.time() - start:6.0f} s")
        if total >= max_sentences:
            break
    for handle in files.values():
        handle.close()
    stats = {
        "articles_read": n_articles,
        "sentences": counts,
        "duplicates_skipped": duplicates,
        "min_syllables": MIN_SYLLABLES,
        "max_syllables": MAX_SYLLABLES,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    return stats


def corpus_overlap(train_path, reference_paths, window=8):
    """Overlap between the training sentences and other splits of the corpus.

    reference_paths: {"val": path, "test": path}. For every reference split it returns
        exact        number of reference sentences whose overlap_key also occurs in train
        ngram_any    number of reference sentences that share at least one run of `window` words
        lines        number of reference sentences
    and train_lines_to_remove: 1-based line numbers of train.txt whose key occurs in any reference.
    The training file is streamed, so it does not have to fit in memory.
    """
    ref_keys, ref_grams, ref_lines = {}, {}, {}
    for name, path in reference_paths.items():
        with open(path, encoding="utf-8") as handle:
            tokens_list = [normalize_tokens(line) for line in handle if line.strip()]
        ref_lines[name] = tokens_list
        ref_keys[name] = {" ".join(tokens) for tokens in tokens_list}
        ref_grams[name] = {gram for tokens in tokens_list for gram in windows(tokens, window)}

    remove = []
    found_keys = {name: set() for name in reference_paths}
    found_grams = {name: set() for name in reference_paths}
    with open(train_path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            tokens = normalize_tokens(line)
            key = " ".join(tokens)
            hit = False
            for name in reference_paths:
                if key in ref_keys[name]:
                    found_keys[name].add(key)
                    hit = True
                for gram in windows(tokens, window):
                    if gram in ref_grams[name]:
                        found_grams[name].add(gram)
            if hit:
                remove.append(number)

    result = {"window": window, "train_lines_to_remove": remove}
    for name, tokens_list in ref_lines.items():
        result[name] = {
            "lines": len(tokens_list),
            "exact": sum(" ".join(t) in found_keys[name] for t in tokens_list),
            "ngram_any": sum(any(g in found_grams[name] for g in windows(t, window)) for t in tokens_list),
        }
    return result


def remove_lines(in_path, out_path, line_numbers):
    """Copy a text file without the given 1-based line numbers. Returns the number of lines kept."""
    drop = set(line_numbers)
    kept = 0
    with open(in_path, encoding="utf-8") as source, open(out_path, "w", encoding="utf-8") as target:
        for number, line in enumerate(source, start=1):
            if number not in drop:
                target.write(line)
                kept += 1
    return kept
