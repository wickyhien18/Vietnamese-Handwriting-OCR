"""Unigram and bigram counts of accented syllables, learned from the clean corpus.

Three tables:
    unigram[w]          how often syllable w appears          ("người": 812345)
    bigram[(a, b)]      how often b comes right after a       (("bàn", "ghế"): 1203)
    forms[key]          accented forms of each plain syllable ("nguoi": {"người": 812345, "ngươi": 210})

All syllables are lower-case and without punctuation. "<s>" and "</s>" mark the start and end of
a sentence, so the model also knows which words like to begin or end a sentence.
"""

import gzip
import math
import pickle
import sys
from collections import Counter

from src.common.text_utils import strip_diacritics
from src.corrector.syllable import split_token

START = "<s>"
END = "</s>"


def sentence_words(sentence):
    """'Hà Nội, năm 2020.' -> ['hà', 'nội', 'năm']  (lower-case, punctuation and numbers dropped)"""
    words = []
    for token in sentence.split():
        core = split_token(token)[1]
        if core:
            words.append(sys.intern(core.lower()))  # intern: one copy of each string in memory
    return words


class NgramModel:
    def __init__(self, interpolation=0.8):
        self.unigram = Counter()
        self.bigram = Counter()
        self.forms = {}
        self.total_words = 0
        self.interpolation = interpolation  # weight of the bigram estimate, the rest goes to unigram

    def train(self, sentences):
        for sentence in sentences:
            words = sentence_words(sentence)
            if not words:
                continue
            self.unigram[START] += 1  # number of sentences, needed for P(w | <s>)
            prev = START
            for w in words:
                self.unigram[w] += 1
                self.bigram[(prev, w)] += 1
                key = strip_diacritics(w)
                if key not in self.forms:
                    self.forms[key] = Counter()
                self.forms[key][w] += 1
                prev = w
            self.bigram[(prev, END)] += 1
            self.total_words += len(words)

    def prune(self, min_count=2):
        """Drop bigrams seen fewer than min_count times. Saves a lot of memory, costs little accuracy."""
        self.bigram = Counter({k: c for k, c in self.bigram.items() if c >= min_count})

    def p_unigram(self, w):
        """Add-one smoothing, so an unseen word still gets a small, non-zero probability."""
        vocab = len(self.unigram) + 1
        return (self.unigram[w] + 1) / (self.total_words + vocab)

    def p_bigram(self, prev, w):
        """P(w | prev) = lambda * count(prev, w) / count(prev) + (1 - lambda) * P(w).

        Linear interpolation: when the pair was never seen, the unigram part keeps the
        probability above zero, so log() never fails.
        """
        count_prev = self.unigram[prev]
        p_bi = self.bigram[(prev, w)] / count_prev if count_prev else 0.0
        lam = self.interpolation
        return lam * p_bi + (1 - lam) * self.p_unigram(w)

    def log_p(self, prev, w):
        return math.log(self.p_bigram(prev, w))

    def candidates(self, observed, k=8):
        """Possible clean forms of an observed syllable: the k most frequent accented forms of
        its plain version, plus the observed form itself if the corpus knows it.

        Unknown syllables (foreign names, typos) have only one candidate: themselves.
        """
        forms = self.forms.get(strip_diacritics(observed))
        if not forms:
            return [observed]
        result = [w for w, _ in forms.most_common(k)]
        if observed in forms and observed not in result:
            result.append(observed)
        return result

    def save(self, path):
        with gzip.open(path, "wb") as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        model = cls()
        with gzip.open(path, "rb") as f:
            model.__dict__.update(pickle.load(f))
        return model
