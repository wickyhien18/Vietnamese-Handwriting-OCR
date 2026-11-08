"""Correctors that follow the group contract:

    correct(text) -> {"text": str, "conf": [float]}     with len(conf) == len(text.split())

Each corrector only decides the lower-case syllables (method correct_words). The shared function
correct_text() takes care of punctuation, numbers and upper-case letters, so every corrector
returns text in the same shape as its input.
"""

from src.common.text_utils import strip_diacritics
from src.corrector.syllable import split_token
from src.corrector.viterbi import local_confidence, viterbi


def restore_case(original, word):
    """Copy the capitalization of the input letter by letter:
    ('HA', 'hà') -> 'HÀ', ('Ha', 'hà') -> 'Hà', ('iPhone', 'iphone') -> 'iPhone'.

    Changing diacritics never changes the number of letters, so the two words line up.
    """
    if len(original) == len(word):
        return "".join(w.upper() if o.isupper() else w for o, w in zip(original, word))
    if original[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def correct_text(text, correct_words):
    """Split text into tokens, correct the syllables, put punctuation and case back."""
    parts = [split_token(token) for token in text.split()]
    observed = [core.lower() for _, core, _ in parts if core]
    fixed, conf = correct_words(observed)

    out_tokens, out_conf = [], []
    j = 0
    for prefix, core, suffix in parts:
        if core:
            out_tokens.append(prefix + restore_case(core, fixed[j]) + suffix)
            out_conf.append(conf[j])
            j += 1
        else:  # only numbers or punctuation: copied unchanged
            out_tokens.append(prefix)
            out_conf.append(1.0)
    return {"text": " ".join(out_tokens), "conf": out_conf}


class IdentityCorrector:
    """Changes nothing. The "no correction" row of the results table."""

    def correct_words(self, observed):
        return list(observed), [1.0] * len(observed)

    def correct(self, text):
        return correct_text(text, self.correct_words)


class FrequencyCorrector:
    """Baseline: each syllable becomes the most frequent accented form of its plain version.

    "ban" -> "bạn" always, whatever the neighbours are. Confidence = share of that form,
    for example bạn appears in 60 % of all "ban" -> conf 0.6.
    """

    def __init__(self, model):
        self.model = model

    def correct_words(self, observed):
        words, conf = [], []
        for o in observed:
            forms = self.model.forms.get(strip_diacritics(o))
            if not forms:  # unknown syllable: keep it
                words.append(o)
                conf.append(1.0)
                continue
            best, count = forms.most_common(1)[0]
            words.append(best)
            conf.append(count / sum(forms.values()))
        return words, conf

    def correct(self, text):
        return correct_text(text, self.correct_words)


class ViterbiCorrector:
    """Bigram hidden Markov model decoded with Viterbi: uses the previous and next words,
    so "ban ghe" -> "bàn ghế" but "ban hang" -> "bán hàng"."""

    def __init__(self, model, emission, k=8):
        self.model = model
        self.emission = emission
        self.k = k

    def correct_words(self, observed):
        path = viterbi(observed, self.model, self.emission, self.k)
        conf = local_confidence(path, observed, self.model, self.emission, self.k)
        return path, conf

    def correct(self, text):
        return correct_text(text, self.correct_words)
