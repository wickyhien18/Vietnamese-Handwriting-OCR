"""Viterbi decoding for diacritic restoration (hand-written, exam part).

The model is a hidden Markov model:
    hidden states : the clean syllables w_1 .. w_n            (what we want)
    observations  : the noisy syllables o_1 .. o_n            (what the OCR gave us)
    transitions   : P(w_i | w_{i-1})  from the bigram model
    emissions     : P(o_i | w_i)      from the Emission class below

We want the sequence w_1 .. w_n with the highest
    P(w_1 | <s>) * P(o_1 | w_1) * P(w_2 | w_1) * P(o_2 | w_2) * ... * P(</s> | w_n)

Trying every sequence costs K^n (K candidates per position). Viterbi needs only n * K^2, because
of one fact: the best path that ends in candidate c at position i must continue the best path that
ends in some candidate p at position i-1. So for every position and every candidate we keep only
    score[i][c] = log-probability of the best path that ends with c at position i
    back[i][c]  = the candidate at position i-1 on that best path
and at the end we follow the back pointers from the best final candidate.
We add log-probabilities instead of multiplying probabilities, so long sentences do not underflow to 0.
"""

import math

from src.corrector.ngram import END, START


class Emission:
    """P(observed | clean): the observed syllable is the clean one with probability p_same,
    otherwise it is one of the other forms.

    p_same is measured on noisy training pairs (see estimate_p_same).
    """

    def __init__(self, p_same):
        self.log_same = math.log(p_same)
        self.log_diff = math.log(1 - p_same)

    def __call__(self, observed, clean):
        return self.log_same if observed == clean else self.log_diff


def estimate_p_same(pairs):
    """Share of syllables that the noise left unchanged, from (noisy, clean) sentence pairs."""
    same = total = 0
    for noisy, clean in pairs:
        for o, c in zip(noisy.split(), clean.split()):
            same += o == c
            total += 1
    return same / total


def viterbi(observed, model, emission, k=8):
    """observed: lower-case noisy syllables. Returns the most probable clean syllables."""
    n = len(observed)
    if n == 0:
        return []
    cands = [model.candidates(o, k) for o in observed]

    # Position 0: the previous word is the sentence start.
    score = [{c: model.log_p(START, c) + emission(observed[0], c) for c in cands[0]}]
    back = [{c: None for c in cands[0]}]

    # Positions 1 .. n-1: for each candidate c, find the best previous candidate p.
    for i in range(1, n):
        score.append({})
        back.append({})
        for c in cands[i]:
            best_prev, best = None, -math.inf
            for p, s in score[i - 1].items():
                value = s + model.log_p(p, c)
                if value > best:
                    best, best_prev = value, p
            score[i][c] = best + emission(observed[i], c)
            back[i][c] = best_prev

    # End of sentence: also count P(</s> | last word), then follow the back pointers.
    last = max(score[-1], key=lambda c: score[-1][c] + model.log_p(c, END))
    path = [last]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    path.reverse()
    return path


def path_log_prob(path, observed, model, emission):
    """Log-probability of one full path. Used by the tests to check Viterbi against brute force."""
    total = 0.0
    prev = START
    for w, o in zip(path, observed):
        total += model.log_p(prev, w) + emission(o, w)
        prev = w
    return total + model.log_p(prev, END)


def local_confidence(path, observed, model, emission, k=8):
    """Confidence of each chosen syllable: its probability among all candidates at that position,
    with its two neighbours on the path held fixed.

        conf_i = exp(score(c_i)) / sum over candidates c of exp(score(c))
        score(c) = log P(c | w_{i-1}) + log P(w_{i+1} | c) + log P(o_i | c)

    One candidate only -> confidence 1.0.
    """
    conf = []
    n = len(path)
    for i, chosen in enumerate(path):
        prev = path[i - 1] if i > 0 else START
        nxt = path[i + 1] if i + 1 < n else END
        scores = {}
        for c in model.candidates(observed[i], k):
            scores[c] = model.log_p(prev, c) + model.log_p(c, nxt) + emission(observed[i], c)
        top = max(scores.values())
        z = sum(math.exp(s - top) for s in scores.values())  # subtract the max to avoid overflow
        conf.append(math.exp(scores[chosen] - top) / z)
    return conf
