"""Experiment E3: a scikit-learn baseline for the diacritic corrector.

Same task as the other levels: for each syllable, choose one of its candidates (at most the
k = 8 most frequent accented forms of its plain version, NgramModel.candidates). Unknown
syllables and syllables with one candidate are handled exactly as in the other levels.

Model: logistic regression that scores every (position, candidate) pair, trained with
SGDClassifier(loss="log_loss") on hashed features (FeatureHasher):
    w=c              the candidate itself (a prior per form)
    w|p=c|prev       candidate with the previous syllable without diacritics
    w|n=c|next       candidate with the next syllable without diacritics
    w|pn=c|prev|next candidate with both neighbours
    same, w|same=c   the candidate is exactly the observed syllable
    logfreq          log(1 + corpus count of the candidate) / 10
The neighbours are used without diacritics, so the features do not depend on the noise.
The chosen form is the candidate with the highest score; the confidence is the softmax of the
scores over the candidates, divided by a temperature fitted on val (calibration.py), as for
the other levels.
"""

import math
import pickle

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier

from src.common.text_utils import strip_diacritics
from src.corrector.correctors import correct_text
from src.corrector.ngram import END, START, sentence_words
from src.corrector.viterbi import softmax_confidence

HYPERPARAMETERS = {"model": "SGDClassifier(loss='log_loss')", "alpha": 1e-6, "max_iter": 20,
                   "tol": 1e-4, "n_features": 2 ** 22, "k": 8}


def candidate_features(observed, i, candidate, model):
    plain = [strip_diacritics(o) for o in observed]
    prev = plain[i - 1] if i > 0 else START
    nxt = plain[i + 1] if i + 1 < len(observed) else END
    same = 1.0 if observed[i] == candidate else 0.0
    return [("w=" + candidate, 1.0),
            ("w|p=" + candidate + "|" + prev, 1.0),
            ("w|n=" + candidate + "|" + nxt, 1.0),
            ("w|pn=" + candidate + "|" + prev + "|" + nxt, 1.0),
            ("same", same),
            ("w|same=" + candidate, same),
            ("logfreq", math.log1p(model.unigram[candidate]) / 10.0)]


def training_rows(pairs, model, k=8):
    """Feature lists and 0/1 labels for every candidate of every syllable that has 2 or more
    candidates and whose clean form is among them (the others cannot teach anything)."""
    features, labels = [], []
    for noisy, clean in pairs:
        observed, truth = sentence_words(noisy), sentence_words(clean)
        if len(observed) != len(truth):
            continue
        for i, o in enumerate(observed):
            cands = model.candidates(o, k)
            if len(cands) < 2 or truth[i] not in cands:
                continue
            for c in cands:
                features.append(candidate_features(observed, i, c, model))
                labels.append(int(c == truth[i]))
    return features, np.array(labels)


class SklearnCorrector:
    """Follows the group contract through correct_text(), like FrequencyCorrector and ViterbiCorrector."""

    def __init__(self, model, classifier, k=8, temperature=1.0, n_features=2 ** 22):
        self.model = model
        self.classifier = classifier
        self.k = k
        self.temperature = temperature
        self.hasher = FeatureHasher(n_features=n_features, input_type="pair", alternate_sign=False)

    def score_rows(self, observed):
        """For each position: (chosen form, {candidate: score})."""
        rows, batch, where = [], [], []
        for i, o in enumerate(observed):
            cands = self.model.candidates(o, self.k)
            if len(cands) < 2:
                rows.append((cands[0], {cands[0]: 0.0}))
                continue
            rows.append(None)
            where.append((i, cands))
            batch.extend(candidate_features(observed, i, c, self.model) for c in cands)
        if batch:
            scores = self.classifier.decision_function(self.hasher.transform(batch))
            start = 0
            for i, cands in where:
                s = {c: float(v) for c, v in zip(cands, scores[start:start + len(cands)])}
                start += len(cands)
                rows[i] = (max(s, key=s.get), s)
        return rows

    def correct_words(self, observed):
        rows = self.score_rows(observed)
        return ([chosen for chosen, _ in rows],
                [softmax_confidence(scores, chosen, self.temperature) for chosen, scores in rows])

    def correct(self, text):
        return correct_text(text, self.correct_words)


def train_sklearn_baseline(model, pairs, seed=0, hp=HYPERPARAMETERS, chunk=10_000):
    """Fit the classifier on noisy training pairs. Returns (SklearnCorrector, number of rows).

    The features are hashed chunk by chunk (chunk sentences at a time), so only the sparse
    matrix is kept in memory, not millions of Python feature lists.
    """
    hasher = FeatureHasher(n_features=hp["n_features"], input_type="pair", alternate_sign=False)
    pairs = list(pairs)
    blocks, label_blocks = [], []
    for start in range(0, len(pairs), chunk):
        features, labels = training_rows(pairs[start:start + chunk], model, hp["k"])
        if len(labels):
            blocks.append(hasher.transform(features))
            label_blocks.append(labels)
    x = sparse.vstack(blocks).tocsr()
    labels = np.concatenate(label_blocks)
    classifier = SGDClassifier(loss="log_loss", alpha=hp["alpha"], max_iter=hp["max_iter"],
                               tol=hp["tol"], random_state=seed)
    classifier.fit(x, labels)
    return SklearnCorrector(model, classifier, hp["k"], 1.0, hp["n_features"]), len(labels)


def save(corrector, path):
    with open(path, "wb") as handle:
        pickle.dump({"classifier": corrector.classifier, "k": corrector.k,
                     "temperature": corrector.temperature,
                     "n_features": corrector.hasher.n_features}, handle)


def load(path, model):
    with open(path, "rb") as handle:
        d = pickle.load(handle)
    return SklearnCorrector(model, d["classifier"], d["k"], d["temperature"], d["n_features"])
