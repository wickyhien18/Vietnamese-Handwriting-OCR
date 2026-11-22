"""Compare the hand-written parts of role B with standard libraries (task item 7).

    n-gram + smoothing   nltk.lm: MLE (bigram) and Laplace (add-one unigram), then our interpolation
    Viterbi              brute force over every candidate sequence (sentences of <= 6 syllables)
                         and hmmlearn's Viterbi on random hidden Markov models
    attention            torch.nn.MultiheadAttention with the same weights

The libraries are imported inside the functions, so this module loads without them.
corrector_06_final.ipynb runs everything and writes outputs/corrector_reference_check.json.
"""

import itertools
import math

import numpy as np

from src.common.text_utils import strip_diacritics
from src.corrector.ngram import END, START, NgramModel, sentence_words
from src.corrector.viterbi import path_log_prob, viterbi

# Which parts of the transformer tagger are hand-written and which come ready-made from PyTorch.
TAGGER_DECLARATION = {
    "hand_written": [
        "multi-head scaled dot-product self-attention (SelfAttention.forward: Q K^T / sqrt(d), padding mask, softmax, weighted sum, head split and merge)",
        "pre-norm transformer block wiring (residual connections, order of layers)",
        "candidate scoring head (dot product of the hidden state with one output vector per candidate, padding with -inf)",
        "noisy training data, padding and batching (NoisySentences, collate)",
        "learning-rate schedule (linear warm-up, linear decay)",
        "temperature scaling and ECE (calibration.py, numpy only)",
    ],
    "ready_made_from_pytorch": [
        "nn.Embedding", "nn.Linear", "nn.LayerNorm", "nn.GELU", "nn.Dropout",
        "F.cross_entropy (loss)", "torch.optim.AdamW (optimizer)", "autograd (gradients)",
        "torch.optim.lr_scheduler.LambdaLR (applies our schedule function)",
    ],
    "not_used": ["nn.MultiheadAttention", "nn.TransformerEncoder"],
}


def compare_ngram_with_nltk(sentences, interpolation=0.8):
    """Our probabilities against nltk.lm on the same small corpus (no pruning).

    Our model: P(w | prev) = lambda * c(prev, w) / c(prev) + (1 - lambda) * (c(w) + 1) / (N + |V|)
    with V = training words + "</s>" + "<unk>". nltk: MLE(2) for the first part (same <s> </s>
    padding) and Laplace(1) with the same vocabulary for the second.
    """
    from nltk.lm import MLE, Laplace
    from nltk.lm.preprocessing import padded_everygram_pipeline

    words = [w for w in (sentence_words(s) for s in sentences) if w]
    ours = NgramModel(interpolation)
    ours.train(sentences)

    train, vocab = padded_everygram_pipeline(2, words)
    mle = MLE(2)
    mle.fit(train, vocab)
    laplace = Laplace(1)
    laplace.fit([[(w,) for w in sent] for sent in words],
                vocabulary_text=[w for sent in words for w in sent] + [END])

    vocabulary = sorted({w for sent in words for w in sent})
    diffs = {"mle_bigram": 0.0, "laplace_unigram": 0.0, "interpolated": 0.0}
    n_pairs = 0
    for prev in [START] + vocabulary:
        for w in vocabulary + [END]:
            our_mle = ours.bigram[(prev, w)] / ours.unigram[prev]
            lib_mle = mle.score(w, [prev])
            lib_lap = laplace.score(w)
            lib_interp = interpolation * lib_mle + (1 - interpolation) * lib_lap
            diffs["mle_bigram"] = max(diffs["mle_bigram"], abs(our_mle - lib_mle))
            diffs["laplace_unigram"] = max(diffs["laplace_unigram"], abs(ours.p_unigram(w) - lib_lap))
            diffs["interpolated"] = max(diffs["interpolated"], abs(ours.p_bigram(prev, w) - lib_interp))
            n_pairs += 1
    return {"sentences": len(words), "pairs_compared": n_pairs,
            "max_abs_diff": diffs, "same": all(d < 1e-12 for d in diffs.values())}


def compare_viterbi_brute_force(model, emission, sentences, k=8, max_syllables=6, max_paths=100_000):
    """Viterbi against trying every candidate sequence, on sentences of at most max_syllables
    (input: the sentence without diacritics). Best path and its log-probability must agree."""
    checked = same_path = same_score = 0
    max_diff = 0.0
    for sentence in sentences:
        observed = [strip_diacritics(w) for w in sentence_words(sentence)]
        if not observed or len(observed) > max_syllables:
            continue
        cands = [model.candidates(o, k) for o in observed]
        if math.prod(len(c) for c in cands) > max_paths:
            continue
        best = viterbi(observed, model, emission, k)
        best_score = path_log_prob(best, observed, model, emission)
        brute = max(itertools.product(*cands), key=lambda p: path_log_prob(p, observed, model, emission))
        brute_score = path_log_prob(brute, observed, model, emission)
        checked += 1
        same_path += list(brute) == best
        same_score += abs(best_score - brute_score) < 1e-9
        max_diff = max(max_diff, abs(best_score - brute_score))
    return {"sentences_checked": checked, "same_path": same_path, "same_log_prob": same_score,
            "max_abs_log_prob_diff": max_diff}


class MatrixModel:
    """A hidden Markov model given by matrices, with the interface viterbi() expects:
    every state is a candidate at every position; there is no end probability (log 1 = 0)."""

    def __init__(self, start, trans):
        self.start, self.trans = start, trans
        self.states = list(range(len(start)))

    def candidates(self, observed, k=None):
        return self.states

    def log_p(self, prev, c):
        if prev == START:
            return math.log(self.start[c])
        if c == END:
            return 0.0
        return math.log(self.trans[prev][c])


def compare_viterbi_hmmlearn(n_states=4, n_symbols=5, length=8, trials=20, seed=0):
    """Our viterbi() against hmmlearn.hmm.CategoricalHMM.decode on random models and observations."""
    from hmmlearn.hmm import CategoricalHMM

    rng = np.random.default_rng(seed)
    same_path = 0
    max_diff = 0.0
    for _ in range(trials):
        start = rng.dirichlet(np.ones(n_states))
        trans = rng.dirichlet(np.ones(n_states), size=n_states)
        emis = rng.dirichlet(np.ones(n_symbols), size=n_states)
        obs = rng.integers(n_symbols, size=length)

        hmm = CategoricalHMM(n_components=n_states, n_features=n_symbols, init_params="", params="")
        hmm.startprob_, hmm.transmat_, hmm.emissionprob_ = start, trans, emis
        lib_score, lib_path = hmm.decode(obs.reshape(-1, 1), algorithm="viterbi")

        model = MatrixModel(start, trans)
        emission = lambda o, c: math.log(emis[c][o])
        path = viterbi(list(obs), model, emission, k=n_states)
        score = path_log_prob(path, list(obs), model, emission)
        same_path += list(path) == list(lib_path)
        max_diff = max(max_diff, abs(score - lib_score))
    return {"trials": trials, "states": n_states, "length": length, "same_path": same_path,
            "max_abs_log_prob_diff": max_diff}


def compare_attention(d_model=32, n_heads=4, batch=3, length=7, seed=0):
    """Our SelfAttention against torch.nn.MultiheadAttention with copied weights and the same padding."""
    import torch
    from src.corrector.tagger import SelfAttention

    torch.manual_seed(seed)
    ours = SelfAttention(d_model, n_heads, dropout=0.0).eval()
    ref = torch.nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True).eval()
    with torch.no_grad():
        ref.in_proj_weight.copy_(ours.qkv.weight)  # both pack Q, K, V in this order
        ref.in_proj_bias.copy_(ours.qkv.bias)
        ref.out_proj.weight.copy_(ours.out.weight)
        ref.out_proj.bias.copy_(ours.out.bias)
        x = torch.randn(batch, length, d_model)
        pad = torch.zeros(batch, length, dtype=torch.bool)
        pad[1, 5:] = True
        pad[2, 3:] = True
        a = ours(x, pad)
        b, _ = ref(x, x, x, key_padding_mask=pad, need_weights=False)
    return {"d_model": d_model, "heads": n_heads, "max_abs_diff": float((a - b).abs().max())}
