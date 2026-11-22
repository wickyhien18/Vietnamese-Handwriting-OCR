"""Repeatable experiments for role B (task items 4, 5, 8 and 11).

    contiguous_folds / ngram_folds   5 folds in contiguous blocks (the corpus is ordered by article,
                                     so shuffling single sentences would put neighbours of training
                                     sentences into the held-out part)
    mean_std                         mean and SAMPLE standard deviation (ddof = 1)
    collect_level_logits             candidate scores of the frequency / bigram level, for
                                     temperature scaling (calibration.py)
    confidence_pairs                 (confidence, right?) per syllable, for ECE and histograms
    contract_violations              check_correct_output on every line

One command for the n-gram levels (CPU), from the corpus to a JSON file with fixed seeds:

    python -m src.corrector.experiments --corpus-dir data/corpus --out outputs/corrector_eval_ngram.json

Accuracy on the corpus is not the result of the project: the result (fixed / broken on real OCR,
tau sweep) is measured by role C on the real OCR val split.
"""

import argparse
import itertools
import json
import math
import time

import numpy as np

from src.common.contract_checks import check_correct_output
from src.corrector.correctors import FrequencyCorrector, ViterbiCorrector
from src.corrector.evaluate import evaluate
from src.corrector.ngram import NgramModel, sentence_words
from src.corrector.noise import NoiseConfig, make_pairs
from src.corrector.viterbi import Emission, estimate_p_same

NGRAM_HYPERPARAMETERS = {"interpolation": 0.8, "min_bigram": 2, "k": 8, "p_same_pairs": 20_000,
                         "p_same_seed": 1, "noise_seed": 123}


def read_lines(path, start=0, n=None):
    with open(path, encoding="utf-8") as handle:
        stop = None if n is None else start + n
        return [line.strip() for line in itertools.islice(handle, start, stop) if line.strip()]


def mean_std(values):
    """Mean and sample standard deviation (divides by n - 1). One value -> std 0."""
    values = [float(v) for v in values]
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def contiguous_folds(n_items, k=5):
    """k blocks [start, end) of consecutive items covering 0 .. n_items, sizes differ by at most 1."""
    bounds = [round(i * n_items / k) for i in range(k + 1)]
    return list(zip(bounds[:-1], bounds[1:]))


def spread_sample(items, n):
    """n items spread evenly over the list (every step-th item), so all articles of a block count."""
    if len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def train_ngram(sentences, hp=NGRAM_HYPERPARAMETERS):
    model = NgramModel(hp["interpolation"])
    model.train(sentences)
    model.prune(hp["min_bigram"])
    pairs = make_pairs(sentences[: hp["p_same_pairs"]], seed=hp["p_same_seed"])
    return model, estimate_p_same(pairs)


def ngram_folds(sentences, k=5, n_eval=2000, config=None, hp=NGRAM_HYPERPARAMETERS, log=print):
    """Frequency table and bigram + Viterbi on k contiguous folds. For fold i the model is trained on
    the other k - 1 blocks and evaluated on n_eval noisy sentences spread over block i."""
    config = config or NoiseConfig()
    runs = {"frequency": [], "bigram": []}
    for fold, (start, end) in enumerate(contiguous_folds(len(sentences), k)):
        t0 = time.time()
        model, p_same = train_ngram(sentences[:start] + sentences[end:], hp)
        held_out = spread_sample(sentences[start:end], n_eval)
        pairs = list(make_pairs(held_out, seed=hp["noise_seed"], config=config))
        train_seconds = time.time() - t0
        for name, corrector in (("frequency", FrequencyCorrector(model, hp["k"])),
                                ("bigram", ViterbiCorrector(model, Emission(p_same), hp["k"]))):
            t1 = time.time()
            result = evaluate(corrector, pairs, model)
            result.update(fold=fold, block=[start, end], p_same=p_same,
                          train_seconds=round(train_seconds, 1), eval_seconds=round(time.time() - t1, 1))
            runs[name].append(result)
            log(f"fold {fold}: {name:<9} all {result['all']:.4f}  ambiguous {result['ambiguous']:.4f}")
    return {name: summarize_runs(r, "fold", hp) for name, r in runs.items()}


def summarize_runs(runs, unit, hyperparameters):
    """mean, std (sample), n_runs and the run ids for every metric, plus the single runs."""
    summary = {"n_runs": len(runs), "std": "sample standard deviation (ddof=1)",
               "runs_are": unit, "seeds": [r.get(unit) for r in runs], "hyperparameters": hyperparameters}
    for metric in ("all", "ambiguous", "noisy", "broken"):
        mean, std = mean_std([r[metric] for r in runs])
        summary[metric] = {"mean": mean, "std": std}
    summary["runs"] = runs
    return summary


def collect_level_logits(corrector, pairs, k_max):
    """Candidate scores (N, k_max) and the index of the right candidate (N,) for every syllable
    with at least two candidates whose clean form is among them. Missing slots hold -inf."""
    logits, targets = [], []
    for noisy, clean in pairs:
        observed, truth = sentence_words(noisy), sentence_words(clean)
        if len(observed) != len(truth):
            continue
        for (chosen, scores), right in zip(corrector.score_rows(observed), truth):
            if len(scores) < 2 or right not in scores:
                continue
            cands = list(scores)
            row = [scores[c] for c in cands] + [-math.inf] * (k_max - len(cands))
            logits.append(row[:k_max])
            targets.append(cands.index(right))
    return np.array(logits, dtype=float), np.array(targets, dtype=int)


def confidence_pairs(corrector, pairs):
    """(confidence, 1 if the output syllable equals the clean one else 0) for every syllable."""
    conf, right = [], []
    for noisy, clean in pairs:
        out = corrector.correct(noisy)
        for token_out, token_clean, c in zip(out["text"].split(), clean.split(), out["conf"]):
            conf.append(c)
            right.append(int(token_out == token_clean))
    return conf, right


def contract_violations(correct_fn, texts, keep=5):
    """Number of inputs whose output breaks the contract, and the first few problems."""
    bad, examples = 0, []
    for text in texts:
        problems = check_correct_output(text, correct_fn(text))
        if problems:
            bad += 1
            if len(examples) < keep:
                examples.append({"input": text, "problems": problems})
    return {"lines": len(texts), "violations": bad, "examples": examples}


def main():
    parser = argparse.ArgumentParser(description="5-fold corpus evaluation of the n-gram levels.")
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-sentences", type=int, default=1_000_000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-eval", type=int, default=2000)
    args = parser.parse_args()
    sentences = read_lines(f"{args.corpus_dir}/train.txt", 0, args.n_sentences)
    result = {"corpus": args.corpus_dir, "n_sentences": len(sentences), "noise": NoiseConfig().__dict__,
              "ambiguous_means": "a syllable whose form without diacritics has 2 or more accented forms "
                                 "in the training text of that fold",
              "levels": ngram_folds(sentences, args.folds, args.n_eval)}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
