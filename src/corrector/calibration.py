"""Temperature scaling and calibration error, hand-written with numpy (exam part).

A neural network is often over-confident: it says 99 % when it is right only 90 % of the time.
Temperature scaling divides all logits by one number T before the softmax:

    p = softmax(z / T)      T > 1 makes probabilities softer, T < 1 makes them sharper.

T is chosen on validation data to minimize the negative log-likelihood (NLL) of the right answers.
The argmax does not change (dividing by T > 0 keeps the order), so accuracy stays the same;
only the confidence numbers become more honest.

Logits come as an array of shape (N, K): N positions, K candidates each. Missing candidates
(a position with fewer than K) hold -inf, so their probability is exactly 0.
"""

import numpy as np


def log_softmax(logits):
    """Row-wise log softmax. Subtracting the row maximum avoids overflow in exp()."""
    top = logits.max(axis=1, keepdims=True)
    shifted = logits - top
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def softmax(logits):
    return np.exp(log_softmax(logits))


def nll(logits, targets, temperature):
    """Mean negative log-probability of the right candidate after dividing logits by T."""
    logp = log_softmax(logits / temperature)
    return -logp[np.arange(len(targets)), targets].mean()


def fit_temperature(logits, targets, t_min=0.05, t_max=20.0, steps=80):
    """Find the T with the lowest NLL by ternary search on beta = 1 / T.

    Why ternary search works: NLL(beta) = mean(logsumexp(beta * z) - beta * z_target).
    logsumexp of a linear function of beta is convex, minus a linear term is still convex,
    so NLL has a single minimum in beta. Each step keeps the two thirds of the interval that
    must contain it; after 80 steps the interval is about (2/3)^80 = 1e-14 of its first width.
    """
    low, high = 1.0 / t_max, 1.0 / t_min
    for _ in range(steps):
        m1 = low + (high - low) / 3
        m2 = high - (high - low) / 3
        if nll(logits, targets, 1.0 / m1) < nll(logits, targets, 1.0 / m2):
            high = m2
        else:
            low = m1
    return 1.0 / ((low + high) / 2)


def expected_calibration_error(confidence, correct, n_bins=10):
    """ECE: put predictions into bins by confidence (0-0.1, 0.1-0.2, ...). In each bin compare the
    average confidence with the real accuracy. ECE = weighted mean of |accuracy - confidence|.

    0 means perfectly calibrated: among answers given with 80 % confidence, 80 % are right.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if lo == 0.0:
            in_bin |= confidence == 0.0
        if in_bin.any():
            gap = abs(correct[in_bin].mean() - confidence[in_bin].mean())
            ece += gap * in_bin.mean()
    return ece


def calibration_report(logits, targets, temperature):
    """Accuracy, NLL and ECE with T = 1 (raw) and with the fitted T."""
    pred = logits.argmax(axis=1)
    correct = pred == targets
    report = {"accuracy": float(correct.mean()), "temperature": float(temperature)}
    for name, t in [("raw", 1.0), ("scaled", temperature)]:
        conf = softmax(logits / t)[np.arange(len(pred)), pred]
        report[f"nll_{name}"] = float(nll(logits, targets, t))
        report[f"ece_{name}"] = float(expected_calibration_error(conf, correct))
    return report
