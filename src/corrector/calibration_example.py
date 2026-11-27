"""Worked example of temperature scaling on simulated data (report section C7).

A model that is too confident: for every sample it reports P(class 0) = sigma(2z),
while the true probability is sigma(z). So the right temperature is exactly T = 2.
We check that calibration.py finds it, and measure ECE before and after.

    python -m src.corrector.calibration_example

Same definitions as for the real corrector: ECE over the predicted class, 10 equal-width
bins weighted by their share of samples; T chosen by the lowest NLL.
"""

import argparse
import json

import numpy as np

from src.corrector.calibration import expected_calibration_error, fit_temperature, nll, softmax


def simulate(n=20_000, scale=2.0, seed=0):
    """z ~ N(0, scale^2). Class 0 is right with probability sigma(z); the model's logits are (2z, 0)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, scale, n)
    class0_right = rng.random(n) < 1.0 / (1.0 + np.exp(-z))
    logits = np.stack([2.0 * z, np.zeros(n)], axis=1)
    targets = np.where(class0_right, 0, 1)
    return logits, targets


def ece_at(logits, targets, temperature):
    p = softmax(logits / temperature)
    pred = p.argmax(axis=1)
    return expected_calibration_error(p[np.arange(len(pred)), pred], pred == targets)


def run(n=20_000, scale=2.0, seed=0):
    logits, targets = simulate(n, scale, seed)
    grid = [round(0.5 + 0.05 * i, 2) for i in range(71)]          # 0.50, 0.55, ..., 4.00
    t_grid = min(grid, key=lambda t: nll(logits, targets, t))
    t_search = fit_temperature(logits, targets)                     # ternary search, as for the corrector
    accuracy = float((logits.argmax(axis=1) == targets).mean())
    return {
        "samples": n, "z": f"N(0, {scale}^2)", "seed": seed, "true_temperature": 2.0,
        "accuracy": accuracy,
        "T_grid_0.5_to_4_step_0.05": t_grid, "T_ternary_search": round(t_search, 4),
        "ece_T1": round(ece_at(logits, targets, 1.0), 4),
        "ece_T_grid": round(ece_at(logits, targets, t_grid), 4),
        "ece_T_search": round(ece_at(logits, targets, t_search), 4),
        "nll_T1": round(float(nll(logits, targets, 1.0)), 4),
        "nll_T_grid": round(float(nll(logits, targets, t_grid)), 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.n, args.scale, args.seed), indent=1))


if __name__ == "__main__":
    main()
