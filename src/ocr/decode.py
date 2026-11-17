"""Hand-written CTC decoders (Greedy and Prefix Beam Search) implemented in NumPy."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np


def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b)) in log-probability space."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def char_confs_to_syllable_confs(text: str, char_confs: list[float]) -> list[float]:
    """Aggregate per-character confidence scores into per-syllable confidence scores."""
    syllables = text.split()
    if not syllables:
        return []

    syllable_confs: list[float] = []
    cursor = 0
    for syl in syllables:
        pos = text.find(syl, cursor)
        if pos == -1:
            syllable_confs.append(0.0)
            continue
        segment = char_confs[pos : pos + len(syl)]
        avg_conf = float(np.mean(segment)) if segment else 0.0
        syllable_confs.append(round(avg_conf, 6))
        cursor = pos + len(syl)
    return syllable_confs


def ctc_greedy_decode(
    log_probs: np.ndarray,
    idx2char: Mapping[int, str],
    blank: int = 0,
    per_syllable: bool = False,
) -> tuple[str, list[float]]:
    """Decode a single CTC log-probability matrix using Greedy (best-path) decoding.

    At each frame t, pick the highest-probability token. Consecutive duplicate
    tokens are merged into one, and `blank` tokens (index 0) are removed.
    The blank token allows two identical characters to appear side-by-side
    (e.g., 'o' + '<blank>' + 'o' -> 'oo').

    Args:
        log_probs: 2D NumPy array of shape (T, num_classes) containing log-probabilities.
        idx2char: Mapping from integer class index to character string.
        blank: Index reserved for the CTC <blank> symbol (default: 0).
        per_syllable: If True, return one confidence per whitespace-separated syllable;
            otherwise return one confidence per decoded character.

    Returns:
        text: Decoded string.
        confs: List of confidence scores in [0.0, 1.0].
    """
    if log_probs.ndim != 2:
        raise ValueError(f"Expected 2D log_probs (T, C), got shape {log_probs.shape}")
    if log_probs.shape[0] == 0:
        return "", []

    best_indices = np.argmax(log_probs, axis=1)
    best_probs = np.exp(np.max(log_probs, axis=1))

    chars: list[str] = []
    char_confs: list[float] = []
    prev_idx: int | None = None
    active_probs: list[float] = []

    for token_idx, prob in zip(best_indices, best_probs):
        idx = int(token_idx)
        p_val = float(np.clip(prob, 0.0, 1.0))

        if idx == blank:
            if active_probs:
                char_confs.append(round(float(np.mean(active_probs)), 6))
                active_probs = []
        elif idx == prev_idx:
            # Same character sustained across consecutive frames without blank separator
            active_probs.append(p_val)
        else:
            if active_probs:
                char_confs.append(round(float(np.mean(active_probs)), 6))
            chars.append(idx2char[idx])
            active_probs = [p_val]

        prev_idx = idx

    if active_probs:
        char_confs.append(round(float(np.mean(active_probs)), 6))

    text = "".join(chars)
    if per_syllable:
        return text, char_confs_to_syllable_confs(text, char_confs)
    return text, char_confs


def ctc_beam_search_decode(
    log_probs: np.ndarray,
    idx2char: Mapping[int, str],
    beam_width: int = 10,
    top_k: int = 15,
    blank: int = 0,
    per_syllable: bool = False,
) -> tuple[str, list[float]]:
    """Decode a single CTC log-probability matrix using Prefix Beam Search in NumPy.

    Unlike greedy decoding, prefix beam search merges multiple frame-level alignments
    that collapse to the same output string. For each candidate prefix, we track:
        - `p_b`: log-probability of all paths ending with <blank>.
        - `p_nb`: log-probability of all paths ending with a non-blank character.

    Tracking `p_b` and `p_nb` separately is essential because appending the same
    character `c` as `prefix[-1]` stays on `prefix` when coming from `p_nb`,
    but extends to `prefix + (c,)` when coming from `p_b`.

    Args:
        log_probs: 2D NumPy array of shape (T, num_classes) containing log-probabilities.
        idx2char: Mapping from integer class index to character string.
        beam_width: Number of top candidate prefixes retained after each frame.
        top_k: Number of highest-scoring classes explored at each frame for speed.
        blank: Index reserved for the CTC <blank> symbol (default: 0).
        per_syllable: If True, return one confidence per whitespace-separated syllable;
            otherwise return one confidence per decoded character.

    Returns:
        text: Best decoded string.
        confs: List of confidence scores in [0.0, 1.0].
    """
    if log_probs.ndim != 2:
        raise ValueError(f"Expected 2D log_probs (T, C), got shape {log_probs.shape}")

    num_steps, num_classes = log_probs.shape
    if num_steps == 0:
        return "", []

    k = min(top_k, num_classes)
    # prefix_tuple -> (log_prob_blank, log_prob_non_blank, tuple_of_char_confs)
    beams: dict[tuple[int, ...], tuple[float, float, tuple[float, ...]]] = {
        (): (0.0, -math.inf, ())
    }

    for t in range(num_steps):
        frame_lp = log_probs[t]
        top_indices = np.argsort(frame_lp)[-k:]
        next_beams: dict[tuple[int, ...], tuple[float, float, tuple[float, ...]]] = {}

        for prefix, (p_b, p_nb, confs) in beams.items():
            p_total = _log_add(p_b, p_nb)

            # Case 1: Emit <blank> -> prefix stays unchanged, ends in blank
            cur_b, cur_nb, cur_c = next_beams.get(prefix, (-math.inf, -math.inf, confs))
            next_beams[prefix] = (
                _log_add(cur_b, p_total + float(frame_lp[blank])),
                cur_nb,
                cur_c,
            )

            # Case 2: Emit a non-blank character c from top-k candidates
            for raw_c in top_indices:
                c = int(raw_c)
                if c == blank:
                    continue

                c_lp = float(frame_lp[c])
                c_prob = round(float(np.clip(math.exp(c_lp), 0.0, 1.0)), 6)

                if prefix and c == prefix[-1]:
                    # 2a. Repeated character without intervening blank -> same prefix
                    same_b, same_nb, same_c = next_beams.get(
                        prefix, (-math.inf, -math.inf, confs)
                    )
                    next_beams[prefix] = (
                        same_b,
                        _log_add(same_nb, p_nb + c_lp),
                        same_c,
                    )

                    # 2b. Repeated character separated by blank -> extend prefix
                    ext_prefix = prefix + (c,)
                    ext_confs = confs + (c_prob,)
                    ext_b, ext_nb, _ = next_beams.get(
                        ext_prefix, (-math.inf, -math.inf, ext_confs)
                    )
                    next_beams[ext_prefix] = (
                        ext_b,
                        _log_add(ext_nb, p_b + c_lp),
                        ext_confs,
                    )
                else:
                    # 2c. Distinct character -> extend prefix from both blank and non-blank
                    ext_prefix = prefix + (c,)
                    ext_confs = confs + (c_prob,)
                    ext_b, ext_nb, _ = next_beams.get(
                        ext_prefix, (-math.inf, -math.inf, ext_confs)
                    )
                    next_beams[ext_prefix] = (
                        ext_b,
                        _log_add(ext_nb, p_total + c_lp),
                        ext_confs,
                    )

        # Prune beams to top `beam_width` hypotheses by total log-probability
        ranked = sorted(
            next_beams.items(),
            key=lambda item: _log_add(item[1][0], item[1][1]),
            reverse=True,
        )
        beams = dict(ranked[:beam_width])

    best_prefix, (_, _, best_confs) = max(
        beams.items(),
        key=lambda item: _log_add(item[1][0], item[1][1]),
    )
    text = "".join(idx2char[idx] for idx in best_prefix)
    char_conf_list = list(best_confs)
    if per_syllable:
        return text, char_confs_to_syllable_confs(text, char_conf_list)
    return text, char_conf_list


if __name__ == "__main__":
    # Quick self-check verifying blank-separated duplicate characters ("o" + blank + "o" -> "oo")
    sample_idx2char = {0: "<blank>", 1: "x", 2: "o", 3: "n", 4: "g"}
    # 6 frames: 'x', 'o', '<blank>', 'o', 'n', 'g' -> "xoong"
    probs = np.array(
        [
            [0.05, 0.85, 0.03, 0.04, 0.03],
            [0.05, 0.03, 0.85, 0.04, 0.03],
            [0.85, 0.03, 0.05, 0.04, 0.03],
            [0.05, 0.03, 0.85, 0.04, 0.03],
            [0.05, 0.03, 0.04, 0.85, 0.03],
            [0.05, 0.03, 0.04, 0.03, 0.85],
        ],
        dtype=np.float32,
    )
    sample_log_probs = np.log(probs)

    g_text, g_confs = ctc_greedy_decode(sample_log_probs, sample_idx2char)
    b_text, b_confs = ctc_beam_search_decode(sample_log_probs, sample_idx2char, beam_width=5)
    assert g_text == "xoong" and b_text == "xoong"
    assert len(g_confs) == 5 and len(b_confs) == 5
    print("decode.py self-check passed:", g_text, g_confs)