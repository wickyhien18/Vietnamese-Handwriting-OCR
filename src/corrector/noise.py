"""Noise generator: make training pairs (noisy sentence, clean sentence) from clean text.

The corrector learns to undo these errors. Four kinds of noise, each with its own probability:
    1. strip the whole sentence   : "người Việt" -> "nguoi Viet"
    2. strip one syllable          : "người" -> "nguoi"
    3. change the tone mark        : "người" -> "ngưới" (or remove it, or add one)
    4. change one shape mark       : "người" -> "nguời", "đi" -> "di", "ăn" -> "ân"

The noise never adds or removes a syllable, so the noisy sentence and the clean sentence always
have the same number of space-separated tokens. Token i of the input matches token i of the output.
"""

import random
from dataclasses import dataclass

from src.common.text_utils import strip_diacritics
from src.corrector.syllable import (
    SHAPE_OPTIONS,
    TONES,
    compose,
    decompose,
    normalize_tone_placement,
    split_token,
    tone_index,
    tone_position,
)


@dataclass
class NoiseConfig:
    p_strip_sentence: float = 0.3   # chance that the whole sentence loses all diacritics
    p_strip_syllable: float = 0.10  # chance per syllable to lose all diacritics
    p_change_tone: float = 0.10     # chance per syllable to get a different tone
    p_change_shape: float = 0.05    # chance per syllable to get a different shape mark


def change_tone(chars, rng):
    """Replace the tone with a different one. "No tone" counts as one of the six choices."""
    current_pos = tone_index(chars)
    pos = current_pos if current_pos is not None else tone_position(chars)
    if pos is None:  # no vowel, nothing to do
        return
    current_tone = chars[pos][2]
    choices = [t for t in [None] + TONES if t != current_tone]
    chars[pos][2] = rng.choice(choices)


def change_shape(chars, rng):
    """Pick one letter that can carry a shape mark and give it a different one (â -> a or ă, ...)."""
    candidates = [i for i, (base, _, _) in enumerate(chars) if base.lower() in SHAPE_OPTIONS]
    if not candidates:
        return
    i = rng.choice(candidates)
    base, shape, _ = chars[i]
    choices = [s for s in SHAPE_OPTIONS[base.lower()] if s != shape]
    chars[i][1] = rng.choice(choices)


def noise_syllable(word, rng, config):
    """Apply syllable-level noise to one word without punctuation."""
    if rng.random() < config.p_strip_syllable:
        return strip_diacritics(word)
    chars = decompose(word)
    if rng.random() < config.p_change_tone:
        change_tone(chars, rng)
    if rng.random() < config.p_change_shape:
        change_shape(chars, rng)
    # A changed shape mark can move where the tone belongs, so put it back in the standard place.
    return normalize_tone_placement(compose(chars))


def add_noise(sentence, rng, config=None):
    """Return a noisy copy of a clean sentence. Same number of tokens as the input."""
    config = config or NoiseConfig()
    if rng.random() < config.p_strip_sentence:
        return strip_diacritics(sentence)
    noisy_tokens = []
    for token in sentence.split():
        prefix, core, suffix = split_token(token)
        if core:
            core = noise_syllable(core, rng, config)
        noisy_tokens.append(prefix + core + suffix)
    return " ".join(noisy_tokens)


def make_pairs(sentences, seed=0, config=None):
    """Yield (noisy, clean) pairs. The same seed always gives the same noise."""
    rng = random.Random(seed)
    for clean in sentences:
        yield add_noise(clean, rng, config), clean
