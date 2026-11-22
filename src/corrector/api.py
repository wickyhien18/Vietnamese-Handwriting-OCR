"""The one entry point the rest of the group uses (contract 4, contracts/corrector_api.md).

    from src.corrector.api import correct
    correct("nhieu ban doc da bay to")
    # {"text": "nhiều bạn đọc đã bày tỏ", "conf": [0.99, 0.87, 0.95, 0.98, 0.97, 0.93]}

correct() loads the default level the first time it is called. The levels:

    "frequency"  most frequent accented form of each syllable   ngram.pkl.gz (+ viterbi.json for T)
    "bigram"     bigram hidden Markov model + Viterbi            ngram.pkl.gz + viterbi.json
    "tagger"     transformer tagger (needs PyTorch)              tagger.pt
    "identity"   no correction (for comparison)

The default level is the one with the best result on val, written by
corrector_06_final.ipynb into default_level.json next to the model files.
If that file does not exist, the default is FALLBACK_LEVEL.

Model folder: argument ckpt_dir, else the environment variable CORRECTOR_CKPT_DIR,
else the shared Drive folder on Colab (DEFAULT_CKPT_DIR). No personal path is written here.
"""

import json
import os

from src.corrector.correctors import FrequencyCorrector, IdentityCorrector, ViterbiCorrector
from src.corrector.ngram import NgramModel
from src.corrector.viterbi import Emission

LEVELS = ("frequency", "bigram", "tagger", "identity")
FALLBACK_LEVEL = "bigram"
CKPT_ENV = "CORRECTOR_CKPT_DIR"
DEFAULT_CKPT_DIR = "/content/drive/MyDrive/Vietnamese-Handwriting-OCR/checkpoints/corrector"

_cache = {}        # (level, folder, device, tagger_file) -> corrector object
_corrector = None  # set by load(); correct() uses it when present


def checkpoint_dir(ckpt_dir=None):
    """The model folder: the argument, else $CORRECTOR_CKPT_DIR, else the shared Drive folder."""
    return ckpt_dir or os.environ.get(CKPT_ENV) or DEFAULT_CKPT_DIR


def default_level(ckpt_dir=None):
    """The level chosen on val (default_level.json), or FALLBACK_LEVEL if it was not chosen yet."""
    path = os.path.join(checkpoint_dir(ckpt_dir), "default_level.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)["level"]
    return FALLBACK_LEVEL


def _read_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def build_corrector(level, ckpt_dir=None, device=None, tagger_file="tagger.pt"):
    """Create the corrector object of one level from the files in the model folder."""
    folder = checkpoint_dir(ckpt_dir)
    if level == "identity":
        return IdentityCorrector()
    if level == "tagger":
        import torch  # imported here, so the other levels work without PyTorch
        from src.corrector.tagger import load_checkpoint
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return load_checkpoint(os.path.join(folder, tagger_file), device)
    if level in ("frequency", "bigram"):
        model = NgramModel.load(os.path.join(folder, "ngram.pkl.gz"))
        settings = _read_json(os.path.join(folder, "viterbi.json"), {})
        k = settings.get("k", 8)
        if level == "frequency":
            return FrequencyCorrector(model, k, settings.get("temperature_frequency", 1.0))
        if "p_same" not in settings:
            raise FileNotFoundError(f"viterbi.json with p_same is missing in {folder}")
        return ViterbiCorrector(model, Emission(settings["p_same"]), k, settings.get("temperature", 1.0))
    raise ValueError(f"unknown level: {level!r}, choose one of {LEVELS}")


def get_corrector(level=None, ckpt_dir=None, device=None, tagger_file="tagger.pt"):
    """Return a function text -> {"text": str, "conf": [float]} for one level (None = default level).
    Each level is loaded once and then reused."""
    level = level or default_level(ckpt_dir)
    key = (level, checkpoint_dir(ckpt_dir), device, tagger_file)
    if key not in _cache:
        _cache[key] = build_corrector(level, ckpt_dir, device, tagger_file)
    return _cache[key].correct


def correct(text):
    """Contract: {"text": str, "conf": [float]}, same number of syllables, only diacritics change."""
    if not text.split():
        return {"text": "", "conf": []}
    if _corrector is not None:
        return _corrector.correct(text)
    return get_corrector()(text)


# ------------------------------------------------------------------ older interface, kept for the
# notebooks corrector_03 and corrector_04: load(dir, method=...) sets the corrector used by correct().

_OLD_NAMES = {"viterbi": "bigram", "transformer": "tagger"}


def load(checkpoint_dir=None, method="transformer", device=None, tagger_file="tagger.pt"):
    """Load one level and make correct() use it. method may also use the old names
    "viterbi" (= bigram) and "transformer" (= tagger). Returns the corrector object."""
    global _corrector
    level = _OLD_NAMES.get(method, method)
    _corrector = build_corrector(level, checkpoint_dir, device, tagger_file)
    return _corrector
