"""The function the rest of the group calls (contract 2 in the README).

    from src.corrector.api import load, correct
    load("/content/drive/MyDrive/Vietnamese-Handwriting-OCR/checkpoints/corrector")   # once
    correct("nhieu ban doc da bay to")
    # {"text": "nhiều bạn đọc đã bày tỏ", "conf": [0.99, 0.87, 0.95, 0.98, 0.97, 0.93]}

method:
    "transformer"  tagger.pt                          (best, needs PyTorch)
    "viterbi"      ngram.pkl.gz + viterbi.json        (bigram + Viterbi, numpy only)
    "frequency"    ngram.pkl.gz                       (baseline)
    "identity"     nothing                            (no correction)
"""

import json
import os

from src.corrector.correctors import FrequencyCorrector, IdentityCorrector, ViterbiCorrector
from src.corrector.ngram import NgramModel
from src.corrector.viterbi import Emission

_corrector = None


def load(checkpoint_dir=None, method="transformer", device=None):
    """Load a corrector once; correct() uses it afterwards. Returns the corrector object."""
    global _corrector
    if method == "identity":
        _corrector = IdentityCorrector()
    elif method == "transformer":
        import torch  # imported here so the other methods work without PyTorch
        from src.corrector.tagger import load_checkpoint
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _corrector = load_checkpoint(os.path.join(checkpoint_dir, "tagger.pt"), device)
    elif method in ("viterbi", "frequency"):
        model = NgramModel.load(os.path.join(checkpoint_dir, "ngram.pkl.gz"))
        if method == "frequency":
            _corrector = FrequencyCorrector(model)
        else:
            with open(os.path.join(checkpoint_dir, "viterbi.json"), encoding="utf-8") as f:
                settings = json.load(f)
            _corrector = ViterbiCorrector(model, Emission(settings["p_same"]), settings["k"])
    else:
        raise ValueError(f"unknown method: {method}")
    return _corrector


def correct(text):
    """Contract: returns {"text": str, "conf": [float]} with len(conf) == len(text.split())."""
    if _corrector is None:
        raise RuntimeError("Call src.corrector.api.load(...) before correct().")
    return _corrector.correct(text)
