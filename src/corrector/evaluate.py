"""Syllable accuracy of a corrector on (noisy, clean) sentence pairs.

Numbers reported (each as correct / total):
    all          every syllable
    ambiguous    syllables whose plain form has 2 or more accented forms in the corpus ("ban": bạn, bàn, bán...)
    noisy        syllables the noise changed: did the corrector repair them?
    broken       syllables that were already right but the corrector changed them (lower is better)
"""

from src.common.text_utils import strip_diacritics
from src.corrector.syllable import split_token


def syllables(sentence):
    return [split_token(t)[1] for t in sentence.split() if split_token(t)[1]]


def evaluate(corrector, pairs, model):
    counts = {
        "all": [0, 0], "ambiguous": [0, 0], "noisy": [0, 0], "already_right": [0, 0],
    }
    for noisy, clean in pairs:
        output = corrector.correct(noisy)["text"]
        for o, c, out in zip(syllables(noisy), syllables(clean), syllables(output)):
            right = out == c
            counts["all"][0] += right
            counts["all"][1] += 1
            forms = model.forms.get(strip_diacritics(c.lower()))
            if forms and len(forms) >= 2:
                counts["ambiguous"][0] += right
                counts["ambiguous"][1] += 1
            group = "already_right" if o == c else "noisy"
            counts[group][0] += right
            counts[group][1] += 1

    result = {name: (ok / total if total else 0.0) for name, (ok, total) in counts.items()}
    result["broken"] = 1 - result["already_right"]
    result["n_syllables"] = counts["all"][1]
    return result


def print_table(rows):
    """rows: {method name: result of evaluate()}"""
    header = f"{'method':<12}{'all':>8}{'ambiguous':>11}{'noisy fixed':>13}{'broken':>9}"
    print(header)
    print("-" * len(header))
    for name, r in rows.items():
        print(f"{name:<12}{r['all']:>8.2%}{r['ambiguous']:>11.2%}{r['noisy']:>13.2%}{r['broken']:>9.2%}")
