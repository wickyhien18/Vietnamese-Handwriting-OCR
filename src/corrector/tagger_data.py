"""Turn syllables into the integer ids the transformer tagger reads and predicts. No PyTorch here.

For every input position the tagger gets
    form id  : id of the observed syllable itself      ("ban" or "bạn" or "bàn" ...)
    key id   : id of its plain version                  ("ban")
    candidates: ids of the possible clean forms         (bạn, bàn, bán, ban, bản, bạn ...)
and has to pick one candidate. These are the same candidates Viterbi uses, so both models
answer the same multiple-choice question.
"""

from src.common.text_utils import strip_diacritics
from src.corrector.ngram import START

PAD = 0
UNK = 1
IGNORE = -100  # target value for positions that are not trained (PyTorch's default ignore_index)


class TaggerVocab:
    def __init__(self, forms, candidates, k=8):
        """forms: list of syllables (index = id). candidates: {plain key: [form ids]}."""
        self.forms = forms
        self.form_id = {w: i for i, w in enumerate(forms)}
        self.candidates = candidates
        self.keys = ["<pad>", "<unk>"] + sorted(candidates)
        self.key_id = {key: i for i, key in enumerate(self.keys)}
        self.k = k

    @classmethod
    def from_ngram(cls, model, min_count=3, k=8):
        """Keep syllables seen at least min_count times. Candidates = top k forms of each key."""
        forms = ["<pad>", "<unk>"] + [
            w for w, c in model.unigram.most_common() if c >= min_count and w != START
        ]
        form_id = {w: i for i, w in enumerate(forms)}
        candidates = {}
        for key, counter in model.forms.items():
            ids = [form_id[w] for w, _ in counter.most_common(k) if w in form_id]
            if ids:
                candidates[key] = ids
        return cls(forms, candidates, k)

    def candidate_ids(self, observed):
        """Candidate ids for one observed syllable; the observed form is added if it is a known form."""
        ids = list(self.candidates.get(strip_diacritics(observed), []))
        own = self.form_id.get(observed)
        if ids and own is not None and own not in ids:
            ids.append(own)
        return ids

    def encode(self, observed, clean=None):
        """Lists of ids for one sentence.

        Returns (form_ids, key_ids, cand_ids, targets). cand_ids[i] is a list (length 1 to k+1);
        a syllable with no candidates (foreign word) gets [UNK] and target IGNORE, so the model
        never has to learn it and the corrector keeps it unchanged.
        targets[i] = position of the clean syllable inside cand_ids[i], or IGNORE if it is not there.
        """
        form_ids, key_ids, cand_ids, targets = [], [], [], []
        for i, o in enumerate(observed):
            form_ids.append(self.form_id.get(o, UNK))
            key_ids.append(self.key_id.get(strip_diacritics(o), UNK))
            cands = self.candidate_ids(o)
            if not cands:
                cand_ids.append([UNK])
                targets.append(IGNORE)
                continue
            cand_ids.append(cands)
            if clean is None:
                targets.append(IGNORE)
            else:
                target_id = self.form_id.get(clean[i])
                targets.append(cands.index(target_id) if target_id in cands else IGNORE)
        return form_ids, key_ids, cand_ids, targets

    def to_dict(self):
        return {"forms": self.forms, "candidates": self.candidates, "k": self.k}

    @classmethod
    def from_dict(cls, d):
        return cls(d["forms"], d["candidates"], d["k"])
