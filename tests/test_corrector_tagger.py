"""Tests for the transformer tagger. Skipped when PyTorch is not installed (the local .venv has
only numpy); they run in Colab, where PyTorch is available."""

import importlib.util
import os
import tempfile
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None

CORPUS = (["Tôi mua bàn ghế mới."] * 20 + ["Cửa hàng bán hàng rất đông."] * 20
          + ["Bạn tôi là bạn tốt."] * 20)
TINY = {"d_model": 32, "n_heads": 2, "n_layers": 1, "d_ff": 64, "max_len": 16, "dropout": 0.0}


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TaggerTest(unittest.TestCase):
    def setUp(self):
        import torch
        from src.corrector.ngram import NgramModel
        from src.corrector.tagger_data import TaggerVocab

        torch.manual_seed(0)
        model = NgramModel()
        model.train(CORPUS)
        self.vocab = TaggerVocab.from_ngram(model, min_count=1)

    def make_model(self):
        from src.corrector.tagger import DiacriticTagger
        return DiacriticTagger(len(self.vocab.forms), len(self.vocab.keys), **TINY)

    def test_shapes_and_padding(self):
        from src.corrector.tagger import collate
        batch = collate([self.vocab.encode(["ban", "ghe"], ["bàn", "ghế"]),
                         self.vocab.encode(["toi", "mua", "ban", "ghe", "moi"])])
        form, key, cand, target = batch
        self.assertEqual(form.shape, (2, 5))
        logits = self.make_model()(form, key, cand)
        self.assertEqual(logits.shape, cand.shape)
        self.assertTrue(bool((logits.isfinite().sum(-1) >= 1).all()))  # every row has a candidate

    def test_attention_ignores_padding(self):
        import torch
        from src.corrector.tagger import collate
        model = self.make_model().eval()
        short = collate([self.vocab.encode(["ban", "ghe"])])
        padded = collate([self.vocab.encode(["ban", "ghe"]), self.vocab.encode(["toi"] * 5)])
        with torch.no_grad():
            a = model(*short[:3])[0]
            b = model(*padded[:3])[0, :2, : a.size(1)]
        torch.testing.assert_close(a, b)

    def test_learns_context_and_contract(self):
        import torch
        from src.corrector.noise import NoiseConfig
        from src.corrector.tagger import (NoisySentences, TransformerCorrector, collate,
                                          load_checkpoint, make_scheduler, save_checkpoint,
                                          train_epoch)
        model = self.make_model()
        data = NoisySentences(CORPUS, self.vocab, NoiseConfig(p_strip_sentence=1.0), max_len=16)
        loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True, collate_fn=collate)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        scheduler = make_scheduler(optimizer, total_steps=40 * len(loader), warmup_steps=10)
        for epoch in range(40):
            data.epoch = epoch
            loss = train_epoch(model, loader, optimizer, scheduler, "cpu", log_every=0)
        self.assertLess(loss, 0.3)

        corrector = TransformerCorrector(model, self.vocab, temperature=1.5)
        out = corrector.correct('"TOI mua ban ghe", 2020.')
        self.assertEqual(out["text"], '"TÔI mua bàn ghế", 2020.')
        self.assertEqual(len(out["conf"]), len(out["text"].split()))
        self.assertEqual(corrector.correct("cua hang ban hang")["text"], "cửa hàng bán hàng")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tagger.pt")
            save_checkpoint(path, model, self.vocab, TINY, temperature=1.5)
            loaded = load_checkpoint(path)
        self.assertEqual(loaded.temperature, 1.5)
        self.assertEqual(loaded.correct("ban toi la ban tot"), corrector.correct("ban toi la ban tot"))


if __name__ == "__main__":
    unittest.main()
