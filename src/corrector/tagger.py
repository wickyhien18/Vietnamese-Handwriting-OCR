"""Small transformer tagger: for each input syllable, pick one clean form among its candidates.

Input of one sentence (L syllables):
    form ids (L), key ids (L), candidate ids (L x K)       -> see tagger_data.py
Model:
    x_i  = E_form[form_i] + E_key[key_i] + E_pos[i]          (embeddings, size d_model)
    x    = N transformer blocks (self-attention + feed-forward), so every syllable sees the whole sentence
    h_i  = LayerNorm(x_i)
    score(candidate c at position i) = h_i . E_out[c] + b[c]
    softmax over the candidates of position i                -> probability of each clean form

The self-attention is written out here (no nn.MultiheadAttention), so every step can be explained.
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.corrector.correctors import correct_text
from src.corrector.ngram import sentence_words
from src.corrector.noise import add_noise
from src.corrector.tagger_data import IGNORE, PAD, UNK, TaggerVocab

DEFAULT_CONFIG = {"d_model": 256, "n_heads": 4, "n_layers": 4, "d_ff": 1024, "max_len": 64, "dropout": 0.1}


class SelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention.

    For each head: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_head)) V
    Padding positions are masked with -inf before the softmax, so nobody attends to them.
    """

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)  # Q, K and V in one matrix multiplication
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        B, L, D = x.shape
        # (B, L, 3, heads, d_head) -> (3, B, heads, L, d_head)
        q, k, v = self.qkv(x).view(B, L, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)       # (B, heads, L, L)
        scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        weights = self.dropout(scores.softmax(dim=-1))
        y = (weights @ v).transpose(1, 2).reshape(B, L, D)               # heads side by side again
        return self.out(y)


class Block(nn.Module):
    """Pre-norm transformer block: x + Attention(LN(x)), then x + FeedForward(LN(x))."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        x = x + self.dropout(self.attn(self.norm1(x), pad_mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class DiacriticTagger(nn.Module):
    def __init__(self, n_forms, n_keys, d_model=256, n_heads=4, n_layers=4, d_ff=1024,
                 max_len=64, dropout=0.1):
        super().__init__()
        self.form_emb = nn.Embedding(n_forms, d_model, padding_idx=PAD)
        self.key_emb = nn.Embedding(n_keys, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.out_emb = nn.Embedding(n_forms, d_model)  # one output vector per clean form
        self.out_bias = nn.Embedding(n_forms, 1)

    def forward(self, form_ids, key_ids, cand_ids):
        """form_ids, key_ids: (B, L). cand_ids: (B, L, K). Returns logits (B, L, K);
        empty candidate slots (PAD) get -inf."""
        B, L = form_ids.shape
        pad_mask = form_ids == PAD
        positions = torch.arange(L, device=form_ids.device)
        x = self.form_emb(form_ids) + self.key_emb(key_ids) + self.pos_emb(positions)[None]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, pad_mask)
        h = self.norm(x)                                   # (B, L, D)
        c = self.out_emb(cand_ids)                         # (B, L, K, D)
        logits = (c * h.unsqueeze(2)).sum(-1) + self.out_bias(cand_ids).squeeze(-1)
        return logits.masked_fill(cand_ids == PAD, float("-inf"))


# ---------------------------------------------------------------- data

class NoisySentences(torch.utils.data.Dataset):
    """Clean sentences; the noise is made on the fly. Every epoch gets new noise (set .epoch),
    unless fixed_noise=True (validation: the same noise every time)."""

    def __init__(self, sentences, vocab, noise_config=None, seed=0, max_len=64, fixed_noise=False):
        self.sentences = sentences
        self.vocab = vocab
        self.noise_config = noise_config
        self.seed = seed
        self.max_len = max_len
        self.fixed_noise = fixed_noise
        self.epoch = 0

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        epoch = 0 if self.fixed_noise else self.epoch
        rng = random.Random(self.seed * 1_000_003 + epoch * 100_000_007 + i)
        clean_sentence = self.sentences[i]
        noisy_sentence = add_noise(clean_sentence, rng, self.noise_config)
        clean = sentence_words(clean_sentence)[: self.max_len]
        observed = sentence_words(noisy_sentence)[: self.max_len]
        if len(observed) != len(clean):  # should never happen: noise keeps the syllable count
            observed = list(clean)
        return self.vocab.encode(observed, clean)


def collate(batch):
    """Pad a list of encoded sentences to tensors. Padding positions get candidate [UNK],
    so every row has at least one finite logit."""
    L = max(len(item[0]) for item in batch)
    K = max(len(c) for item in batch for c in item[2])
    forms, keys, cands, targets = [], [], [], []
    for form_ids, key_ids, cand_ids, target in batch:
        n = len(form_ids)
        forms.append(form_ids + [PAD] * (L - n))
        keys.append(key_ids + [PAD] * (L - n))
        targets.append(target + [IGNORE] * (L - n))
        rows = [c + [PAD] * (K - len(c)) for c in cand_ids]
        rows += [[UNK] + [PAD] * (K - 1)] * (L - n)
        cands.append(rows)
    return (torch.tensor(forms), torch.tensor(keys), torch.tensor(cands), torch.tensor(targets))


# ---------------------------------------------------------------- training

def make_scheduler(optimizer, total_steps, warmup_steps):
    """Learning rate goes up linearly for warmup_steps, then down linearly to 0."""
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_epoch(model, loader, optimizer, scheduler, device, log_every=500):
    model.train()
    total_loss, n_batches = 0.0, 0
    for step, batch in enumerate(loader):
        form, key, cand, target = (t.to(device) for t in batch)
        if (target != IGNORE).sum() == 0:
            continue
        logits = model(form, key, cand)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                               ignore_index=IGNORE)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        n_batches += 1
        if log_every and step % log_every == 0:
            print(f"  step {step:>6}  loss {loss.item():.4f}  lr {scheduler.get_last_lr()[0]:.2e}")
    return total_loss / max(1, n_batches)


@torch.no_grad()
def collect_logits(model, loader, device, k_max):
    """Logits and targets of every trained position, as numpy arrays (N, k_max) and (N,).
    Used for accuracy and for temperature scaling."""
    model.eval()
    all_logits, all_targets = [], []
    for batch in loader:
        form, key, cand, target = (t.to(device) for t in batch)
        logits = model(form, key, cand)
        keep = target != IGNORE
        z = logits[keep]
        if z.size(1) < k_max:  # pad to a fixed width so batches can be stacked
            z = F.pad(z, (0, k_max - z.size(1)), value=float("-inf"))
        all_logits.append(z.float().cpu())
        all_targets.append(target[keep].cpu())
    return torch.cat(all_logits).numpy(), torch.cat(all_targets).numpy()


# ---------------------------------------------------------------- corrector

class TransformerCorrector:
    """Follows the group contract through correct_text(): {"text": str, "conf": [float]}."""

    def __init__(self, model, vocab, temperature=1.0, device="cpu", max_len=64):
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.temperature = temperature
        self.device = device
        self.max_len = max_len

    @torch.no_grad()
    def correct_words(self, observed):
        words, conf = [], []
        for start in range(0, len(observed), self.max_len):  # long inputs: one chunk at a time
            chunk = observed[start:start + self.max_len]
            form, key, cand, _ = (t.to(self.device) for t in collate([self.vocab.encode(chunk)]))
            logits = self.model(form, key, cand)[0] / self.temperature
            probs = logits.softmax(dim=-1).cpu()
            cand = cand[0].cpu()
            for i, o in enumerate(chunk):
                if not self.vocab.candidate_ids(o):  # unknown word: keep it
                    words.append(o)
                    conf.append(1.0)
                    continue
                j = int(probs[i].argmax())
                words.append(self.vocab.forms[int(cand[i, j])])
                conf.append(float(probs[i, j]))
        return words, conf

    def correct(self, text):
        return correct_text(text, self.correct_words)


def save_checkpoint(path, model, vocab, config, temperature=1.0):
    torch.save({"config": config, "state_dict": model.state_dict(),
                "vocab": vocab.to_dict(), "temperature": temperature}, path)


def load_checkpoint(path, device="cpu"):
    """Returns a ready TransformerCorrector."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    vocab = TaggerVocab.from_dict(ckpt["vocab"])
    model = DiacriticTagger(len(vocab.forms), len(vocab.keys), **ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return TransformerCorrector(model, vocab, ckpt["temperature"], device, ckpt["config"]["max_len"])
