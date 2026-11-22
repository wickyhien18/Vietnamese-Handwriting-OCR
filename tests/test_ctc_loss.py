"""Unit tests verifying HandWrittenCTCLoss against torch.nn.CTCLoss in float64."""

import itertools
import unittest

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from src.ocr.ctc_loss import HandWrittenCTCLoss, ctc_forward_loss
    from src.ocr.preprocess import compute_cnn_output_length
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is not installed in this environment")
class TestHandWrittenCTCLoss(unittest.TestCase):
    TOLERANCE = 1e-12

    def test_random_batch_with_cnn_widths_matches_pytorch_float64(self) -> None:
        torch.manual_seed(42)
        num_classes = 25
        # Simulated image widths converted via model's exact formula (w // 4) + 1
        widths = [64, 96, 128, 160, 200, 240]
        input_lengths = torch.tensor([compute_cnn_output_length(w) for w in widths], dtype=torch.long)
        target_lengths = torch.tensor([3, 6, 8, 12, 15, 19], dtype=torch.long)

        max_t = int(input_lengths.max().item())
        batch_size = len(widths)
        max_l = int(target_lengths.max().item())

        logits = torch.randn((max_t, batch_size, num_classes), dtype=torch.float64)
        log_probs = F.log_softmax(logits, dim=-1)

        targets = torch.randint(1, num_classes, (batch_size, max_l), dtype=torch.long)
        for b, length in enumerate(target_lengths.tolist()):
            targets[b, length:] = 0

        ref_loss_fn = nn.CTCLoss(blank=0, reduction="none", zero_infinity=False)
        our_loss_fn = HandWrittenCTCLoss(blank=0, reduction="none", zero_infinity=False)

        ref_losses = ref_loss_fn(log_probs, targets, input_lengths, target_lengths)
        our_losses = our_loss_fn(log_probs, targets, input_lengths, target_lengths)

        self.assertTrue(
            torch.allclose(our_losses, ref_losses, atol=self.TOLERANCE, rtol=self.TOLERANCE),
            msg=f"Max abs diff: {(our_losses - ref_losses).abs().max().item()}",
        )

    def test_repeated_characters_and_impossible_alignments(self) -> None:
        torch.manual_seed(7)
        num_classes = 6
        # Target [1, 1, 2, 2, 2] has L=5 and 3 adjacent duplicate pairs -> needs at least T = 5 + 3 = 8 frames
        targets = torch.tensor(
            [
                [1, 1, 2, 2, 2],  # T=8 -> valid (exact minimum length with blanks)
                [1, 1, 2, 2, 2],  # T=7 -> impossible alignment (too short for required blanks -> +inf)
                [1, 2, 3, 0, 0],  # L=3, T=3 -> valid (no repeated characters)
            ],
            dtype=torch.long,
        )
        input_lengths = torch.tensor([8, 7, 3], dtype=torch.long)
        target_lengths = torch.tensor([5, 5, 3], dtype=torch.long)

        logits = torch.randn((8, 3, num_classes), dtype=torch.float64)
        log_probs = F.log_softmax(logits, dim=-1)

        # 1. zero_infinity=False -> sample index 1 must be +inf
        ref_raw = nn.CTCLoss(blank=0, reduction="none", zero_infinity=False)(
            log_probs, targets, input_lengths, target_lengths
        )
        our_raw = ctc_forward_loss(
            log_probs, targets, input_lengths, target_lengths, zero_infinity=False
        )
        self.assertTrue(torch.isinf(our_raw[1]) and our_raw[1] > 0)
        self.assertTrue(torch.allclose(our_raw[[0, 2]], ref_raw[[0, 2]], atol=self.TOLERANCE, rtol=self.TOLERANCE))

        # 2. zero_infinity=True -> sample index 1 must become 0.0
        ref_zero = nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)(
            log_probs, targets, input_lengths, target_lengths
        )
        our_zero = ctc_forward_loss(
            log_probs, targets, input_lengths, target_lengths, zero_infinity=True
        )
        self.assertEqual(our_zero[1].item(), 0.0)
        self.assertTrue(torch.allclose(our_zero, ref_zero, atol=self.TOLERANCE, rtol=self.TOLERANCE))

    def test_autograd_gradients_match_pytorch(self) -> None:
        torch.manual_seed(99)
        logits_ref = torch.randn((15, 4, 10), dtype=torch.float64, requires_grad=True)
        logits_our = logits_ref.detach().clone().requires_grad_(True)

        targets = torch.tensor([[1, 2, 2, 3], [4, 5, 0, 0], [1, 1, 1, 0], [2, 3, 4, 5]], dtype=torch.long)
        input_lengths = torch.tensor([15, 12, 10, 14], dtype=torch.long)
        target_lengths = torch.tensor([4, 2, 3, 4], dtype=torch.long)

        loss_ref = nn.CTCLoss(blank=0, reduction="mean")(
            F.log_softmax(logits_ref, dim=-1), targets, input_lengths, target_lengths
        )
        loss_our = ctc_forward_loss(
            F.log_softmax(logits_our, dim=-1), targets, input_lengths, target_lengths, reduction="mean"
        )

        loss_ref.backward()
        loss_our.backward()

        self.assertTrue(
            torch.allclose(logits_our.grad, logits_ref.grad, atol=self.TOLERANCE, rtol=self.TOLERANCE),
            msg=f"Max grad diff: {(logits_our.grad - logits_ref.grad).abs().max().item()}",
        )

    @staticmethod
    def _ctc_collapse(path: tuple[int, ...], blank: int = 0) -> tuple[int, ...]:
        """Collapse a frame-level path by merging adjacent duplicates and removing blanks."""
        collapsed: list[int] = []
        prev: int | None = None
        for token in path:
            if token != prev and token != blank:
                collapsed.append(token)
            prev = token
        return tuple(collapsed)

    def test_exhaustive_enumeration_matches_forward_dp(self) -> None:
        """Brute-force sum over all C^T paths must match log-space forward DP (Part B)."""
        torch.manual_seed(2026)
        num_classes = 4  # 0: <blank>, 1: 'a', 2: 'b', 3: 'c'
        time_steps = 5   # 4^5 = 1024 total paths

        logits = torch.randn((time_steps, 1, num_classes), dtype=torch.float64)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs[:, 0, :])  # (T, C)

        # Compute exact probability of every collapsed label via exhaustive enumeration
        exact_label_probs: dict[tuple[int, ...], torch.Tensor] = {}
        for path in itertools.product(range(num_classes), repeat=time_steps):
            path_prob = torch.tensor(1.0, dtype=torch.float64)
            for t, token in enumerate(path):
                path_prob = path_prob * probs[t, token]
            label = self._ctc_collapse(path, blank=0)
            exact_label_probs[label] = exact_label_probs.get(
                label, torch.tensor(0.0, dtype=torch.float64)
            ) + path_prob

        # Verify against ctc_forward_loss on both standard label (1, 2) and repeated label (1, 1)
        for label_tuple in [(1, 2), (1, 1), (2,)]:
            targets = torch.tensor([list(label_tuple)], dtype=torch.long)
            input_lengths = torch.tensor([time_steps], dtype=torch.long)
            target_lengths = torch.tensor([len(label_tuple)], dtype=torch.long)

            dp_loss = ctc_forward_loss(
                log_probs, targets, input_lengths, target_lengths, reduction="none"
            )[0]
            brute_force_loss = -torch.log(exact_label_probs[label_tuple])

            self.assertTrue(
                torch.allclose(dp_loss, brute_force_loss, atol=self.TOLERANCE, rtol=self.TOLERANCE),
                msg=f"Label {label_tuple}: DP={dp_loss.item()}, BruteForce={brute_force_loss.item()}",
            )


if __name__ == "__main__":
    unittest.main()