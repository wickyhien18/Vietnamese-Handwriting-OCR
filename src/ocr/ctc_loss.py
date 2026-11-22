"""Hand-written log-space Forward CTC Loss vectorized across batch and states."""

from __future__ import annotations

import torch
import torch.nn as nn


def _safe_logsumexp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable logsumexp that avoids (-inf) - (-inf) = NaN in backward()."""
    max_val, _ = torch.max(x, dim=dim, keepdim=True)
    is_all_inf = torch.isneginf(max_val)
    safe_max = torch.where(is_all_inf, torch.zeros_like(max_val), max_val)

    exp_shifted = torch.exp(x - safe_max)
    sum_exp = torch.sum(exp_shifted, dim=dim, keepdim=True)
    safe_sum = torch.where(is_all_inf, torch.ones_like(sum_exp), sum_exp)

    out = safe_max + torch.log(safe_sum)
    return torch.where(is_all_inf, max_val, out).squeeze(dim)


def _prepare_padded_targets(
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    """Convert 1D concatenated targets into 2D padded (B, max_L) if needed."""
    if targets.ndim == 2:
        return targets
    if targets.ndim != 1:
        raise ValueError(f"Expected 1D or 2D targets tensor, got shape {targets.shape}")

    batch_size = int(target_lengths.shape[0])
    max_len = int(target_lengths.max().item()) if batch_size > 0 else 0
    padded = torch.zeros((batch_size, max_len), dtype=targets.dtype, device=targets.device)

    offset = 0
    for b in range(batch_size):
        length = int(target_lengths[b].item())
        if length > 0:
            padded[b, :length] = targets[offset : offset + length]
            offset += length
    return padded


def ctc_forward_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
    reduction: str = "none",
    zero_infinity: bool = False,
) -> torch.Tensor:
    """Compute CTC loss via log-space forward dynamic programming.

    Steps:
        1. Expand each label sequence l of length L into l' of length 2L + 1 by
           interleaving `blank` tokens: [blank, l_0, blank, l_1, ..., l_{L-1}, blank].
        2. Initialize at t = 0:
               alpha_0(0) = log_probs[0, b, blank]
               alpha_0(1) = log_probs[0, b, l'_1]   (if L > 0)
               alpha_0(s) = -inf                   (for all s >= 2)
        3. Recurrence for t = 1 .. T - 1 (vectorized across batch and state s):
               alpha_t(s) = logsumexp(
                   alpha_{t-1}(s),
                   alpha_{t-1}(s - 1),
                   alpha_{t-1}(s - 2)  [only if l'(s) != blank and l'(s) != l'(s - 2)]
               ) + log_probs[t, b, l'(s)]
        4. Termination at t_end = input_lengths[b] - 1:
               loss_b = -logsumexp(alpha_{t_end}(2L), alpha_{t_end}(2L - 1))

    Args:
        log_probs: Log-probabilities of shape (T, B, C).
        targets: Target indices of shape (B, max_L) or 1D concatenated (sum(L_b),).
        input_lengths: Valid sequence lengths of shape (B,), each <= T.
        target_lengths: Target sequence lengths of shape (B,).
        blank: Index of the CTC <blank> symbol (default: 0).
        reduction: 'none' | 'mean' | 'sum' (default: 'none').
        zero_infinity: If True, replace infinite losses with 0.0.

    Returns:
        Loss tensor of shape (B,) when reduction='none', or scalar otherwise.
    """
    if log_probs.ndim != 3:
        raise ValueError(f"Expected 3D log_probs (T, B, C), got shape {log_probs.shape}")

    max_time, batch_size, _ = log_probs.shape
    device = log_probs.device
    dtype = log_probs.dtype

    input_lengths = input_lengths.to(device=device, dtype=torch.long)
    target_lengths = target_lengths.to(device=device, dtype=torch.long)
    targets_2d = _prepare_padded_targets(targets.to(device=device, dtype=torch.long), target_lengths)

    max_target_len = int(target_lengths.max().item()) if batch_size > 0 else 0
    num_states = 2 * max_target_len + 1  # U = 2L + 1

    # 1. Build extended target sequence l' of shape (B, 2L + 1) filled with `blank`
    ext_targets = torch.full((batch_size, num_states), blank, dtype=torch.long, device=device)
    if max_target_len > 0:
        valid_pos = torch.arange(max_target_len, device=device).unsqueeze(0) < target_lengths.unsqueeze(1)
        odd_slots = torch.where(valid_pos, targets_2d[:, :max_target_len], torch.tensor(blank, device=device))
        ext_targets[:, 1::2] = odd_slots

    # Mask for skip-2 transitions: allowed iff l'(s) != blank and l'(s) != l'(s - 2)
    can_jump = torch.zeros((batch_size, num_states), dtype=torch.bool, device=device)
    if num_states > 2:
        can_jump[:, 2:] = (ext_targets[:, 2:] != blank) & (ext_targets[:, 2:] != ext_targets[:, :-2])

    neg_inf = torch.tensor(float("-inf"), dtype=dtype, device=device)

    # 2. Initialize alpha at t = 0: shape (B, U)
    emit_0 = torch.gather(log_probs[0], dim=1, index=ext_targets)
    alpha = torch.full((batch_size, num_states), float("-inf"), dtype=dtype, device=device)
    alpha[:, 0] = log_probs[0, :, blank]
    if num_states > 1:
        has_target = target_lengths > 0
        alpha[:, 1] = torch.where(has_target, emit_0[:, 1], neg_inf)

    alphas = [alpha]

    # 3. Vectorized forward recurrence across time steps t = 1 .. T - 1
    pad_1 = torch.full((batch_size, 1), float("-inf"), dtype=dtype, device=device)
    pad_2 = torch.full((batch_size, 2), float("-inf"), dtype=dtype, device=device)

    for t in range(1, max_time):
        emit_t = torch.gather(log_probs[t], dim=1, index=ext_targets)
        stay = alpha
        step_1 = torch.cat([pad_1, alpha[:, :-1]], dim=1)
        step_2_raw = torch.cat([pad_2, alpha[:, :-2]], dim=1)
        step_2 = torch.where(can_jump, step_2_raw, neg_inf)

        trans_log_prob = _safe_logsumexp(torch.stack([stay, step_1, step_2], dim=-1), dim=-1)
        alpha = trans_log_prob + emit_t
        alphas.append(alpha)

    # Stack all time steps: (T, B, U) and pick frame t_end = input_lengths[b] - 1 per sample
    alpha_all = torch.stack(alphas, dim=0)
    last_steps = (input_lengths - 1).clamp(min=0, max=max(0, max_time - 1))
    batch_idx = torch.arange(batch_size, device=device)
    alpha_end = alpha_all[last_steps, batch_idx]  # (B, U)

    # 4. Termination: -logsumexp(alpha_{T-1}(2L), alpha_{T-1}(2L - 1))
    idx_even = (2 * target_lengths).clamp(max=num_states - 1).unsqueeze(1)
    idx_odd = (2 * target_lengths - 1).clamp(min=0, max=num_states - 1).unsqueeze(1)

    end_even = torch.gather(alpha_end, dim=1, index=idx_even).squeeze(1)
    end_odd = torch.gather(alpha_end, dim=1, index=idx_odd).squeeze(1)
    end_odd = torch.where(target_lengths > 0, end_odd, neg_inf)

    total_log_prob = _safe_logsumexp(torch.stack([end_even, end_odd], dim=-1), dim=-1)

    # Invalidate samples with non-positive or out-of-bounds input lengths
    valid_input = (input_lengths > 0) & (input_lengths <= max_time)
    total_log_prob = torch.where(valid_input, total_log_prob, neg_inf)

    loss = -total_log_prob

    if zero_infinity:
        loss = torch.where(torch.isinf(loss), torch.zeros_like(loss), loss)

    if reduction == "none":
        return loss
    if reduction == "mean":
        return (loss / target_lengths.clamp(min=1).to(dtype)).mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unsupported reduction mode: {reduction}")


class HandWrittenCTCLoss(nn.Module):
    """nn.Module wrapper matching torch.nn.CTCLoss interface."""

    def __init__(self, blank: int = 0, reduction: str = "none", zero_infinity: bool = False) -> None:
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        return ctc_forward_loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=self.blank,
            reduction=self.reduction,
            zero_infinity=self.zero_infinity,
        )