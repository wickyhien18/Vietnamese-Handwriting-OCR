"""Hand-written image preprocessing utilities for Vietnamese handwriting OCR."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


def preprocess_line_image(
    img_path: str | Path,
    fixed_height: int = 32,
    min_width: int = 16,
    invert: bool = False,
) -> np.ndarray:
    """Load a handwritten line image and normalize it for CRNN input.

    Steps:
        1. Convert the image to single-channel grayscale ('L').
        2. Resize to `fixed_height` while preserving the original aspect ratio
           so character stroke proportions are not distorted.
        3. Normalize pixel intensities from [0, 255] to [-1.0, 1.0], where
           white paper background is around +1.0 and dark ink is around -1.0.

    Args:
        img_path: Path to the line image file.
        fixed_height: Target image height in pixels (default: 32).
        min_width: Minimum allowed width after resizing to avoid empty CNN outputs.
        invert: If True, invert pixel intensities (useful for white chalk on dark board).

    Returns:
        2D float32 NumPy array of shape (fixed_height, new_width) in [-1.0, 1.0].
    """
    with Image.open(img_path) as img:
        gray_img = img.convert("L")
        orig_w, orig_h = gray_img.size

        # Preserve aspect ratio when scaling height to fixed_height
        scale = fixed_height / float(orig_h)
        new_w = max(min_width, int(round(orig_w * scale)))
        resized_img = gray_img.resize((new_w, fixed_height), Image.Resampling.BILINEAR)

        arr = np.asarray(resized_img, dtype=np.float32)

    if invert:
        arr = 255.0 - arr

    # Scale [0, 255] -> [0.0, 1.0] -> [-1.0, 1.0]
    arr = (arr / 255.0 - 0.5) / 0.5
    return arr


def compute_cnn_output_length(image_width: int) -> int:
    """Compute the sequence length T produced by the baseline CRNN encoder.

    In our CRNN backbone:
        - Two MaxPool2d(2, 2) layers divide width by 4.
        - Two MaxPool2d((2, 2), (2, 1), (0, 1)) layers add +2 to width.
        - One final Conv2d(kernel_size=2, stride=1) subtracts 1 from width.
    Net formula: (image_width // 4) + 1.
    """
    return (image_width // 4) + 1


def pad_line_batch(
    arrays: Sequence[np.ndarray],
    pad_value: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad variable-width grayscale line arrays into a single 4D batch.

    Args:
        arrays: Sequence of 2D arrays with shape (H, W_i).
        pad_value: Padding constant (+1.0 matches white paper background).

    Returns:
        batch: Float32 NumPy array of shape (B, 1, H, max_W).
        input_lengths: Int64 NumPy array of shape (B,) with valid CNN frame counts.
    """
    if not arrays:
        raise ValueError("Cannot pad an empty sequence of images.")

    batch_size = len(arrays)
    height = arrays[0].shape[0]
    widths = [arr.shape[1] for arr in arrays]
    max_width = max(widths)

    batch = np.full((batch_size, 1, height, max_width), pad_value, dtype=np.float32)
    for idx, arr in enumerate(arrays):
        batch[idx, 0, :, : arr.shape[1]] = arr

    input_lengths = np.array([compute_cnn_output_length(w) for w in widths], dtype=np.int64)
    return batch, input_lengths


if __name__ == "__main__":
    # Quick self-check with synthetic white-background image containing dark strokes
    dummy = np.ones((40, 120), dtype=np.float32)
    dummy[10:30, 20:25] = -1.0
    batched, lengths = pad_line_batch([dummy, dummy[:, :80]])
    assert batched.shape == (2, 1, 40, 120)
    assert lengths.tolist() == [31, 21]
    print("preprocess.py self-check passed:", batched.shape, lengths.tolist())