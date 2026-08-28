"""
The EMNIST 128x128 -> 28x28 conversion.

Reference
---------
Cohen, Afshar, Tapson and van Schaik, *EMNIST: an extension of MNIST to
handwritten letters*, 2017 (arXiv:1702.05373), **section II-A "Conversion
Process"** and the caption of its Figure 1.  The implementation below follows
that text step by step; the paper's own words for each step are quoted at the
step that implements it.

    "The conversion process transforms the 128 x 128 pixel binary images found
     in the NIST dataset to 28 x 28 pixel images with an 8-bit gray-scale
     resolution that match the characteristics of the digits in the MNIST
     dataset."

The five steps, in the paper's order:

1. **Gaussian blur.**  "each digit is loaded individually and blurred using a
   Gaussian filter"; the caption fixes the width: "A Gaussian filter with
   sigma = 1 is applied to the image to soften the edges".  The body text gives
   no sigma, so :data:`SIGMA` comes from the caption.
2. **Bounding box.**  "A bounding box is fitted to the character in the image
   and extracted."  The box is fitted on the *blurred* image, which is the
   order the paper states.
3. **Centre in a square.**  "the extracted region of interest is centered in a
   square frame with lengths equal to the largest dimension, with the aspect
   ratio of the extracted region of interest preserved."
4. **Pad by 2 pixels.**  "This square frame is then padded with an empty 2
   pixel border to prevent the digits and characters from touching the border."
5. **Bi-cubic down-sample and rescale.**  "Finally, the image is down-sampled
   to 28 x 28 pixels using a bi-cubic interpolation algorithm, resulting in a
   spectrum of intensities which are then scaled to the 8-bit range."

Two documented deviations from "MNIST convention", both of them the paper's own
-------------------------------------------------------------------------------
* **No 20x20 box and no centre of mass.**  MNIST fits the digit into a 20x20
  box and centres it in the 28x28 field by its centre of mass.  EMNIST does
  neither, explicitly: "Whereas the original MNIST conversion technique
  down-sampled the digits to either a 20x20 pixel or a 32x32 pixel frame before
  placing it into the final 28 x 28 pixel frame, the technique used in this
  paper attempts to make use of the maximum amount of space available."  The
  centring is purely geometric - the phrase "centre of mass" does not occur in
  the paper at all - and the character therefore fills the frame.  This module
  reproduces **EMNIST**, not MNIST; the difference is a deliberate part of the
  benchmark this study compares itself to.
* **Where the 2-pixel border is applied.**  The paper places the padding before
  the down-sampling (both in the body text and in the caption), which puts the
  border at *source* resolution, where two pixels are a fraction of an output
  pixel.  The caption's stated purpose - "matching the clear border around all
  the digits in the MNIST dataset" - would instead imply a border visible in
  the 28x28 output.  The text does not resolve this.  :data:`BORDER_SPACE`
  selects the literal reading (``"source"``, the default); ``"target"``
  reserves the border in the output frame instead, i.e. resizes to
  ``28 - 2*border`` and pads that.

Polarity
--------
The paper says nothing about polarity: it states only that the result is 8-bit
gray-scale scaled to [0, 255].  This project stores NIST in its native polarity
(dark ink on a light background) and inverts its MNIST proxy set *to* that
polarity, so the conversion keeps the native polarity as well and the proxy set
stays comparable.  Internally the ink is worked with as the high value, because
a bounding box has to be fitted to the ink and not to the paper.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

#: Width of the Gaussian filter, from the caption of the paper's Figure 1.
SIGMA = 1.0

#: Empty border added around the square frame, in pixels.
BORDER = 2

#: Where the border of :data:`BORDER` pixels is reserved; see the module
#: docstring.  ``"source"`` is the paper's literal ordering.
BORDER_SPACE = "source"

#: Edge length of the produced images.
TARGET_SIZE = 28

#: Ink threshold of the bounding box, on the blurred ink image.  The sources are
#: bilevel scans, so any non-zero ink belongs to the character.
INK_THRESHOLD = 0


def _to_ink(array: np.ndarray) -> np.ndarray:
    """The image with the ink as the high value (255) and the paper as 0."""
    return (255 - array.astype(np.int16)).clip(0, 255).astype(np.uint8)


def bounding_box(ink: np.ndarray, threshold: int = INK_THRESHOLD) -> Optional[Tuple[int, int, int, int]]:
    """
    Tight box around the ink, as ``(top, left, bottom, sentinel)`` bounds.

    Args:
        ink: Ink image, high values are ink.
        threshold: Values greater than this count as ink.

    Returns:
        ``(row0, row1, col0, col1)`` with the ends exclusive, or ``None`` when
        the image carries no ink at all.
    """
    mask = ink > threshold
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def convert_image(
    array: np.ndarray,
    size: int = TARGET_SIZE,
    sigma: float = SIGMA,
    border: int = BORDER,
    border_space: str = BORDER_SPACE,
) -> np.ndarray:
    """
    Convert one 128x128 bilevel NIST scan into a ``size x size`` gray-scale image.

    Args:
        array: Source image as ``uint8`` ``[H, W]``, in NIST polarity (dark ink
            on a light background).
        size: Edge length of the result (28, the EMNIST resolution).
        sigma: Width of the Gaussian filter (step 1).
        border: Empty border in pixels (step 4).
        border_space: ``"source"`` reserves the border before the down-sampling,
            as the paper's ordering states; ``"target"`` reserves it in the
            output frame.  See the module docstring.

    Returns:
        ``uint8`` ``[size, size]`` image in the same polarity as the input.
    """
    if border_space not in ("source", "target"):
        raise ValueError(
            f"Unknown border_space {border_space!r}; expected 'source' or 'target'"
        )

    ink = _to_ink(np.asarray(array, dtype=np.uint8))

    # (1) blur, so that the bilevel edges are softened before anything is measured
    blurred = np.asarray(
        Image.fromarray(ink, mode="L").filter(ImageFilter.GaussianBlur(radius=float(sigma))),
        dtype=np.uint8,
    )

    # (2) fit a bounding box to the character and extract it
    box = bounding_box(blurred)
    if box is None:
        # A blank scan has no region of interest; an empty frame is the only
        # honest answer and the caller keeps the row so the indices stay aligned.
        return np.full((size, size), 255, dtype=np.uint8)
    row0, row1, col0, col1 = box
    region = blurred[row0:row1, col0:col1]

    # (3) centre the region in a square whose side is its largest dimension,
    #     with the aspect ratio preserved
    side = max(region.shape)
    square = np.zeros((side, side), dtype=np.uint8)
    top = (side - region.shape[0]) // 2
    left = (side - region.shape[1]) // 2
    square[top : top + region.shape[0], left : left + region.shape[1]] = region

    # (4)+(5) pad by an empty border and down-sample bi-cubically to size x size
    if border_space == "source":
        padded = np.zeros((side + 2 * border, side + 2 * border), dtype=np.uint8)
        padded[border : border + side, border : border + side] = square
        resized = Image.fromarray(padded, mode="L").resize((size, size), Image.BICUBIC)
        out = np.asarray(resized, dtype=np.float32)
    else:
        inner = max(1, size - 2 * border)
        resized = Image.fromarray(square, mode="L").resize((inner, inner), Image.BICUBIC)
        out = np.zeros((size, size), dtype=np.float32)
        offset = (size - inner) // 2
        out[offset : offset + inner, offset : offset + inner] = np.asarray(
            resized, dtype=np.float32
        )

    # "a spectrum of intensities which are then scaled to the 8-bit range"
    out = np.clip(out, 0.0, 255.0)
    peak = float(out.max())
    if peak > 0:
        out = out * (255.0 / peak)
    ink_out = np.rint(out).clip(0, 255).astype(np.uint8)

    # back to the project's storage polarity (dark ink on a light background)
    return (255 - ink_out.astype(np.int16)).clip(0, 255).astype(np.uint8)


def conversion_info(
    size: int = TARGET_SIZE,
    sigma: float = SIGMA,
    border: int = BORDER,
    border_space: str = BORDER_SPACE,
) -> dict:
    """Description of the applied conversion, for the cache index and provenance."""
    return {
        "name": "emnist",
        "reference": (
            "Cohen, Afshar, Tapson, van Schaik, EMNIST: an extension of MNIST "
            "to handwritten letters, 2017 (arXiv:1702.05373), section II-A"
        ),
        "size": int(size),
        "sigma": float(sigma),
        "border": int(border),
        "border_space": str(border_space),
        "interpolation": "bicubic",
        "polarity": "dark ink on a light background (NIST native)",
        "centring": "geometric, square of the largest ROI dimension (not MNIST centre of mass)",
    }
