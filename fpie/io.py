"""Image I/O helpers for Fast Poisson Image Editing."""

import os
import warnings

import cv2
import numpy as np


def read_image(name: str) -> np.ndarray:
    """Read an image file and normalize it to three channels."""
    img = cv2.imread(name)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {name}")
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :-1]
    return img


def write_image(name: str, image: np.ndarray) -> None:
    """Write an image array to disk."""
    cv2.imwrite(name, image)


def read_images(
    src_name: str,
    mask_name: str,
    tgt_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the source, mask, and target images from disk."""
    src, tgt = read_image(src_name), read_image(tgt_name)
    if os.path.exists(mask_name):
        mask = read_image(mask_name)
    else:
        warnings.warn("No mask file found, use default setting", stacklevel=2)
        mask = np.zeros_like(src) + 255
    return src, mask, tgt
