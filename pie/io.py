import os
import warnings
from typing import Tuple

import cv2
import numpy as np


def read_image(name: str) -> np.ndarray:
  img = cv2.imread(name)
  if len(img.shape) == 2:
    img = np.stack([img, img, img], axis=-1)
  elif len(img.shape) == 4:
    img = img[..., :-1]
  return img


def write_image(name: str, image: np.ndarray) -> None:
  cv2.imwrite(name, image)


def read_images(
  src_name: str,
  mask_name: str,
  tgt_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  src, tgt = read_image(src_name), read_image(tgt_name)
  if os.path.exists(mask_name):
    mask = read_image(mask_name)
  else:
    warnings.warn("No mask file found, use default setting")
    mask = np.zeros_like(src) + 255
  return src, mask, tgt
