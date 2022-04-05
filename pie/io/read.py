from typing import Tuple

import cv2
import numpy as np


def read_image(name: str) -> np.ndarray:
  return cv2.imread(str)


def write_image(name: str, image: np.ndarray) -> None:
  cv2.imwrite(name, image)


def read_images(
  src_name: str,
  mask_name: str,
  tgt_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  return read_image(src_name), read_image(mask_name), read_image(tgt_name)
