from typing import Tuple

import numpy as np

import pie_core


class Processor(object):
  """PIE Processor"""

  def __init__(self, backend: str):
    super().__init__()
    self.backend = backend
    self.core = pie_core

  def reset(
    self,
    src: np.ndarray,
    mask: np.ndarray,
    tgt: np.ndarray,
    mask_on_src: Tuple[int, int],
    mask_on_tgt: Tuple[int, int],
  ) -> None:
    assert 0 <= mask_on_src[0]
    assert mask_on_src[0] + mask.shape[0] <= src.shape[0]
    assert mask_on_tgt[0] + mask.shape[0] <= tgt.shape[0]
    assert 0 <= mask_on_src[1]
    assert mask_on_src[1] + mask.shape[1] <= src.shape[1]
    assert mask_on_src[1] + mask.shape[1] <= tgt.shape[1]
    if len(mask.shape) == 3:
      mask = mask.mean(-1)
    mask[mask >= 200] = 255
    self.core.reset(
      src.astype(np.uint8), mask.astype(np.uint8), tgt.astype(np.uint8),
      *mask_on_src, *mask_on_tgt
    )

  def step(self, iteration: int) -> np.ndarray:
    return self.core.step(iteration)
