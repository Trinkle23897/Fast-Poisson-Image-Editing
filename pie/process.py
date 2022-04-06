from typing import Tuple

import numpy as np

import pie_core


class Processor(object):
  """PIE Processor"""

  def __init__(self, backend: str):
    super().__init__()
    self.backend = backend
    self.core = pie_core

  def mask2index(
    self, mask: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # ids = self.core.partition(mask)
    ids = np.cumsum((mask > 0).reshape(-1)).reshape(mask.shape)  # or others
    max_id = ids[-1, -1] + 1
    ids[mask == 0] = 0  # reserve id=0 for constant
    index = np.zeros((max_id, 3))
    x, y = np.nonzero(mask)
    index = ids[x, y].argsort()
    return ids, max_id, x[index], y[index]

  def reset(
    self,
    src: np.ndarray,
    mask: np.ndarray,
    tgt: np.ndarray,
    mask_on_src: Tuple[int, int],
    mask_on_tgt: Tuple[int, int],
  ) -> None:
    # check validity
    assert 0 <= mask_on_src[0] and 0 <= mask_on_src[1]
    assert mask_on_src[0] + mask.shape[0] <= src.shape[0]
    assert mask_on_src[1] + mask.shape[1] <= src.shape[1]
    assert mask_on_src[0] + mask.shape[0] <= tgt.shape[0]
    assert mask_on_src[1] + mask.shape[1] <= tgt.shape[1]

    if len(mask.shape) == 3:
      mask = mask.mean(-1)
    mask = (mask >= 128).astype(int)

    # zero-out edge
    mask[0] = 0
    mask[-1] = 0
    mask[:, 0] = 0
    mask[:, -1] = 0

    src_grad = src * 4.0
    src_grad[:-1] -= src[1:]
    src_grad[1:] -= src[-1:]
    src_grad[:, :-1] -= src[:, 1:]
    src_grad[:, 1:] -= src[:, :-1]

    ids, max_id, index_x, index_y = self.mask2index(mask)
    A = np.zeros((max_id, 4), int)
    X = np.zeros((max_id, 3), float)
    B = np.zeros((max_id, 3), float)

    X[1:] = src[index_x + mask_on_src[0], index_y + mask_on_src[1]]
    # four-way
    A[1:, 0] = ids[index_x - 1, index_y]
    A[1:, 1] = ids[index_x + 1, index_y]
    A[1:, 2] = ids[index_x, index_y - 1]
    A[1:, 3] = ids[index_x, index_y + 1]
    B[1:] = src_grad[index_x + mask_on_src[0], index_y + mask_on_src[1]]
    m = (mask[index_x - 1, index_y] == 0).astype(float).reshape(-1, 1)
    B[1:] += m * tgt[index_x + mask_on_tgt[0] - 1, index_y + mask_on_tgt[1]]
    m = (mask[index_x + 1, index_y] == 0).astype(float).reshape(-1, 1)
    B[1:] += m * tgt[index_x + mask_on_tgt[0] + 1, index_y + mask_on_tgt[1]]
    m = (mask[index_x, index_y - 1] == 0).astype(float).reshape(-1, 1)
    B[1:] += m * tgt[index_x + mask_on_tgt[0], index_y + mask_on_tgt[1] - 1]
    m = (mask[index_x, index_y + 1] == 0).astype(float).reshape(-1, 1)
    B[1:] += m * tgt[index_x + mask_on_tgt[0], index_y + mask_on_tgt[1] + 1]

    self.tgt = tgt.copy()
    self.tgt_index = (index_x + mask_on_tgt[0], index_y + mask_on_tgt[1])
    self.core.reset(max_id, A, X, B)

  def step(self, iteration: int) -> np.ndarray:
    x, err = self.core.step(iteration)
    self.tgt[self.tgt_index] = x[1:]
    return self.tgt
