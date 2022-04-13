#!/usr/bin/env python3

import os
import sys
from typing import List, Tuple

import cv2
import numpy as np


def download(links: List[Tuple[str, str]]) -> None:
  for link, filename in links:
    if not os.path.exists(filename):
      os.system(f"wget {link} -O {filename}")


def square(x: int) -> None:
  r = int((4**x)**.5 + 2)
  img = np.zeros([r, r, 3], np.uint8) + 255
  cv2.imwrite(f"square{x}.png", img)


def circle(x: int) -> None:
  r = int(((4**x) * 4 / np.pi)**.5 + 2)
  img = np.zeros([r, r, 3], np.uint8)
  img = cv2.circle(
    img, (int(r / 2), int(r / 2)), int(r / 2), (255, 255, 255), -1
  )
  cv2.imwrite(f"circle{x}.png", img)


if __name__ == "__main__":
  if sys.argv[-1] != "benchmark":
    links = [i.split() for i in open("data.txt").read().splitlines()]
    download(links)
  for i in range(6, 11):
    square(i)
    circle(i)
