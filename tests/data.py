#!/usr/bin/env python3

import os
from typing import List, Tuple


def download(links: List[Tuple[str, str]]) -> None:
  for link, filename in links:
    if not os.path.exists(filename):
      os.system(f"wget {link} -O {filename}")


if __name__ == "__main__":
  links = [i.split() for i in open("data.txt").read().splitlines()]
  download(links)
