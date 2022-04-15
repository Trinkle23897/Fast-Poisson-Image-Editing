#!/usr/bin/env python3

import math
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns

data = {
  "Problem Size": [],
  "Time per Op (ns)": [],
  "Problem Type": [],
  "Method": [],
  "Backend": [],
}


def parse_benchmark_table(raw: str) -> Tuple[np.ndarray, int, int]:
  global data
  raw = raw.replace("# of vars", "N").strip().splitlines()
  head, raw = raw[0], raw[2:]
  head = head[1:-1].split("|")
  method = head[0].strip()
  prob_type = head[1].strip().capitalize()[:-1]
  result = {}
  for i in raw:
    i = i[1:-1].split("|")
    try:
      a = [float(x.strip()) for x in i[1:]]
    except ValueError:
      # xxxs
      a = [float(x.strip()[:-1]) for x in i[1:]]
    result[i[0].strip()] = np.array(a)
  base = result["N"]
  del result["N"]
  t = []
  for k, v in result.items():
    v = v / base / 5000 * 1e9
    t += v.tolist()
    for n, i in zip(base, v):
      data["Problem Size"].append(n)
      data["Time per Op (ns)"].append(i)
      data["Backend"].append(k)
      data["Method"].append(method)
      data["Problem Type"].append(prob_type)
  return base.astype(int), min(t), max(t)


def main() -> None:
  global data
  sep = "<!--benchmark-->"
  raw = open("../benchmark/README.md").read().split(sep)
  base = []
  for i in [1, 2, 4, 5]:
    base.append(parse_benchmark_table(raw[i]))
  data = pd.DataFrame(data)
  print(data)
  g = sns.FacetGrid(
    data,
    row="Method",
    col="Problem Type",
    hue="Backend",
    height=3,
    aspect=1.6,
    sharex=False,
    sharey=False,
  )
  g.map(sns.lineplot, "Problem Size", "Time per Op (ns)", marker="o")

  g.set_axis_labels("Problem Size", "Time per Op (ns)")
  g.add_legend(bbox_to_anchor=(0.52, 1.02), ncol=7)
  axes = g.axes.flatten()
  for ax, b in zip(axes, base):
    b, tmin, tmax = b
    y = np.logspace(math.log10(tmin), math.log10(tmax), 6)
    y = (y * 100).astype(int) / 100.
    ax.set(xscale="log", yscale="log")
    ax.set(xticks=b, yticks=y)
    ax.set(xticklabels=b, yticklabels=y)
    title = ax.get_title()
    method, pt = title.split(" | ")
    method = method.split(" = ")[-1]
    pt = pt.split(" = ")[-1]
    title = f"{method}, {pt}"
    ax.set_title(title)
  g.savefig("benchmark.png")


if __name__ == '__main__':
  main()
