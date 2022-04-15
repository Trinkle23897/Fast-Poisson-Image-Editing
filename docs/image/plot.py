#!/usr/bin/env python3

import math
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns

data = {}


def reset_data(new_col=None):
  global data
  data = {
    "Problem Size": [],
    "Time per Op (ns)": [],
    "Problem Type": [],
    "Method": [],
    "Backend": [],
  }
  if new_col:
    data[new_col] = []


def parse_table(raw: str,
                ttype: str,
                new_col: str = None) -> Tuple[np.ndarray, int, int]:
  global data
  raw = raw.replace("# of vars", "N").strip().splitlines()
  head, raw = raw[0], raw[2:]
  head = head[1:-1].split("|")
  if ttype == "benchmark":
    method = head[0].strip()
    prob_type = head[1].strip().capitalize()[:-1]
  else:
    method = raw[-1][1:-1].split("|")[0].strip()
    prob_type = head[0].strip().capitalize()
    head = [float(i.strip()) for i in head[1:]]
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
  r = []
  del result["N"]
  t = []
  for k, v in result.items():
    v = v / base / 5000 * 1e9
    t += v.tolist()
    for n, i, h in zip(base, v, head):
      data["Problem Size"].append(n)
      data["Time per Op (ns)"].append(i)
      data["Backend"].append(k)
      data["Method"].append(method)
      data["Problem Type"].append(prob_type)
      if new_col:
        data[new_col].append(h)
        r.append(h)
  if new_col:
    return np.array(r, int), min(t), max(t)
  return base.astype(int), min(t), max(t)


def benchmark():
  global data
  reset_data()
  sep = "<!--benchmark-->"
  raw = open("../benchmark/README.md").read().split(sep)
  base = []
  for i in [1, 2, 4, 5]:
    base.append(parse_table(raw[i], "benchmark"))
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


def ablation(backend, new_col):
  global data
  reset_data(new_col)
  sep = f"<!--{backend}-->"
  raw = open("../benchmark/README.md").read().split(sep)
  base = []
  if backend != "cuda":
    for i in [1, 2, 3, 4]:
      base.append(parse_table(raw[i], backend, new_col))
    # tmin = min([i[1] for i in base])
    # tmax = max([i[2] for i in base])
    # print(tmin, tmax)
    tmin = 0.35
    tmax = 4.4
  else:
    for i in [1, 2]:
      base.append(parse_table(raw[i], backend, new_col))
    tmin = min([i[1] for i in base])
    tmax = max([i[2] for i in base])
  data = pd.DataFrame(data)
  print(data)
  g = sns.FacetGrid(
    data,
    col="Problem Type",
    hue="Method",
    height=3,
    aspect=1.6,
    sharex=False,
    sharey=False,
  )
  g.map(sns.lineplot, new_col, "Time per Op (ns)", marker="o")

  g.set_axis_labels(new_col, "Time per Op (ns)")
  g.add_legend(bbox_to_anchor=(0.52, 1.02), ncol=7)
  axes = g.axes.flatten()
  for ax, b in zip(axes, base):
    b = b[0]
    if backend != "cuda":
      y = np.logspace(math.log10(tmin), math.log10(tmax), 6)
      y = (y * 100).astype(int) / 100.
      ax.set(xscale="log", yscale="log")
    else:
      y = np.linspace(tmin, tmax, 6)
      y = (y * 1000).astype(int) / 1000.
      ax.set(xscale="log")
    ax.set(xticks=b, yticks=y)
    ax.set(xticklabels=b, yticklabels=y)
    title = ax.get_title()
    title = title.split(" = ")[-1]
    ax.set_title(title)
  g.savefig(f"{backend}.png")


if __name__ == '__main__':
  benchmark()
  ablation("openmp", "# of Threads")
  ablation("mpi", "# of Workers")
  ablation("cuda", "# of Threads per Block")
