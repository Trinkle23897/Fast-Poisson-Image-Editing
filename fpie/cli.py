import argparse
import time

import fpie
from fpie.io import read_images, write_image
from fpie.process import (
  ALL_BACKEND,
  CPU_COUNT,
  DEFAULT_BACKEND,
  BaseProcessor,
  EquProcessor,
  GridProcessor,
)


def get_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-v", "--version", action="store_true", help="show the version and exit"
  )
  parser.add_argument(
    "--check-backend", action="store_true", help="print all available backends"
  )
  parser.add_argument(
    "-b",
    "--backend",
    type=str,
    choices=ALL_BACKEND,
    default=DEFAULT_BACKEND,
    help="backend choice",
  )
  parser.add_argument(
    "-c",
    "--cpu",
    type=int,
    default=CPU_COUNT,
    help="number of CPU used",
  )
  parser.add_argument(
    "-z",
    "--block-size",
    type=int,
    default=1024,
    help="cuda block size (only for equ solver)",
  )
  parser.add_argument(
    "--method",
    type=str,
    choices=["equ", "grid"],
    default="equ",
    help="how to parallelize computation",
  )
  parser.add_argument("-s", "--source", type=str, help="source image filename")
  parser.add_argument(
    "-m",
    "--mask",
    type=str,
    help="mask image filename (default is to use the whole source image)",
    default="",
  )
  parser.add_argument("-t", "--target", type=str, help="target image filename")
  parser.add_argument("-o", "--output", type=str, help="output image filename")
  parser.add_argument(
    "-h0", type=int, help="mask position (height) on source image", default=0
  )
  parser.add_argument(
    "-w0", type=int, help="mask position (width) on source image", default=0
  )
  parser.add_argument(
    "-h1", type=int, help="mask position (height) on target image", default=0
  )
  parser.add_argument(
    "-w1", type=int, help="mask position (width) on target image", default=0
  )
  parser.add_argument(
    "-g",
    "--gradient",
    type=str,
    choices=["max", "src", "avg"],
    default="max",
    help="how to calculate gradient for PIE",
  )
  parser.add_argument(
    "-n",
    type=int,
    help="how many iteration would you perfer, the more the better",
    default=5000,
  )
  parser.add_argument(
    "-p", type=int, help="output result every P iteration", default=0
  )
  parser.add_argument(
    "--mpi-sync-interval",
    type=int,
    help="MPI sync iteration interval",
    default=100,
  )
  parser.add_argument(
    "--grid-x", type=int, help="x axis stride for grid solver", default=16
  )
  parser.add_argument(
    "--grid-y", type=int, help="y axis stride for grid solver", default=16
  )
  return parser.parse_args()


def main() -> None:
  args = get_args()
  if args.version:
    print(fpie.__version__)
    exit(0)
  if args.check_backend:
    print(ALL_BACKEND)
    exit(0)

  proc: BaseProcessor
  if args.method == "equ":
    proc = EquProcessor(
      args.gradient,
      args.backend,
      args.cpu,
      args.mpi_sync_interval,
      args.block_size,
    )
  else:
    proc = GridProcessor(
      args.gradient,
      args.backend,
      args.cpu,
      args.mpi_sync_interval,
      args.block_size,
      args.grid_x,
      args.grid_y,
    )

  if proc.root:
    print(
      f"Successfully initialize PIE {args.method} solver "
      f"with {args.backend} backend"
    )
    src, mask, tgt = read_images(args.source, args.mask, args.target)
    n = proc.reset(src, mask, tgt, (args.h0, args.w0), (args.h1, args.w1))
    print(f"# of vars: {n}")
  proc.sync()

  if proc.root:
    result = tgt
    t = time.time()
  if args.p == 0:
    args.p = args.n

  for i in range(0, args.n, args.p):
    if proc.root:
      result, err = proc.step(args.p)  # type: ignore
      print(f"Iter {i + args.p}, abs error {err}")
      if i + args.p < args.n:
        write_image(f"iter{i + args.p:05d}.png", result)
    else:
      proc.step(args.p)

  if proc.root:
    dt = time.time() - t
    print(f"Time elapsed: {dt:.2f}s")
    write_image(args.output, result)
    print(f"Successfully write image to {args.output}")
