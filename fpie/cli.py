import time

from fpie.args import get_args
from fpie.io import read_images, write_image
from fpie.process import BaseProcessor, EquProcessor, GridProcessor


def main() -> None:
  args = get_args("cli")

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
    print(f"Time elapsed: {dt:.4f}s")
    write_image(args.output, result)
    print(f"Successfully write image to {args.output}")
