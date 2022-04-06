import argparse
import time

from pie.io import read_images, write_image
from pie.process import Processor


def get_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-b",
    "--backend",
    type=str,
    choices=["numpy", "openmp"],
    help="backend choice",
  )
  parser.add_argument("-s", "--source", type=str, help="source image filename")
  parser.add_argument("-m", "--mask", type=str, help="mask image filename")
  parser.add_argument("-t", "--target", type=str, help="target image filename")
  parser.add_argument("-o", "--output", type=str, help="output image filename")
  parser.add_argument(
    "-h0", type=int, help="mask position (height) on source image"
  )
  parser.add_argument(
    "-w0", type=int, help="mask position (width) on source image"
  )
  parser.add_argument(
    "-h1", type=int, help="mask position (height) on target image"
  )
  parser.add_argument(
    "-w1", type=int, help="mask position (width) on target image"
  )
  parser.add_argument(
    "-n",
    type=int,
    help="how many iteration would you perfer, the more the better"
  )
  parser.add_argument(
    "-p", type=int, help="output result every P iteration", default=0
  )
  return parser.parse_args()


def main() -> None:
  args = get_args()
  proc = Processor(args.backend)
  src, mask, tgt = read_images(args.source, args.mask, args.target)
  n = proc.reset(src, mask, tgt, (args.h0, args.w0), (args.h1, args.w1))
  print(f"# of vars: {n}")
  if args.p == 0:
    args.p = args.n
  result = tgt
  t = time.time()
  for i in range(0, args.n, args.p):
    result, err = proc.step(args.p)
    print(f"Iter {i + args.p}, abs error {err}")
    write_image(f"iter{i:05d}.png", result)
  dt = time.time() - t
  print(f"Time elapsed: {dt:.2f}s")
  write_image(args.output, result)
