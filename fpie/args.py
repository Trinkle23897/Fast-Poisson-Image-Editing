import argparse
import os

import fpie
from fpie.process import ALL_BACKEND, CPU_COUNT, DEFAULT_BACKEND


def get_args(gen_type: str) -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-v", "--version", action="store_true", help="show the version and exit"
  )
  parser.add_argument(
    "--check-backend", action="store_true", help="print all available backends"
  )
  if gen_type == "gui" and "mpi" in ALL_BACKEND:
    # gui doesn't support MPI backend
    ALL_BACKEND.remove("mpi")
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
  if gen_type == "cli":
    parser.add_argument(
      "-m",
      "--mask",
      type=str,
      help="mask image filename (default is to use the whole source image)",
      default="",
    )
  parser.add_argument("-t", "--target", type=str, help="target image filename")
  parser.add_argument("-o", "--output", type=str, help="output image filename")
  if gen_type == "cli":
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
  if gen_type == "cli":
    parser.add_argument(
      "-p", type=int, help="output result every P iteration", default=0
    )
  if "mpi" in ALL_BACKEND:
    parser.add_argument(
      "--mpi-sync-interval",
      type=int,
      help="MPI sync iteration interval",
      default=100,
    )
  parser.add_argument(
    "--grid-x", type=int, help="x axis stride for grid solver", default=8
  )
  parser.add_argument(
    "--grid-y", type=int, help="y axis stride for grid solver", default=8
  )

  args = parser.parse_args()
  if args.version:
    print(fpie.__version__)
    exit(0)
  if args.check_backend:
    print(ALL_BACKEND)
    exit(0)
  if not os.path.exists(args.source):
    print(f"Source image {args.source} not found.")
    exit(1)
  if not os.path.exists(args.target):
    print(f"Target image {args.target} not found.")
    exit(1)
  args.mpi_sync_interval = getattr(args, "mpi_sync_interval", 0)

  return args
