import argparse


def get_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
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
    "-i",
    type=int,
    help="how many iteration would you perfer, the more the better"
  )
  parser.add_argument("-p", type=int, help="output result every P iteration")
  return parser.parse_args()


def main() -> None:
  args = get_args()
  print("here", args)
