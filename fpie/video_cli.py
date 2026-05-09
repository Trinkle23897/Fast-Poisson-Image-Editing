"""CLI entrypoint for video and stream Poisson image editing."""

import argparse

import fpie
from fpie.process import ALL_BACKEND, CPU_COUNT, DEFAULT_BACKEND
from fpie.video import BlendOptions, blend_video


def get_args() -> argparse.Namespace:
    """Parse video command-line arguments."""
    video_backends = [backend for backend in ALL_BACKEND if backend != "mpi"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--version", action="store_true", help="show the version and exit"
    )
    parser.add_argument(
        "--check-backend",
        action="store_true",
        help="print all available video backends",
    )
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        choices=video_backends,
        default=DEFAULT_BACKEND if DEFAULT_BACKEND != "mpi" else "numpy",
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
    parser.add_argument(
        "-s",
        "--source",
        help="source image/video filename, stream URL, or camera index",
    )
    parser.add_argument(
        "-m",
        "--mask",
        default="",
        help="mask image filename (default is to use the whole source frame)",
    )
    parser.add_argument(
        "-t",
        "--target",
        help="target video filename, stream URL, or camera index",
    )
    parser.add_argument("-o", "--output", help="output video filename")
    parser.add_argument(
        "-h0",
        type=int,
        help="mask position (height) on source frame",
        default=0,
    )
    parser.add_argument(
        "-w0",
        type=int,
        help="mask position (width) on source frame",
        default=0,
    )
    parser.add_argument(
        "-h1",
        type=int,
        help="mask position (height) on target frame",
        default=0,
    )
    parser.add_argument(
        "-w1",
        type=int,
        help="mask position (width) on target frame",
        default=0,
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
        help="how many iterations to run per frame",
        default=5000,
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="override output FPS (default uses target FPS or 30)",
    )
    parser.add_argument(
        "--fourcc",
        type=str,
        default=None,
        help="four-character video codec, e.g. mp4v or MJPG",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="stop after this many frames (0 means no limit)",
    )
    parser.add_argument(
        "--loop-source",
        action="store_true",
        help="loop source video if it ends before the target stream",
    )
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
        raise SystemExit(0)
    if args.check_backend:
        print(video_backends)
        raise SystemExit(0)
    missing = [
        option
        for option, value in (
            ("-s/--source", args.source),
            ("-t/--target", args.target),
            ("-o/--output", args.output),
        )
        if not value
    ]
    if missing:
        parser.error(
            f"the following arguments are required: {', '.join(missing)}"
        )
    return args


def main() -> None:
    """Run the video command-line application."""
    args = get_args()
    options = BlendOptions(
        method=args.method,
        gradient=args.gradient,
        backend=args.backend,
        iterations=args.n,
        n_cpu=args.cpu,
        mpi_sync_interval=args.mpi_sync_interval,
        block_size=args.block_size,
        grid_x=args.grid_x,
        grid_y=args.grid_y,
    )
    result = blend_video(
        args.source,
        args.target,
        args.output,
        mask=args.mask,
        mask_on_src=(args.h0, args.w0),
        mask_on_tgt=(args.h1, args.w1),
        options=options,
        fps=args.fps,
        fourcc=args.fourcc,
        max_frames=args.max_frames or None,
        loop_source=args.loop_source,
    )
    print(
        f"Successfully wrote {result.frame_count} frames "
        f"({result.size[0]}x{result.size[1]} @ {result.fps:.2f} FPS) "
        f"to {result.output}"
    )


if __name__ == "__main__":
    main()
