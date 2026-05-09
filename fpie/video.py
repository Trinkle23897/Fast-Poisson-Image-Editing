"""Video and stream processing helpers for Poisson image editing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from fpie.io import read_image
from fpie.process import (
    CPU_COUNT,
    DEFAULT_BACKEND,
    BaseProcessor,
    EquProcessor,
    GridProcessor,
)

DEFAULT_FPS = 30.0


@dataclass(frozen=True)
class BlendOptions:
    """Configuration shared by image, video, and stream blending."""

    method: str = "equ"
    gradient: str = "max"
    backend: str = DEFAULT_BACKEND
    iterations: int = 5000
    n_cpu: int = CPU_COUNT
    mpi_sync_interval: int = 100
    block_size: int = 1024
    grid_x: int = 8
    grid_y: int = 8


@dataclass(frozen=True)
class VideoBlendResult:
    """Summary returned after a video blend completes."""

    output: str
    frame_count: int
    fps: float
    size: tuple[int, int]


class _FrameSource:
    def __init__(self, source: str | int, *, loop: bool = False):
        self.source = source
        self.loop = loop
        self.image = None
        self.capture = None

        if isinstance(source, str):
            self.image = cv2.imread(source)
        if self.image is not None:
            if self.image.ndim == 2:
                self.image = np.stack([self.image, self.image, self.image], axis=-1)
            elif self.image.ndim == 3 and self.image.shape[-1] == 4:
                self.image = self.image[..., :-1]
            return

        self.capture = cv2.VideoCapture(_coerce_capture_source(source))
        if not self.capture.isOpened():
            raise FileNotFoundError(f"Failed to open video source: {source}")

    def read(self) -> np.ndarray | None:
        if self.image is not None:
            return self.image

        assert self.capture is not None
        ok, frame = self.capture.read()
        if ok:
            return frame
        if not self.loop:
            return None

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self.capture.read()
        if ok:
            return frame
        return None

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()


def create_processor(options: BlendOptions) -> BaseProcessor:
    """Create a processor for reusable frame-by-frame blending."""
    if options.backend == "mpi":
        raise ValueError("Video and stream processing do not support the MPI backend.")

    if options.method == "equ":
        return EquProcessor(
            options.gradient,
            options.backend,
            options.n_cpu,
            options.mpi_sync_interval,
            options.block_size,
        )
    if options.method == "grid":
        return GridProcessor(
            options.gradient,
            options.backend,
            options.n_cpu,
            options.mpi_sync_interval,
            options.block_size,
            options.grid_x,
            options.grid_y,
        )
    raise ValueError(f"Invalid method: {options.method}")


def blend_frame(
    src: np.ndarray,
    mask: np.ndarray,
    tgt: np.ndarray,
    *,
    mask_on_src: tuple[int, int] = (0, 0),
    mask_on_tgt: tuple[int, int] = (0, 0),
    options: BlendOptions | None = None,
    processor: BaseProcessor | None = None,
) -> np.ndarray:
    """Blend one source frame into one target frame."""
    options = options or BlendOptions()
    proc = processor or create_processor(options)
    proc.reset(src, mask, tgt, mask_on_src, mask_on_tgt)
    if options.iterations <= 0:
        return tgt.copy()

    result = proc.step(options.iterations)
    if result is None:
        raise RuntimeError("The selected processor did not return a root result.")
    frame, _err = result
    return frame


def blend_frames(
    src_frames: Iterable[np.ndarray],
    mask: np.ndarray,
    tgt_frames: Iterable[np.ndarray],
    *,
    mask_on_src: tuple[int, int] = (0, 0),
    mask_on_tgt: tuple[int, int] = (0, 0),
    options: BlendOptions | None = None,
) -> Iterable[np.ndarray]:
    """Yield blended frames from source and target frame iterables."""
    options = options or BlendOptions()
    processor = create_processor(options)
    for src, tgt in zip(src_frames, tgt_frames, strict=False):
        yield blend_frame(
            src,
            mask,
            tgt,
            mask_on_src=mask_on_src,
            mask_on_tgt=mask_on_tgt,
            options=options,
            processor=processor,
        )


def blend_video(
    source: str | int,
    target: str | int,
    output: str,
    *,
    mask: str | np.ndarray | None = None,
    mask_on_src: tuple[int, int] = (0, 0),
    mask_on_tgt: tuple[int, int] = (0, 0),
    options: BlendOptions | None = None,
    fps: float | None = None,
    fourcc: str | None = None,
    max_frames: int | None = None,
    loop_source: bool = False,
) -> VideoBlendResult:
    """Blend a source image/video into a target video or realtime stream."""
    options = options or BlendOptions()
    processor = create_processor(options)
    source_frames = _FrameSource(source, loop=loop_source)
    target_capture = cv2.VideoCapture(_coerce_capture_source(target))
    if not target_capture.isOpened():
        source_frames.release()
        raise FileNotFoundError(f"Failed to open target video source: {target}")

    output_path = Path(output)
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    target_fps = fps or target_capture.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    width = int(target_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(target_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_frame: np.ndarray | None = None
    if width <= 0 or height <= 0:
        ok, first_frame = target_capture.read()
        if not ok:
            source_frames.release()
            target_capture.release()
            raise RuntimeError(f"No frames available from target source: {target}")
        height, width = first_frame.shape[:2]

    writer = cv2.VideoWriter(
        output,
        cv2.VideoWriter_fourcc(*_pick_fourcc(output, fourcc)),
        target_fps,
        (width, height),
    )
    if not writer.isOpened():
        source_frames.release()
        target_capture.release()
        raise RuntimeError(f"Failed to open video writer: {output}")

    mask_image = _load_mask(mask)
    frame_count = 0
    try:
        while max_frames is None or frame_count < max_frames:
            if first_frame is not None:
                target_frame = first_frame
                first_frame = None
            else:
                ok, target_frame = target_capture.read()
                if not ok:
                    break
            source_frame = source_frames.read()
            if source_frame is None:
                break

            frame_mask = (
                mask_image
                if mask_image is not None
                else np.zeros(source_frame.shape[:2], dtype=np.uint8) + 255
            )
            blended = blend_frame(
                source_frame,
                frame_mask,
                target_frame,
                mask_on_src=mask_on_src,
                mask_on_tgt=mask_on_tgt,
                options=options,
                processor=processor,
            )
            writer.write(blended)
            frame_count += 1
    finally:
        writer.release()
        target_capture.release()
        source_frames.release()

    return VideoBlendResult(
        output=output,
        frame_count=frame_count,
        fps=float(target_fps),
        size=(width, height),
    )


def _load_mask(mask: str | np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    if isinstance(mask, np.ndarray):
        return mask
    if mask == "":
        return None
    return read_image(mask)


def _coerce_capture_source(source: str | int) -> str | int:
    if isinstance(source, int):
        return source
    if source.isdecimal():
        return int(source)
    return source


def _pick_fourcc(output: str, fourcc: str | None) -> str:
    if fourcc is not None:
        if len(fourcc) != 4:
            raise ValueError("fourcc must be exactly 4 characters.")
        return fourcc
    if Path(output).suffix.lower() == ".mp4":
        return "mp4v"
    return "MJPG"
