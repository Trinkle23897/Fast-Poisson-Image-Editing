"""Smoke tests for the public processors and CLI."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from fpie.process import (
    ALL_BACKEND,
    EquProcessor,
    GridProcessor,
)
from fpie.video import BlendOptions, blend_frame, blend_video


class SmokeTest(unittest.TestCase):
    """Exercise the main local backends and CLI."""

    def setUp(self) -> None:
        """Create a tiny synthetic blending problem."""
        self.src = np.zeros((6, 6, 3), dtype=np.uint8)
        self.mask = np.zeros((6, 6), dtype=np.uint8)
        self.mask[2:4, 2:4] = 255
        self.tgt = np.ones((6, 6, 3), dtype=np.uint8) * 10

    def test_equ_processor_numpy_backend(self) -> None:
        """Verify the equation processor produces a valid output."""
        proc = EquProcessor(backend="numpy")
        n = proc.reset(self.src, self.mask, self.tgt, (0, 0), (0, 0))
        out, err = proc.step(2)

        self.assertGreater(n, 0)
        self.assertEqual(out.shape, self.tgt.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(err.shape, (3,))

    def test_grid_processor_numpy_backend(self) -> None:
        """Verify the grid processor produces a valid output."""
        proc = GridProcessor(backend="numpy")
        n = proc.reset(self.src, self.mask, self.tgt, (0, 0), (0, 0))
        out, err = proc.step(2)

        self.assertGreater(n, 0)
        self.assertEqual(out.shape, self.tgt.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(err.shape, (3,))

    @unittest.skipUnless("openmp" in ALL_BACKEND, "OpenMP backend unavailable")
    def test_grid_processor_openmp_matches_numpy(self) -> None:
        """OpenMP grid solver should match the NumPy Jacobi update."""
        rng = np.random.default_rng(0)
        src = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
        tgt = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
        mask = np.zeros((24, 24), dtype=np.uint8)
        mask[2:-2, 2:-2] = (rng.random((20, 20)) > 0.35).astype(np.uint8) * 255

        proc_np = GridProcessor(backend="numpy", grid_x=1, grid_y=1)
        proc_omp = GridProcessor(backend="openmp", n_cpu=4, grid_x=1, grid_y=1)
        proc_np.reset(src, mask, tgt.copy(), (0, 0), (0, 0))
        proc_omp.reset(src, mask, tgt.copy(), (0, 0), (0, 0))

        out_np, err_np = proc_np.step(5)
        out_omp, err_omp = proc_omp.step(5)

        np.testing.assert_array_equal(out_omp, out_np)
        np.testing.assert_allclose(err_omp, err_np, rtol=1e-5, atol=1e-5)

    def test_cli_check_backend(self) -> None:
        """Verify the CLI can report available backends."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "from fpie.cli import main; "
                    "sys.argv = ['fpie', '--check-backend']; "
                    "main()"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("numpy", result.stdout)

    def test_video_cli_check_backend(self) -> None:
        """Verify the video CLI can report available backends."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "from fpie.video_cli import main; "
                    "sys.argv = ['fpie-video', '--check-backend']; "
                    "main()"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("numpy", result.stdout)

    def test_blend_frame_numpy_backend(self) -> None:
        """Verify the public frame interface blends one target frame."""
        out = blend_frame(
            self.src,
            self.mask,
            self.tgt,
            options=BlendOptions(backend="numpy", iterations=2),
        )

        self.assertEqual(out.shape, self.tgt.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_blend_video_numpy_backend(self) -> None:
        """Verify the video interface writes a blended output stream."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src_path = tmp_path / "src.png"
            mask_path = tmp_path / "mask.png"
            target_path = tmp_path / "target.avi"
            output_path = tmp_path / "out.avi"

            cv2.imwrite(str(src_path), self.src)
            cv2.imwrite(str(mask_path), self.mask)
            writer = cv2.VideoWriter(
                str(target_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                5.0,
                (self.tgt.shape[1], self.tgt.shape[0]),
            )
            self.assertTrue(writer.isOpened())
            writer.write(self.tgt)
            writer.write(self.tgt + 1)
            writer.release()

            result = blend_video(
                str(src_path),
                str(target_path),
                str(output_path),
                mask=str(mask_path),
                options=BlendOptions(backend="numpy", iterations=2),
                fps=5.0,
                fourcc="MJPG",
            )

            self.assertEqual(result.frame_count, 2)
            self.assertTrue(output_path.exists())

            capture = cv2.VideoCapture(str(output_path))
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
            ok, frame = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertEqual(frame.shape[:2], self.tgt.shape[:2])


if __name__ == "__main__":
    unittest.main()
