"""Smoke tests for the public processors and CLI."""

import subprocess
import sys
import unittest

import numpy as np

from fpie.process import (
    ALL_BACKEND,
    EquProcessor,
    GridProcessor,
)


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
        mask[2:-2, 2:-2] = (
            rng.random((20, 20)) > 0.35
        ).astype(np.uint8) * 255

        proc_np = GridProcessor(backend="numpy", grid_x=1, grid_y=1)
        proc_omp = GridProcessor(
            backend="openmp", n_cpu=4, grid_x=1, grid_y=1
        )
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

if __name__ == "__main__":
    unittest.main()
