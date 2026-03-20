"""Smoke tests for the public processors and CLI."""

import subprocess
import sys
import unittest

import numpy as np

from fpie.process import EquProcessor, GridProcessor


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
