"""NumPy-based Poisson solvers."""

import numpy as np


class EquSolver:
    """Provide a NumPy-based Jacobi equation solver."""

    def __init__(self) -> None:
        """Initialize the equation solver state."""
        super().__init__()
        self.N = 0

    def partition(self, mask: np.ndarray) -> np.ndarray:
        """Assign contiguous ids to masked pixels."""
        return np.cumsum((mask > 0).reshape(-1)).reshape(mask.shape)

    def reset(
        self, N: int, A: np.ndarray, X: np.ndarray, B: np.ndarray
    ) -> None:
        """Load the linear system for the equation solver."""
        self.N = N
        self.A = A
        self.B = B
        self.X = X

    def sync(self) -> None:
        """Synchronize backend state if needed."""
        pass

    def step(self, iteration: int) -> tuple[np.ndarray, np.ndarray]:
        """Run a fixed number of Jacobi iterations."""
        for _ in range(iteration):
            # X = (B + AX) / 4
            self.X = (
                self.B
                + self.X[self.A[:, 0]]
                + self.X[self.A[:, 1]]
                + self.X[self.A[:, 2]]
                + self.X[self.A[:, 3]]
            ) / 4.0
        tmp = (
            self.B
            + self.X[self.A[:, 0]]
            + self.X[self.A[:, 1]]
            + self.X[self.A[:, 2]]
            + self.X[self.A[:, 3]]
            - 4.0 * self.X
        )
        err = np.abs(tmp).sum(axis=0)
        x = self.X.copy()
        x[x < 0] = 0
        x[x > 255] = 255
        return x, err


class GridSolver:
    """Provide a NumPy-based Jacobi grid solver."""

    def __init__(self) -> None:
        """Initialize the grid solver state."""
        super().__init__()
        self.N = 0

    def reset(
        self, N: int, mask: np.ndarray, tgt: np.ndarray, grad: np.ndarray
    ) -> None:
        """Load the masked target state for the grid solver."""
        self.N = N
        self.mask = mask
        self.bool_mask = mask.astype(bool)
        self.tgt = tgt
        self.grad = grad

    def sync(self) -> None:
        """Synchronize backend state if needed."""
        pass

    def step(self, iteration: int) -> tuple[np.ndarray, np.ndarray]:
        """Run a fixed number of Jacobi iterations."""
        for _ in range(iteration):
            # tgt = (grad + Atgt) / 4
            tgt = self.grad.copy()
            tgt[1:] += self.tgt[:-1]
            tgt[:-1] += self.tgt[1:]
            tgt[:, 1:] += self.tgt[:, :-1]
            tgt[:, :-1] += self.tgt[:, 1:]
            self.tgt[self.bool_mask] = tgt[self.bool_mask] / 4.0

        tmp = 4 * self.tgt - self.grad
        tmp[1:] -= self.tgt[:-1]
        tmp[:-1] -= self.tgt[1:]
        tmp[:, 1:] -= self.tgt[:, :-1]
        tmp[:, :-1] -= self.tgt[:, 1:]

        err = np.abs(tmp[self.bool_mask]).sum(axis=0)

        tgt = self.tgt.copy()
        tgt[tgt < 0] = 0
        tgt[tgt > 255] = 255
        return tgt, err
