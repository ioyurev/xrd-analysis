"""Supervisor for multi-start Charge Flipping ab initio structure solution."""

from typing import Any

import gemmi
import numpy as np
from numpy.typing import NDArray


def calc_pbc_distance(cell: gemmi.UnitCell, f1: NDArray[Any], f2: NDArray[Any]) -> float:
    """Calculate minimum Euclidean distance between fractional coordinates with Periodic Boundary Conditions (PBC)."""
    delta = np.asarray(f1, dtype=float) - np.asarray(f2, dtype=float)
    delta_pbc = delta - np.round(delta)
    orth = cell.orthogonalize(
        gemmi.Fractional(float(delta_pbc[0]), float(delta_pbc[1]), float(delta_pbc[2]))
    )
    return float(np.linalg.norm([orth.x, orth.y, orth.z]))


class MultiStartChargeFlipping:
    """Crystallographic PBC contact filter for pruning spurious Fourier series truncation ripples."""

    def __init__(
        self,
        ensemble_size: int = 30,
        r_dense_threshold: float = 0.25,
        min_distance_angstrom: float = 1.20,
    ) -> None:
        self.ensemble_size = ensemble_size
        self.r_dense_threshold = r_dense_threshold
        self.min_distance_angstrom = min_distance_angstrom

    def prune_close_contacts(
        self,
        cell_lengths: tuple[float, float, float],
        cell_angles: tuple[float, float, float],
        sites: list[tuple[str, NDArray[Any]]],
    ) -> tuple[list[tuple[str, NDArray[Any]]], int]:
        """Prune unphysical close-contact peak artifacts (Fourier ripples, d < 1.2 A) with PBC."""
        a, b, c = cell_lengths
        alpha, beta, gamma = cell_angles
        u_cell = gemmi.UnitCell(a, b, c, alpha, beta, gamma)

        kept: list[tuple[str, NDArray[Any]]] = []
        pruned_count = 0

        for elem, frac in sites:
            f_vec = np.asarray(frac, dtype=float)
            is_too_close = False
            for _, kept_frac in kept:
                k_vec = np.asarray(kept_frac, dtype=float)
                d = calc_pbc_distance(u_cell, f_vec, k_vec)
                if d < self.min_distance_angstrom:
                    is_too_close = True
                    break
            if not is_too_close:
                kept.append((elem, frac))
            else:
                pruned_count += 1

        return kept, pruned_count
