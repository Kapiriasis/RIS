"""Fading models (Rayleigh, Rician)."""

import numpy as np


def rayleigh_gain(rng: np.random.Generator, size: int = 1) -> np.ndarray:
    """Rayleigh fading: |h|^2 with E[|h|^2]=1."""
    h = (rng.standard_normal(size) + 1j * rng.standard_normal(size)) / np.sqrt(2)
    return np.abs(h) ** 2


def rician_gain(rng: np.random.Generator, K_db: float, size: int = 1) -> np.ndarray:
    """Rician fading with factor K (dB). E[|h|^2]=1."""
    K_lin = 10 ** (K_db / 10)
    los = np.sqrt(K_lin / (1 + K_lin))
    scat = 1 / np.sqrt(2 * (1 + K_lin))
    h = los + scat * (rng.standard_normal(size) + 1j * rng.standard_normal(size))
    return np.abs(h) ** 2
