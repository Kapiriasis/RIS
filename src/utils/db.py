"""Linear <-> dB conversions."""

import numpy as np


def to_db(x: float | np.ndarray) -> float | np.ndarray:
    """Power to dB: 10*log10(x). Safely handles zeros."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10 * np.log10(np.where(x > 0, x, np.nan))
    return float(out) if out.size == 1 else out


def from_db(x_db: float | np.ndarray) -> float | np.ndarray:
    """dB to linear power: 10^(x_db/10)."""
    x_db = np.asarray(x_db, dtype=float)
    out = 10 ** (x_db / 10)
    return float(out) if out.size == 1 else out
