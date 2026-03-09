"""Path loss models (log-distance, free space)."""

import numpy as np


def path_loss_db(distance_m: float, freq_hz: float, exponent: float = 2.0) -> float:
    """
    Log-distance path loss in dB.
    PL(d) = PL(d0) + 10*n*log10(d/d0). Free-space at d0=1m as reference.
    """
    d0 = 1.0
    c = 3e8
    lam = c / freq_hz
    pl_d0_db = 20 * np.log10(4 * np.pi * d0 / lam)
    return pl_d0_db + 10 * exponent * np.log10(max(distance_m, d0) / d0)


def path_loss_linear(distance_m: float, freq_hz: float, exponent: float = 2.0) -> float:
    """Path loss as linear power gain (< 1)."""
    return 10 ** (-path_loss_db(distance_m, freq_hz) / 10)
