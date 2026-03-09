"""Direct link S -> D (no relay, no RIS)."""

import numpy as np


def snr_direct(
    tx_power_linear: float,
    channel_gain_linear: float,
    noise_power_linear: float,
) -> float:
    """SNR for direct link: P * |h|^2 / N."""
    return tx_power_linear * channel_gain_linear / noise_power_linear


def outage_direct(
    snr_linear: float | np.ndarray,
    threshold_linear: float,
) -> float | np.ndarray:
    """Outage indicator: 1 if SNR < threshold, else 0. For arrays, mean = outage prob."""
    return (snr_linear < threshold_linear).astype(float)
