"""Conventional relay link S -> R -> D (AF or DF, half-duplex)."""

import numpy as np


def snr_relay_af(
    snr_sr_linear: float | np.ndarray,
    snr_rd_linear: float | np.ndarray,
) -> float | np.ndarray:
    """End-to-end SNR for AF relay (half-duplex): harmonic mean of SNR_SR and SNR_RD."""
    # 1/SNR_e2e = 1/SNR_sr + 1/SNR_rd (approximation for AF)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(snr_sr_linear > 0, 1 / snr_sr_linear, np.inf) + np.where(
            snr_rd_linear > 0, 1 / snr_rd_linear, np.inf
        )
        out = np.where(inv > 0, 1 / inv, 0.0)
    return out


def snr_relay_df(
    snr_sr_linear: float | np.ndarray,
    snr_rd_linear: float | np.ndarray,
) -> float | np.ndarray:
    """End-to-end SNR for DF relay: min(SNR_SR, SNR_RD) (bottleneck)."""
    return np.minimum(snr_sr_linear, snr_rd_linear)


def outage_relay(
    snr_e2e_linear: float | np.ndarray,
    threshold_linear: float,
) -> float | np.ndarray:
    """Outage indicator for relay link."""
    return (snr_e2e_linear < threshold_linear).astype(float)
