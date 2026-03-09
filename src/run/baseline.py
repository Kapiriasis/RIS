"""Run baseline simulation: direct link and conventional relay."""

import numpy as np
from pathlib import Path

from src.channel import path_loss_linear, rayleigh_gain
from src.links import snr_direct, snr_relay_af, outage_direct, outage_relay
from src.utils import noise_power_linear, from_db


def run_baseline(
    *,
    freq_hz: float = 2.4e9,
    bandwidth_hz: float = 1e6,
    noise_figure_db: float = 0.0,
    d_sd_m: float = 100.0,
    d_sr_m: float = 50.0,
    d_rd_m: float = 50.0,
    path_loss_exponent: float = 2.0,
    tx_power_dbm: float = 20.0,
    snr_threshold_db: float = 0.0,
    n_trials: int = 10_000,
    seed: int = 42,
    output_dir: str | Path = "results",
) -> dict:
    """
    Run Monte Carlo baseline: direct (S->D) and relay (S->R->D, AF).
    Returns dict with snr_dbm, outage_direct, outage_relay arrays and params.
    """
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parameters in linear scale
    tx_power_linear = from_db(tx_power_dbm - 30)  # dBm -> W (approx, 0 dBm = 1 mW)
    threshold_linear = from_db(snr_threshold_db)
    N = noise_power_linear(bandwidth_hz, noise_figure_db)

    # Path gains (no fading first: deterministic baseline)
    g_sd = path_loss_linear(d_sd_m, freq_hz, path_loss_exponent)
    g_sr = path_loss_linear(d_sr_m, freq_hz, path_loss_exponent)
    g_rd = path_loss_linear(d_rd_m, freq_hz, path_loss_exponent)

    # Fading realizations
    h_sd = rayleigh_gain(rng, n_trials)
    h_sr = rayleigh_gain(rng, n_trials)
    h_rd = rayleigh_gain(rng, n_trials)

    # Channel gains
    gain_sd = g_sd * h_sd
    gain_sr = g_sr * h_sr
    gain_rd = g_rd * h_rd

    # SNRs
    snr_d = snr_direct(tx_power_linear, gain_sd, N)
    snr_sr = tx_power_linear * gain_sr / N
    snr_rd = tx_power_linear * gain_rd / N  # same Tx power from relay
    snr_r = snr_relay_af(snr_sr, snr_rd)

    # Outage
    out_d = outage_direct(snr_d, threshold_linear)
    out_r = outage_relay(snr_r, threshold_linear)

    results = {
        "snr_direct_linear": snr_d,
        "snr_relay_linear": snr_r,
        "outage_direct": out_d,
        "outage_relay": out_r,
        "outage_prob_direct": float(np.mean(out_d)),
        "outage_prob_relay": float(np.mean(out_r)),
        "params": {
            "freq_hz": freq_hz,
            "d_sd_m": d_sd_m,
            "d_sr_m": d_sr_m,
            "d_rd_m": d_rd_m,
            "tx_power_dbm": tx_power_dbm,
            "snr_threshold_db": snr_threshold_db,
            "n_trials": n_trials,
            "seed": seed,
        },
    }
    return results
