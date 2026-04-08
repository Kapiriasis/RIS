import os
import numpy as np
from typing import Any, Dict, Optional
from src.channel import free_space_path_loss, log_distance_path_loss, noise_power, rician_fading
from src.ris_channel import simulate_ris_link
from src.signal import ber_vs_snr, theoretical_bpsk_ber
from src.plot import plot_ber_vs_snr
from src.utils import db2lin, lin2db

def _default_results_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "results")

def _direct_link_channel(params: Dict[str, Any]) -> np.ndarray:
    """
    Build complex effective channel h_eff for the direct TX→RX path.

    Returns h_eff of shape (N,) consistent with simulate_ris_link output.
    """
    f_c = params["frequency"]
    N = int(params["N"])
    K_dB = params["K_dB"]
    d = params["distance"]
    n_exp = params["path_loss_exponent"]
    sigma_shadow_dB = params.get("shadowing_sigma_dB", 4.0)

    L0_ref_dB = lin2db(free_space_path_loss(10.0, f_c))
    Xg_dB = sigma_shadow_dB * np.random.standard_normal(N)
    L_dB = log_distance_path_loss(L0_ref_dB, Xg_dB, n_exp, d)
    L = db2lin(L_dB)
    return np.sqrt(1.0 / L) * rician_fading(K_dB, N)

def run_ber(
    params: Dict[str, Any],
    snr_db_range: Optional[np.ndarray] = None,
    results_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute and plot BPSK BER vs. average SNR for:
      - Direct link only
      - RIS-assisted link (no direct path)
      - RIS-assisted link (with direct path)
      - Theoretical AWGN (reference)

    The signal model used at each SNR point is:

        y[n] = h_eff[n] · sqrt(P_tx) · s[n] + noise[n]

    where h_eff is the effective channel for the relevant configuration.
    P_tx is adjusted at each SNR point so that:

        E[ |h_eff|^2 ] · P_tx / P_noise = SNR_target

    Parameters
    ----------
    params        : simulation parameter dict (same as used by run_direct / run_ris)
    snr_db_range  : SNR values in dB for the x-axis (default: -5 to 25 dB)
    results_dir   : directory for output plots

    Returns
    -------
    dict with keys: snr_db, ber_direct, ber_ris, ber_ris_direct, ber_theoretical
    """
    if results_dir is None:
        results_dir = _default_results_dir()
    if snr_db_range is None:
        snr_db_range = np.arange(-5, 26, dtype=float)

    B = params["bandwidth"]
    P_noise = noise_power(B, params.get("noise_figure_dB", 0.0))

    # --- Channel realizations for each configuration ---
    h_eff_direct = _direct_link_channel(params)

    ris_out = simulate_ris_link(params, include_direct=False)
    h_eff_ris = ris_out["h_eff"]

    ris_direct_out = simulate_ris_link(params, include_direct=True)
    h_eff_ris_direct = ris_direct_out["h_eff"]

    # --- BER sweep ---
    ber_direct = ber_vs_snr(h_eff_direct, snr_db_range, P_noise)
    ber_ris = ber_vs_snr(h_eff_ris, snr_db_range, P_noise)
    ber_ris_direct = ber_vs_snr(h_eff_ris_direct, snr_db_range, P_noise)

    snr_lin = 10.0 ** (snr_db_range / 10.0)
    ber_theoretical = theoretical_bpsk_ber(snr_lin)

    # Replace exact zeros with a floor so semilogy doesn't drop points
    floor = 0.5 / int(params["N"])
    ber_direct = np.maximum(ber_direct, floor)
    ber_ris = np.maximum(ber_ris, floor)
    ber_ris_direct = np.maximum(ber_ris_direct, floor)

    # --- Plot ---
    out_path = os.path.join(results_dir, "ber_vs_snr.png")
    plot_ber_vs_snr(
        snr_db=snr_db_range,
        ber_curves=[ber_direct, ber_ris, ber_ris_direct, ber_theoretical],
        labels=["Direct", "RIS only", "RIS + Direct", "Theoretical AWGN"],
        out_path=out_path,
    )

    return {
        "snr_db": snr_db_range.tolist(),
        "ber_direct": ber_direct.tolist(),
        "ber_ris": ber_ris.tolist(),
        "ber_ris_direct": ber_ris_direct.tolist(),
        "ber_theoretical": ber_theoretical.tolist(),
    }
