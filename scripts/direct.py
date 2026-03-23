import os
from typing import Any, Dict, Optional
import numpy as np
from src.channel import (free_space_path_loss, gain, log_distance_path_loss, noise_power, rician_fading)
from src.plot import plot_capacity_hist, plot_snr_cdf
from src.utils import capacity, db2lin, lin2db, snr_linear

def _default_results_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "results")

def run_direct(params: Dict[str, Any], results_dir: Optional[str] = None) -> Dict[str, Any]:

    if results_dir is None:
        results_dir = _default_results_dir()

    P_tx = params["P_tx"]
    f_c = params["frequency"]
    B = params["bandwidth"]
    N = params["N"]
    K_dB = params["K_dB"]
    d = params["distance"]
    n_exp = params["path_loss_exponent"]

    h = rician_fading(K_dB, N)
    # Use a Rician-derived random dB term as Xg in the log-distance model.
    Xg_dB = lin2db(np.abs(rician_fading(K_dB, N)) ** 2)
    L0_dB = lin2db(free_space_path_loss(10.0, f_c))
    L_dB = log_distance_path_loss(L0_dB, Xg_dB, n_exp, d)
    L0 = db2lin(L_dB)
    G = gain(L0, h)

    P_noise = noise_power(B)
    snr = snr_linear(P_tx, G, P_noise)
    C = capacity(snr, B)

    mean_snr_db = float(lin2db(np.mean(snr)))
    outage_threshold_db = 5.0
    outage_threshold = 10.0 ** (outage_threshold_db / 10.0)
    outage_prob = float(np.mean(snr < outage_threshold))
    mean_capacity = float(np.mean(C))

    # Plots
    snr_cdf_path = os.path.join(results_dir, "direct_snr_cdf.png")
    cap_hist_path = os.path.join(results_dir, "direct_capacity_hist.png")
    plot_snr_cdf(snr, snr_cdf_path, label="direct")
    plot_capacity_hist(C, cap_hist_path, label="direct")

    return {
        "mean_snr_db": mean_snr_db,
        "outage_prob_5dB": outage_prob,
        "mean_capacity_bits_per_s": mean_capacity,
    }
