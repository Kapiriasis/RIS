import os
from typing import Any, Dict, Optional
import numpy as np

from src.channel import free_space_path_loss, gain, noise_power, rician_fading
from src.plot import plot_capacity_hist, plot_snr_cdf
from src.utils import capacity, lin2db, snr_linear

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

    L0 = free_space_path_loss(d, f_c)
    h = rician_fading(K_dB, N)
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
