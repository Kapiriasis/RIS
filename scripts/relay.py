import os
from typing import Any, Dict, Optional
import numpy as np

from src.channel import free_space_path_loss, gain, noise_power, rician_fading
from src.plot import plot_capacity_hist, plot_snr_cdf
from src.utils import capacity, lin2db, snr_linear

def _default_results_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "results")

def run_relay_df(params: Dict[str, Any], results_dir: Optional[str] = None) -> Dict[str, Any]:

    if results_dir is None:
        results_dir = _default_results_dir()

    P_tx = params["P_tx"]
    f_c = params["frequency"]
    B = params["bandwidth"]
    N = params["N"]
    K_dB = params["K_dB"]
    d_total = params["distance"]

    # Two equal hops: Tx->Relay and Relay->Rx
    d1 = d_total / 2.0
    d2 = d_total / 2.0

    # Hop 1
    L0_1 = free_space_path_loss(d1, f_c)
    h1 = rician_fading(K_dB, N)
    G1 = gain(L0_1, h1)

    # Hop 2
    L0_2 = free_space_path_loss(d2, f_c)
    h2 = rician_fading(K_dB, N)
    G2 = gain(L0_2, h2)

    P_noise = noise_power(B)
    snr1 = snr_linear(P_tx, G1, P_noise)
    snr2 = snr_linear(P_tx, G2, P_noise)

    # Decode-and-forward: limited by the weaker hop
    snr_df = np.minimum(snr1, snr2)
    C_df = capacity(snr_df, B)

    mean_snr_db = float(lin2db(np.mean(snr_df)))
    outage_threshold_db = 0.0
    outage_threshold = 10.0 ** (outage_threshold_db / 10.0)
    outage_prob = float(np.mean(snr_df < outage_threshold))
    mean_capacity = float(np.mean(C_df))

    snr_cdf_path = os.path.join(results_dir, "relay_snr_cdf.png")
    cap_hist_path = os.path.join(results_dir, "relay_capacity_hist.png")
    plot_snr_cdf(snr_df, snr_cdf_path, label="relay DF")
    plot_capacity_hist(C_df, cap_hist_path, label="relay DF")

    return {
        "mean_snr_db": mean_snr_db,
        "outage_prob_0dB": outage_prob,
        "mean_capacity_bits_per_s": mean_capacity,
    }
