import numpy as np
from typing import Any, Dict
from src.channel import (free_space_path_loss, gain, log_distance_path_loss, noise_power, rician_fading)
from src.utils import capacity, db2lin, lin2db, snr_linear

def run_direct(params: Dict[str, Any]) -> Dict[str, Any]:

    P_tx = params["P_tx"]
    f_c = params["frequency"]
    B = params["bandwidth"]
    N = params["N"]
    K_dB = params["K_dB"]
    d = params["distance"]
    n_exp = params["path_loss_exponent"]
    sigma_shadow_dB = params.get("shadowing_sigma_dB", 4.0)

    h = rician_fading(K_dB, N)
    Xg_dB = sigma_shadow_dB * np.random.standard_normal(N)
    L0_dB = lin2db(free_space_path_loss(10.0, f_c))
    L_dB = log_distance_path_loss(L0_dB, Xg_dB, n_exp, d)
    L0 = db2lin(L_dB)
    G = gain(L0, h)

    P_noise = noise_power(B, params.get("noise_figure_dB", 0.0))
    snr = snr_linear(P_tx, G, P_noise)
    C = capacity(snr, B)

    mean_snr_db = float(lin2db(np.mean(snr)))
    outage_threshold_db = 5.0
    outage_threshold = 10.0 ** (outage_threshold_db / 10.0)
    outage_prob = float(np.mean(snr < outage_threshold))
    mean_capacity = float(np.mean(C))

    return {
        "mean_snr_db": mean_snr_db,
        "outage_prob_5dB": outage_prob,
        "mean_capacity_bits_per_s": mean_capacity,
        "snr_linear": snr,
        "capacity": C,
    }
