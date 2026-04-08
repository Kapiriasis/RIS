import numpy as np
from src.channel import free_space_path_loss, log_distance_path_loss, noise_power, rician_fading
from src.ris_element import ris_element_coefficient
from src.utils import capacity, db2lin, lin2db, snr_linear

def ris_link_distances(total_distance, ris_position):
    d_tx_ris = float(ris_position)
    d_ris_rx = float(total_distance) - d_tx_ris

    if d_tx_ris <= 0.0 or d_ris_rx <= 0.0:
        raise ValueError("ris_position must satisfy 0 < ris_position < total_distance.")

    return d_tx_ris, d_ris_rx

def ris_phase_profile(h_tx_ris, h_ris_rx, n_bits=None):
    """
    Compute element-wise RIS reflection coefficients for coherent combining.

    For each element m:
        phi_m = -angle(h_tx_ris_m) - angle(h_ris_rx_m)
    so each reflected term aligns in phase at the receiver.
    """
    target_phase = -(np.angle(h_tx_ris) + np.angle(h_ris_rx))
    return ris_element_coefficient(phase_rad=target_phase, amplitude=1.0, n_bits=n_bits)

def ris_cascaded_channel(h_tx_ris, h_ris_rx, gamma, L_tx_ris, L_ris_rx):
    """
    Build the equivalent RIS cascaded channel over all elements.

    Equivalent channel (per sample):
        h_ris = sqrt(beta_1 * beta_2) * sum_m(gamma_m * h1_m * h2_m)
    where beta_i = 1 / L_i are large-scale path-loss gains.
    """
    h1 = np.asarray(h_tx_ris)
    h2 = np.asarray(h_ris_rx)
    g = np.asarray(gamma)
    if h1.shape != h2.shape or h1.shape != g.shape:
        raise ValueError("h_tx_ris, h_ris_rx, and gamma must have the same shape [M, N].")

    L1 = np.asarray(L_tx_ris)
    L2 = np.asarray(L_ris_rx)
    large_scale = np.sqrt((1.0 / L1) * (1.0 / L2))
    return np.sum(large_scale * g * h1 * h2, axis=0)

def combined_channel(h_direct, h_ris):
    return np.asarray(h_direct, dtype=complex) + np.asarray(h_ris, dtype=complex)


def simulate_ris_link(params, include_direct=True):
    """
    Simulate a RIS-assisted link and return key physical signals and metrics.

    Required params keys:
        P_tx, frequency, bandwidth, N, K_dB, distance, ris_array_size, ris_position
    Optional params keys:
        ris_phase_bits, include_direct
    """
    P_tx = params["P_tx"]
    f_c = params["frequency"]
    B = params["bandwidth"]
    N = int(params["N"])
    K_dB = params["K_dB"]
    d_total = params["distance"]
    M = int(params["ris_array_size"])
    ris_pos = params["ris_position"]
    n_bits = params.get("ris_phase_bits", None)
    n_exp = params["path_loss_exponent_los"]
    sigma_shadow_dB = params.get("shadowing_sigma_dB", 4.0)

    d_tx_ris, d_ris_rx = ris_link_distances(d_total, ris_pos)

    # Small-scale fading for each RIS element and each Monte-Carlo sample [M, N].
    h_tx_ris = np.vstack([rician_fading(K_dB, N) for _ in range(M)])
    h_ris_rx = np.vstack([rician_fading(K_dB, N) for _ in range(M)])

    Xg_tx_ris_dB = sigma_shadow_dB * np.random.standard_normal((1, N))
    Xg_ris_rx_dB = sigma_shadow_dB * np.random.standard_normal((1, N))
    d_min = d_total / 4.0
    L0_ref_dB = lin2db(free_space_path_loss(10.0, f_c))
    L_tx_ris = db2lin(log_distance_path_loss(L0_ref_dB, Xg_tx_ris_dB, n_exp, max(d_tx_ris, d_min)))
    L_ris_rx = db2lin(log_distance_path_loss(L0_ref_dB, Xg_ris_rx_dB, n_exp, max(d_ris_rx, d_min)))

    gamma = ris_phase_profile(h_tx_ris, h_ris_rx, n_bits=n_bits)
    h_ris = ris_cascaded_channel(h_tx_ris, h_ris_rx, gamma, L_tx_ris, L_ris_rx)

    # Optional direct path contribution.
    use_direct = include_direct or params.get("include_direct", False)
    if use_direct:
        Xg_direct_dB = sigma_shadow_dB * np.random.standard_normal(N)
        L_direct = db2lin(log_distance_path_loss(L0_ref_dB, Xg_direct_dB, n_exp, d_total))
        h_direct = np.sqrt(1.0 / L_direct) * rician_fading(K_dB, N)
    else:
        h_direct = np.zeros(N, dtype=complex)

    h_eff = combined_channel(h_direct, h_ris)

    G_eff = np.abs(h_eff) ** 2
    P_noise = noise_power(B, params.get("noise_figure_dB", 0.0))
    snr = snr_linear(P_tx, G_eff, P_noise)
    C = capacity(snr, B)

    outage_threshold = 10.0 ** (5.0 / 10.0)  # 5 dB in linear scale
    metrics = {
        "mean_snr_db": float(lin2db(np.mean(snr))),
        "outage_prob_5dB": float(np.mean(snr < outage_threshold)),
        "mean_capacity_bits_per_s": float(np.mean(C)),
    }

    return {
        "metrics": metrics,
        "snr_linear": snr,
        "capacity_bits_per_s": C,
        "h_eff": h_eff,
        "h_ris": h_ris,
        "h_direct": h_direct,
        "gamma": gamma,
        "distances_m": {
            "tx_ris": d_tx_ris,
            "ris_rx": d_ris_rx,
            "tx_rx": d_total,
        },
        "path_loss_linear": {
            "tx_ris": L_tx_ris,
            "ris_rx": L_ris_rx,
        },
    }
