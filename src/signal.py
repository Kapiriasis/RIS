import numpy as np
from scipy.special import erfc

def bpsk_symbols(N):
    """
    Generate N random BPSK symbols.

    Returns
    -------
    s : (N,) int array with values ∈ {+1, -1}
    """
    bits = np.random.randint(0, 2, int(N))
    return 2 * bits - 1  # map 0 → -1, 1 → +1

def received_signal(s, h_eff, P_tx, P_noise):
    """
    Explicit point-to-point signal model:

        y[n] = h_eff[n] · sqrt(P_tx) · s[n] + noise[n]

    h_eff is the effective complex channel combining the direct path and all
    RIS-reflected paths:

        h_eff = h_direct + Σ_m [ α_m · e^(j·φ_m) · h1_m · h2_m ]

    noise[n] is complex AWGN with total power P_noise, split equally between
    the real and imaginary components.

    Parameters
    ----------
    s       : (N,) real BPSK symbols ∈ {+1, -1}
    h_eff   : (N,) complex effective channel (direct + RIS-reflected)
    P_tx    : scalar transmit power [W]
    P_noise : scalar one-sided noise power [W]

    Returns
    -------
    y : (N,) complex received signal
    """
    s = np.asarray(s, dtype=float)
    h_eff = np.asarray(h_eff, dtype=complex)
    N = len(s)
    noise = np.sqrt(P_noise / 2.0) * (
        np.random.standard_normal(N) + 1j * np.random.standard_normal(N)
    )
    return h_eff * np.sqrt(float(P_tx)) * s + noise

def detect_bpsk(y, h_eff):
    """
    Coherent MRC detection for BPSK.

    The sufficient statistic for BPSK over a complex channel is:

        z[n] = Re( conj(h_eff[n]) · y[n] )

    Decision: s_hat[n] = sign( z[n] )

    Parameters
    ----------
    y     : (N,) complex received signal
    h_eff : (N,) complex effective channel (must be the same realizations used
            when transmitting)

    Returns
    -------
    s_hat : (N,) detected symbols ∈ {+1, -1}
    """
    z = np.real(np.conj(np.asarray(h_eff, dtype=complex)) * np.asarray(y, dtype=complex))
    return np.sign(z)

def bpsk_ber(h_eff, P_tx, P_noise):
    """
    Empirical BPSK BER for a given set of channel realizations.

    One symbol is transmitted per channel realization; the channel is treated
    as perfectly known at the receiver (coherent detection).

    Parameters
    ----------
    h_eff   : (N,) complex effective channel realizations
    P_tx    : transmit power [W]
    P_noise : noise power [W]

    Returns
    -------
    ber : scalar BER estimate in [0, 1]
    """
    h_eff = np.asarray(h_eff, dtype=complex)
    N = len(h_eff)
    s = bpsk_symbols(N)
    y = received_signal(s, h_eff, P_tx, P_noise)
    s_hat = detect_bpsk(y, h_eff)
    return float(np.mean(s_hat != s))

def theoretical_bpsk_ber(snr_linear):
    """
    Theoretical BPSK BER over an AWGN channel:

        BER = 0.5 · erfc( sqrt(SNR) )

    Used as a baseline reference curve on BER vs. SNR plots.

    Parameters
    ----------
    snr_linear : array-like, linear SNR values (not dB)

    Returns
    -------
    ber : ndarray of theoretical BER values
    """
    return 0.5 * erfc(np.sqrt(np.asarray(snr_linear, dtype=float)))

def ber_vs_snr(h_eff, snr_db_range, P_noise):
    """
    Semi-analytical BPSK BER across a range of average SNR values.

    P_tx is set at each point so that E[|h_eff|²]·P_tx/P_noise = SNR_target,
    then the per-realization conditional BER is averaged:

        BER ≈ mean_n( 0.5·erfc( sqrt( |h_eff[n]|² · P_tx / P_noise ) ) )

    Averaging the closed-form expression over channel realizations gives smooth
    curves regardless of how low the BER is, avoiding the statistical noise of
    counting discrete bit errors at high SNR.
    """
    h_eff = np.asarray(h_eff, dtype=complex)
    gain_per_sample = np.abs(h_eff) ** 2
    mean_channel_gain = float(np.mean(gain_per_sample))

    snr_db_range = np.asarray(snr_db_range, dtype=float)
    ber_out = np.empty(len(snr_db_range))
    for i, snr_db in enumerate(snr_db_range):
        snr_lin = 10.0 ** (snr_db / 10.0)
        P_tx = snr_lin * P_noise / mean_channel_gain
        snr_per_sample = gain_per_sample * P_tx / P_noise
        ber_out[i] = float(np.mean(0.5 * erfc(np.sqrt(snr_per_sample))))

    return ber_out
