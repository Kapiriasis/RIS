import numpy as np

def quantize_phase(phase_rad, n_bits):
    # Quantize phase(s) to a uniform codebook over [0, 2*pi)
    if n_bits is None:
        return np.asarray(phase_rad)

    if n_bits <= 0:
        raise ValueError("n_bits must be a positive integer or None.")

    levels = 2 ** int(n_bits)
    phase = np.asarray(phase_rad) % (2.0 * np.pi)
    step = (2.0 * np.pi) / levels
    indices = np.round(phase / step) % levels
    return indices * step

def ris_element_coefficient(phase_rad, amplitude=1.0, n_bits=None):
    # Compute the complex reflection coefficient of one RIS element.
    alpha = np.asarray(amplitude)
    if np.any(alpha < 0.0) or np.any(alpha > 1.0):
        raise ValueError("amplitude must lie in [0, 1].")

    phi_q = quantize_phase(phase_rad, n_bits=n_bits)
    return alpha * np.exp(1j * phi_q)
