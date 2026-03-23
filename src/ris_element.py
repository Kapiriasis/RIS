import numpy as np

def normalize_phase(phase_rad):
    # Wrap phase(s) to [-pi, pi)
    phase = np.asarray(phase_rad)
    wrapped = (phase + np.pi) % (2.0 * np.pi) - np.pi
    return wrapped

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
    """
    Compute the complex reflection coefficient of one RIS element.

    The element response is:
        Gamma = alpha * exp(j * phi_q)

    where:
    - alpha is the reflection amplitude in [0, 1]
    - phi_q is the element phase (optionally quantized by n_bits)
    """
    alpha = np.asarray(amplitude)
    if np.any(alpha < 0.0) or np.any(alpha > 1.0):
        raise ValueError("amplitude must lie in [0, 1].")

    phi_q = quantize_phase(phase_rad, n_bits=n_bits)
    return alpha * np.exp(1j * phi_q)

def apply_ris_element(incident_signal, phase_rad, amplitude=1.0, n_bits=None):
    """
    Reflect an incident complex signal through a single RIS element.

    Returns:
        reflected_signal = incident_signal * ris_element_coefficient(...)
    """
    gamma = ris_element_coefficient(
        phase_rad=phase_rad,
        amplitude=amplitude,
        n_bits=n_bits,
    )
    return np.asarray(incident_signal) * gamma
