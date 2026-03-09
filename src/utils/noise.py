"""Noise power and AWGN."""

import numpy as np


def noise_power_linear(
    bandwidth_hz: float,
    noise_figure_db: float = 0.0,
    temperature_k: float = 290.0,
) -> float:
    """Thermal noise power in linear scale: N = k*T*B*F."""
    k_boltzmann = 1.380649e-23
    F_linear = 10 ** (noise_figure_db / 10)
    return k_boltzmann * temperature_k * bandwidth_hz * F_linear
