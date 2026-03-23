import numpy as np
from numpy.random import standard_normal
from src.utils import db2lin

def free_space_path_loss(distance, frequency):
    L0 = ((4 * np.pi * distance) / (3e8 / frequency)) ** 2
    return L0

def rician_fading(K_dB, N):
    K = db2lin(K_dB)
    mu = np.sqrt(K / (2 * (K + 1)))     # mean
    sigma = np.sqrt(1 / (2 * (K + 1)))  # sigma
    h = (sigma * standard_normal(N) + mu) + 1j * (sigma * standard_normal(N) + mu)
    return h

def log_distance_path_loss(L0, Xg, path_loss_exponent, distance):
    L_dB = L0 + 10 * path_loss_exponent * np.log10(distance / 10) + Xg
    return L_dB

def noise_power(bandwidth):
    k = 1.380649e-23    # Boltzmann constant
    temperature = 290   # 20°C
    P_noise = k * temperature * bandwidth
    return P_noise

def gain(L0, h):
    G = (1 / L0) * abs(h) ** 2
    return G
