import numpy as np

def db2lin(dB):
    return 10.0 ** (dB / 10.)

def lin2db(lin):
    return 10.0 * np.log10(lin)

def snr_linear(P_tx, gain, P_noise):
    snr = (np.asarray(P_tx) * np.asarray(gain)) / np.asarray(P_noise)
    return snr

def capacity(snr_linear, bandwidth):
    C = np.asarray(bandwidth) * np.log2(1.0 + np.asarray(snr_linear))
    return C
