import numpy as np

def db2lin(dB):
    return 10. ** (dB / 10.)

def lin2db(lin):
    return 10. * np.log10(lin)


def snr_linear(P_tx, gain, P_noise):
    snr = (P_tx * gain) / P_noise
    return snr

def capacity(snr_linear, bandwidth):
    C = bandwidth * np.log2(1 + snr_linear)
    return C
