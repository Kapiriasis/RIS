import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_PATH = os.path.join(ROOT, "data", "params.json")

DEFAULT_PARAMS = {
    "P_tx": 0.1,                # wifi access point: 100 mW, base station: 10 W
    "frequency": 2.4e9,         # 2.4 GHz
    "bandwidth": 20e6,          # 20 MHz
    "N": 1000,                  # number of samples
    "K_dB": 15,                 # outdoor: 10 - 15, indoor: 0 - 6,
    "path_loss_exponent": 4.0,  # free space: 2, outdoor: 4, indoor: 6
    "shadowing_sigma_dB": 4.0,  # log-normal shadowing std-dev in dB
    "distance": 20,             # meters
    "ris_element_size": 0.5,    # wavelengths
    "ris_array_size": 400,      # number of elements
    "ris_position": 10,         # meters
    "ris_phase_bits": 4,        # number of bits for phase quantization
}

def generate_params(params=None, path=None):
    path = path or PARAMS_PATH
    params = params if params is not None else DEFAULT_PARAMS.copy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    return path
