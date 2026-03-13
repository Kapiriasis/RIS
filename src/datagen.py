import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_PATH = os.path.join(ROOT, "data", "params.json")

DEFAULT_PARAMS = {
    "distance": 20,
    "frequency": 2.4e9,
    "K_dB": 15,
    "N": 1000,
    "path_loss_exponent": 4.0,
    "bandwidth": 20e6,
}

def generate_params(params=None, path=None):
    path = path or PARAMS_PATH
    params = params if params is not None else DEFAULT_PARAMS.copy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    return path
