import json
import matplotlib.pyplot as plt
import os

# Single source of default parameters (used when creating a new config file and for setdefault)
DEFAULT_PARAMS = {
    "num_simulations": 1000,
    # RIS geometry and configuration
    "element_size": 0.5,                # base element size (e.g. in wavelengths or meters)
    "placement": [0, 0, 5],             # default RIS center position [x, y, z]
    "num_elements": 200,                # maximum number of RIS elements
    "layout": "2d",                     # "1d" or "2d" RIS layout
    # RIS hardware non-idealities
    "reflection_amplitude": 0.9,
    "reflection_angle_exponent": 0.5,   # angle-dependent |Γ|: (|cos_inc|*|cos_refl|)^exponent; 0 = off
    "phase_quantization_bits": 2,
    "phase_error_std": 0.0,
    # Mutual coupling
    "use_mutual_coupling": True,        # mutual coupling between RIS elements
    "coupling_strength": 0.05,          # coupling matrix scale (alpha)
    "coupling_decay": 1.0,              # decay distance in meters (d0)
    # Element and antenna patterns
    "use_element_pattern": True,        # cos(θ) element pattern (False = isotropic)
    "pattern_exponent": 1.0,            # gain = |cos(θ)|^pattern_exponent
    # Optional sweeps for analysis
    "element_size_list": [0.25, 0.5, 1.0],
    "placement_list": [[0, 0, 5], [4, 0, 5], [8, 0, 5], [12, 0, 5], [16, 0, 5], [20, 0, 5]],
    # Channel and noise parameters
    "frequency": 2.4e9,
    "fading_type": "Rayleigh",
    "path_loss_exponent": 2.0,          # for RIS hops (TX-RIS, RIS-RX)
    "direct_path_loss_exponent": 2.0,   # for direct TX-RX (e.g. higher if blocked/NLOS)
    "direct_path_loss_factor": 1,       # extra blockage factor on direct (>=1)
    # Transmitter / receiver geometry
    "tx_position": [0, 0, 0],
    "rx_position": None,
    "distance_tx_ris": 10,
    "distance_ris_rx": 10,
    # Transmitter / receiver parameters
    "tx_power_dbm": 30,
    "bandwidth_hz": 10e6,
    "noise_figure_db": 5,
    "use_tx_rx_pattern": True,          # TX/RX antenna patterns (False = isotropic)
    "tx_boresight": None,               # null = point at RIS center; or [x,y,z] unit vector
    "rx_boresight": None,               # null = point at RIS center; or [x,y,z] unit vector
    "random_seed": None,                # integer for reproducible runs; null = do not set
}

def load_input_parameters(file_path):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(DEFAULT_PARAMS, f, indent=4)
        print(f"Created default {file_path}")
        return dict(DEFAULT_PARAMS)

    try:
        with open(file_path, 'r') as f:
            params = json.load(f)

        for key, default in DEFAULT_PARAMS.items():
            if key == "direct_path_loss_exponent":
                params.setdefault(key, params.get("path_loss_exponent", 2.0))
            else:
                params.setdefault(key, default)

        # Keep distances consistent with geometry when positions are provided
        try:
            import numpy as np
            tx_pos = np.array(params.get("tx_position", [0, 0, 0]), dtype=float)
            ris_pos = np.array(params.get("placement", [0, 0, 5]), dtype=float)
            params["distance_tx_ris"] = float(np.linalg.norm(ris_pos - tx_pos))
            rx_pos = params.get("rx_position", None)
            if rx_pos is not None:
                rx_pos = np.array(rx_pos, dtype=float)
                params["distance_ris_rx"] = float(np.linalg.norm(rx_pos - ris_pos))
        except Exception:
            pass

        return params
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure the file exists.")
        return None


def _plot_one(x_data, y_data, xlabel, ylabel, title, filepath):
    # Create one figure, plot, save, show, and close.
    plt.figure(figsize=(12, 8))
    plt.plot(x_data, y_data, label='SNR', marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath)
    print(f"Plot saved to {filepath}")
    plt.show()
    plt.close()


def plot_results(results):
    print("Plotting results...")
    os.makedirs('results', exist_ok=True)

    if 'num_elements_list' in results and 'snr_values' in results:
        _plot_one(
            results['num_elements_list'],
            results['snr_values'],
            'Number of Cells',
            'SNR (dB)',
            'SNR vs RIS Resolution (Number of Elements)',
            'results/snr_vs_elements.png',
        )

    if 'element_size_list' in results and 'snr_element_size' in results:
        _plot_one(
            results['element_size_list'],
            results['snr_element_size'],
            'Element Size (wavelengths)',
            'SNR (dB)',
            'SNR vs RIS Element Size',
            'results/snr_vs_element_size.png',
        )

    if 'placement_list' in results and 'snr_placement' in results:
        x_coords = [
            p[0] if isinstance(p, (list, tuple)) and len(p) > 0 else 0
            for p in results['placement_list']
        ]
        _plot_one(
            x_coords,
            results['snr_placement'],
            'RIS x-position',
            'SNR (dB)',
            'SNR vs RIS Placement (x-axis)',
            'results/snr_vs_placement_x.png',
        )
