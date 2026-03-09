import json
import matplotlib.pyplot as plt
import os

def load_input_parameters(file_path):
    if not os.path.exists(file_path):
        # Create default parameters if file doesn't exist
        default_params = {
            "num_simulations": 1000,
            # RIS geometry and configuration
            "element_size": 0.5,              # base element size (e.g. in wavelengths or meters)
            "placement": [0, 0, 5],           # default RIS center position [x, y, z]
            "num_elements": 200,              # maximum number of RIS elements
            "layout": "2d",                   # "1d" or "2d" RIS layout
            # RIS hardware non-idealities
            "reflection_amplitude": 0.9,
            "reflection_angle_exponent": 0.5,  # angle-dependent |Γ|: (|cos_inc|*|cos_refl|)^exponent; 0 = off
            "phase_quantization_bits": 2,
            "phase_error_std": 0.0,

            "use_mutual_coupling": True,       # mutual coupling between RIS elements
            "coupling_strength": 0.05,         # coupling matrix scale (alpha)
            "coupling_decay": 1.0,             # decay distance in meters (d0)

            "use_element_pattern": True,      # cos(θ) element pattern (False = isotropic)
            "pattern_exponent": 1.0,          # gain = |cos(θ)|^pattern_exponent
            # Optional sweeps for analysis
            "element_size_list": [0.25, 0.5, 1.0],
            "placement_list": [
                [0, 0, 5],
                [4, 0, 5],
                [8, 0, 5],
                [12, 0, 5],
                [16, 0, 5],
                [20, 0, 5]
            ],
            # Channel and noise parameters
            "frequency": 2.4e9,
            "fading_type": "Rayleigh",

            "path_loss_exponent": 2.0,        # for RIS hops (TX-RIS, RIS-RX)
            "direct_path_loss_exponent": 2.0, # for direct TX-RX (e.g. higher if blocked/NLOS)
            "direct_path_loss_factor": 1,   # extra blockage factor on direct (>=1)
            # Transmitter / receiver geometry
            "tx_position": [0, 0, 0],
            "rx_position": None,
            "distance_tx_ris": 10,
            "distance_ris_rx": 10,
            # Transmitter / receiver parameters
            "tx_power_dbm": 30,
            "bandwidth_hz": 10e6,
            "noise_figure_db": 5,

            "use_tx_rx_pattern": True,        # TX/RX antenna patterns (False = isotropic)
            "tx_boresight": None,             # null = point at RIS center; or [x,y,z] unit vector
            "rx_boresight": None              # null = point at RIS center; or [x,y,z] unit vector
        }
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(default_params, file, indent=4)
        print(f"Created default {file_path}")
        return default_params
    try:
        with open(file_path, 'r') as file:
            params = json.load(file)

        params.setdefault("element_size", 0.5)
        params.setdefault("placement", [0, 0, 5])
        params.setdefault("num_elements", 200)
        params.setdefault("reflection_amplitude", 0.9)
        params.setdefault("phase_quantization_bits", 2)
        params.setdefault("phase_error_std", 0.0)
        params.setdefault("reflection_angle_exponent", 0.5)
        params.setdefault("frequency", 2.4e9)
        params.setdefault("distance_tx_ris", 10)
        params.setdefault("distance_ris_rx", 10)
        params.setdefault("num_simulations", 1000)
        params.setdefault("fading_type", "Rayleigh")
        params.setdefault("path_loss_exponent", 2.0)
        params.setdefault("direct_path_loss_exponent", params.get("path_loss_exponent", 2.0))
        params.setdefault("direct_path_loss_factor", 1)
        params.setdefault("layout", "2d")
        params.setdefault("use_element_pattern", True)
        params.setdefault("use_tx_rx_pattern", True)
        params.setdefault("tx_boresight", None)
        params.setdefault("rx_boresight", None)
        params.setdefault("pattern_exponent", 1.0)
        params.setdefault("use_mutual_coupling", True)
        params.setdefault("coupling_strength", 0.05)
        params.setdefault("coupling_decay", 1.0)
        params.setdefault("tx_power_dbm", 30)
        params.setdefault("bandwidth_hz", 10e6)
        params.setdefault("noise_figure_db", 5)
        params.setdefault("tx_position", [0, 0, 0])
        params.setdefault("rx_position", None)
        params.setdefault("element_size_list", None)
        params.setdefault("placement_list", None)

        # Keep scalar distances consistent with geometry if both positions are provided
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

def plot_results(results):
    print("Plotting results...")
    os.makedirs('results', exist_ok=True)

    if 'num_elements_list' in results and 'snr_values' in results:
        plt.figure(figsize=(12, 8))
        plt.plot(results['num_elements_list'], results['snr_values'], label='SNR', marker='o')
        plt.xlabel('Number of Cells')
        plt.ylabel('SNR (dB)')
        plt.title('SNR vs RIS Resolution (Number of Elements)')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/snr_vs_elements.png')
        print("Plot saved to results/snr_vs_elements.png")
        plt.show()
        plt.close()

    if 'element_size_list' in results and 'snr_element_size' in results:
        plt.figure(figsize=(12, 8))
        plt.plot(results['element_size_list'], results['snr_element_size'], label='SNR', marker='o')
        plt.xlabel('Element Size (wavelengths)')
        plt.ylabel('SNR (dB)')
        plt.title('SNR vs RIS Element Size')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/snr_vs_element_size.png')
        print("Plot saved to results/snr_vs_element_size.png")
        plt.show()
        plt.close()

    if 'placement_list' in results and 'snr_placement' in results:
        x_coords = [p[0] if isinstance(p, (list, tuple)) and len(p) > 0 else 0 for p in results['placement_list']]
        plt.figure(figsize=(12, 8))
        plt.plot(x_coords, results['snr_placement'], label='SNR', marker='o')
        plt.xlabel('RIS x-position')
        plt.ylabel('SNR (dB)')
        plt.title('SNR vs RIS Placement (x-axis)')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/snr_vs_placement_x.png')
        print("Plot saved to results/snr_vs_placement_x.png")
        plt.show()
        plt.close()
