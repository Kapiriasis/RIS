import json
import matplotlib.pyplot as plt
import os

def load_input_parameters(file_path):
    if not os.path.exists(file_path):
        # Create default parameters if file doesn't exist
        default_params = {
            # RIS geometry and configuration
            "element_size": 0.5,              # base element size (e.g. in wavelengths or meters)
            "placement": [0, 0, 5],           # default RIS center position [x, y, z]
            "num_elements": 200,              # maximum number of RIS elements
            # Optional sweeps for analysis of Part 1
            # If you don't want to sweep these, you can remove or set them to null in the JSON.
            "element_size_list": [0.25, 0.5, 1.0],
            "placement_list": [
                [0, 0, 5],
                [5, 0, 5],
                [10, 0, 5]
            ],
            # Transmitter / receiver geometry
            "tx_position": [0, 0, 0],
            # Default RX on x-axis at distance_tx_ris + distance_ris_rx if not overridden
            "rx_position": None,
            # Channel parameters
            "frequency": 2.4e9,
            "distance_tx_ris": 10,
            "distance_ris_rx": 10,
            "num_simulations": 1000,
            "fading_type": "Rayleigh",
            "path_loss_exponent": 2.0,
            "noise_power": -90
        }
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(default_params, file, indent=4)
        print(f"Created default {file_path}")
        return default_params
    try:
        with open(file_path, 'r') as file:
            params = json.load(file)

        # Backwards-compatible defaults so older JSON files still work
        params.setdefault("element_size", 0.5)
        params.setdefault("placement", [0, 0, 5])
        params.setdefault("num_elements", 200)
        params.setdefault("frequency", 2.4e9)
        params.setdefault("distance_tx_ris", 10)
        params.setdefault("distance_ris_rx", 10)
        params.setdefault("num_simulations", 1000)
        params.setdefault("fading_type", "Rayleigh")
        params.setdefault("path_loss_exponent", 2.0)
        params.setdefault("noise_power", -90)

        # New geometry / sweep parameters for RIS analysis
        params.setdefault("tx_position", [0, 0, 0])
        # If rx_position is missing or null, Simulation will fall back to the default on x-axis
        params.setdefault("rx_position", None)
        # Optional sweeps
        params.setdefault("element_size_list", None)
        params.setdefault("placement_list", None)

        # Keep scalar distances consistent with geometry if both positions are provided
        try:
            tx_pos = params.get("tx_position", [0, 0, 0])
            ris_pos = params.get("placement", [0, 0, 5])
            rx_pos = params.get("rx_position", None)

            import numpy as np

            tx_pos = np.array(tx_pos, dtype=float)
            ris_pos = np.array(ris_pos, dtype=float)
            params["distance_tx_ris"] = float(np.linalg.norm(ris_pos - tx_pos))

            if rx_pos is not None:
                rx_pos = np.array(rx_pos, dtype=float)
                params["distance_ris_rx"] = float(np.linalg.norm(rx_pos - ris_pos))
        except Exception:
            # If anything goes wrong, fall back to whatever was in the JSON
            pass

        return params
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure the file exists.")
        return None

def plot_results(results):
    print("Plotting results...")
    os.makedirs('results', exist_ok=True)

    # 1) Plot: SNR vs Number of Elements
    if 'num_elements_list' in results and 'snr_values' in results:
        plt.figure(figsize=(12, 8))  # Larger figure for better resolution
        plt.plot(results['num_elements_list'], results['snr_values'], label='SNR', marker='o')
        plt.xlabel('Number of Cells')
        plt.ylabel('SNR (dB)')
        plt.title('SNR vs RIS Resolution (Number of Elements)')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/snr_vs_elements.png')  # Save plot to file
        print("Plot saved to results/snr_vs_elements.png")
        plt.show()
        plt.close()

    # 2) Optional: Plot SNR vs Element Size
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

    # 3) Optional: Plot SNR vs Placement (x-coordinate)
    if 'placement_list' in results and 'snr_placement' in results:
        # For visualization, use the x-coordinate of each placement
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
