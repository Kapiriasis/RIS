import json
import matplotlib.pyplot as plt
import os

def load_input_parameters(file_path):
    if not os.path.exists(file_path):
        # Create default parameters if file doesn't exist
        default_params = {
            "element_size": 0.5,
            "placement": [0, 0, 5],
            "num_elements": 200,
            "frequency": 2.4e9,
            "distance_tx_ris": 10,
            "distance_ris_rx": 10,
            "num_simulations": 10000,
            "fading_type": "Rayleigh",
            "path_loss_exponent": 2.0,
            "noise_power": 1e-10
        }
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(default_params, file, indent=4)
        print(f"Created default {file_path}")
        return default_params
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure the file exists.")
        return None

def plot_results(results):
    print("Plotting results...")
    # Plot: SNR vs Number of Elements
    plt.figure(figsize=(12, 8))  # Larger figure for better resolution
    plt.plot(results['num_elements_list'], results['snr_values'], label='SNR', marker='o')
    plt.xlabel('Number of Cells')
    plt.ylabel('SNR (dB)')
    plt.title('SNR vs RIS Resolution')
    plt.legend()
    plt.grid(True)
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/snr_vs_elements.png')  # Save plot to file
    print("Plot saved to results/snr_vs_elements.png")
    plt.show()
    plt.close()
