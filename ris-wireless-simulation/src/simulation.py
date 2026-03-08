import numpy as np

class Simulation:
    def __init__(self, channel_model, ris_model):
        self.channel_model = channel_model
        self.ris_model = ris_model

    def run_simulation(self, input_params):
        results = {
            'num_elements_list': [],
            'snr_values': []
        }
        # Simulate different number of elements (higher resolution)
        for num in np.linspace(1, input_params['num_elements'], 50):  # More points for better resolution
            num = int(num)
            self.ris_model.set_num_elements(num)
            snr = self.simulate_single_run(input_params)
            results['num_elements_list'].append(num)
            results['snr_values'].append(snr)
            print(f"Simulated num_elements: {num}, SNR: {snr:.2f} dB")
        return results

    def simulate_single_run(self, input_params):
        num_simulations = input_params['num_simulations']
        noise_power = input_params.get('noise_power', 1)  # in dBm. Default to 1 if not provided
        snr_list = []
        for _ in range(num_simulations):
            # Calculate path losses in linear scale
            path_loss_tx_ris_linear = self.channel_model.calculate_path_loss_linear(
                input_params['frequency'], input_params['distance_tx_ris'], input_params['path_loss_exponent']
            )
            path_loss_ris_rx_linear = self.channel_model.calculate_path_loss_linear(
                input_params['frequency'], input_params['distance_ris_rx'], input_params['path_loss_exponent']
            )
            fading = self.channel_model.generate_fading(input_params['fading_type'], 1)[0]
            # RIS effect: sum of reflections (depends on number of elements)
            ris_gain = np.sum(self.ris_model.calculate_reflection_coefficient(0, 0))  # Complex sum
            total_path_loss_linear = path_loss_tx_ris_linear * path_loss_ris_rx_linear
            # SNR calculation: received power / noise power
            received_power = 1 / (total_path_loss_linear * fading / abs(ris_gain)**2)  # Assuming P_tx = 1
            noise_power_linear = 10**(noise_power/10)
            snr_linear = received_power / noise_power_linear
            snr_db = 10 * np.log10(snr_linear)
            snr_list.append(snr_db)
        return np.mean(snr_list)  # Average SNR over simulations

    def analyze_results(self, results):
        # Basic analysis: difference between initial and final SNR
        snr_difference = results['snr_values'][-1] - results['snr_values'][0]
        print(f"SNR Difference: {snr_difference:.2f} dB")
