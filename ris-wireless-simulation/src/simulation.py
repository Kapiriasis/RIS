import numpy as np

class Simulation:
    def __init__(self, channel_model, ris_model):
        self.channel_model = channel_model
        self.ris_model = ris_model

    def run_simulation(self, input_params):
        results = {}

        # 1) Sweep over number of RIS elements (existing functionality)
        results['num_elements_list'] = []
        results['snr_values'] = []
        for num in np.linspace(1, input_params['num_elements'], 50):  # More points for better resolution
            num = int(num)
            if num <= 0:
                continue
            self.ris_model.set_num_elements(num)
            snr = self.simulate_single_run(input_params)
            results['num_elements_list'].append(num)
            results['snr_values'].append(snr)
            print(f"Simulated num_elements: {num}, SNR: {snr:.2f} dB")

        # 2) Optional sweep over RIS element sizes
        element_size_list = input_params.get('element_size_list')
        if element_size_list is not None:
            results['element_size_list'] = []
            results['snr_element_size'] = []
            # Fix number of elements to the maximum specified in input
            self.ris_model.set_num_elements(input_params['num_elements'])
            for size in element_size_list:
                self.ris_model.set_element_size(size)
                snr = self.simulate_single_run(input_params)
                results['element_size_list'].append(size)
                results['snr_element_size'].append(snr)
                print(f"Simulated element_size: {size}, SNR: {snr:.2f} dB")

        # 3) Optional sweep over RIS placements
        placement_list = input_params.get('placement_list')
        if placement_list is not None:
            results['placement_list'] = []
            results['snr_placement'] = []
            # Fix number of elements and element size to defaults
            self.ris_model.set_num_elements(input_params['num_elements'])
            self.ris_model.set_element_size(input_params.get('element_size', 0.5))
            for placement in placement_list:
                self.ris_model.set_placement(placement)
                snr = self.simulate_single_run(input_params)
                results['placement_list'].append(placement)
                results['snr_placement'].append(snr)
                print(f"Simulated placement: {placement}, SNR: {snr:.2f} dB")

        return results

    def simulate_single_run(self, input_params):
        num_simulations = input_params['num_simulations']
        noise_power = input_params.get('noise_power', 1)  # in dBm. Default to 1 if not provided
        snr_list = []

        # Geometry: TX and RX positions in 3D
        tx_position = np.array(input_params.get('tx_position', [0.0, 0.0, 0.0]))
        # Default RX is placed on x-axis at the sum of TX-RIS and RIS-RX distances
        default_rx_x = input_params['distance_tx_ris'] + input_params['distance_ris_rx']
        rx_param = input_params.get('rx_position', None)
        if rx_param is None:
            rx_position = np.array([default_rx_x, 0.0, 0.0])
        else:
            rx_position = np.array(rx_param)

        # RIS element positions (1D linear array centered around RIS placement)
        element_positions = self.ris_model.get_element_positions()

        # Wavelength and wave number
        wavelength = self.channel_model.c / input_params['frequency']
        k = 2 * np.pi / wavelength

        for _ in range(num_simulations):
            num_elements = self.ris_model.num_elements

            # Distances from TX to each element and from each element to RX
            d_tx_ris = np.linalg.norm(element_positions - tx_position, axis=1)
            d_ris_rx = np.linalg.norm(element_positions - rx_position, axis=1)

            # Path loss per hop (power domain) for each element
            pl_tx_ris = self.channel_model.calculate_path_loss_linear(
                input_params['frequency'], d_tx_ris, input_params['path_loss_exponent']
            )
            pl_ris_rx = self.channel_model.calculate_path_loss_linear(
                input_params['frequency'], d_ris_rx, input_params['path_loss_exponent']
            )

            # Convert path loss to amplitude attenuation for each hop (per element)
            amp_tx_ris = 1 / np.sqrt(pl_tx_ris)
            amp_ris_rx = 1 / np.sqrt(pl_ris_rx)

            # Fading per element for each hop
            fading_tx_ris = self.channel_model.generate_fading(
                input_params['fading_type'], num_elements
            )
            fading_ris_rx = self.channel_model.generate_fading(
                input_params['fading_type'], num_elements
            )

            # Baseband complex channel per element without RIS phase control:
            # includes amplitude, fading, and propagation phase for TX->RIS->RX
            total_distance = d_tx_ris + d_ris_rx
            propagation_phase = np.exp(-1j * k * total_distance)
            h_elements = (
                amp_tx_ris * fading_tx_ris * amp_ris_rx * fading_ris_rx * propagation_phase
            )

            # Phase optimisation at the RIS:
            # choose reflection phase so that each element's contribution is aligned at RX
            optimal_phases = -np.angle(h_elements)
            self.ris_model.set_phase_reflections(optimal_phases)
            reflection_coeffs = self.ris_model.calculate_reflection_coefficient(0, 0)

            # Effective per-element signals after RIS phase control
            element_signals = h_elements * reflection_coeffs

            # Coherent summation of all optimized element contributions at the RX
            combined_signal = np.sum(element_signals)

            # Received power is the squared magnitude of the combined complex signal
            received_power = np.abs(combined_signal) ** 2
            noise_power_linear = 10**(noise_power/10)
            snr_linear = received_power / noise_power_linear
            snr_db = 10 * np.log10(snr_linear)
            snr_list.append(snr_db)
        return np.mean(snr_list)  # Average SNR over simulations

    def analyze_results(self, results):
        # Basic analysis: difference between initial and final SNR
        snr_difference = results['snr_values'][-1] - results['snr_values'][0]
        print(f"SNR Difference: {snr_difference:.2f} dB")
