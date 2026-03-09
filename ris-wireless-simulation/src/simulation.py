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
        snr_list = []

        # Link budget parameters
        tx_power_dbm = input_params.get('tx_power_dbm', 30)  # 1 W default
        tx_power_w = 10 ** ((tx_power_dbm - 30) / 10)
        bandwidth_hz = input_params.get('bandwidth_hz', 10e6)
        noise_figure_db = input_params.get('noise_figure_db', 5)
        noise_figure_linear = 10 ** (noise_figure_db / 10)

        # Thermal noise power in Watts: k T B * NF
        k_boltzmann = 1.38064852e-23  # J/K
        temperature_k = 290.0
        noise_power_linear = k_boltzmann * temperature_k * bandwidth_hz * noise_figure_linear

        # Geometry: TX and RX positions in 3D
        tx_position = np.array(input_params.get('tx_position', [0.0, 0.0, 0.0]))
        # Default RX is placed on x-axis at the sum of TX-RIS and RIS-RX distances
        default_rx_x = input_params['distance_tx_ris'] + input_params['distance_ris_rx']
        rx_param = input_params.get('rx_position', None)
        if rx_param is None:
            rx_position = np.array([default_rx_x, 0.0, 0.0])
        else:
            rx_position = np.array(rx_param)

        # RIS element positions and current RIS center (placement)
        element_positions = self.ris_model.get_element_positions()
        placement = np.array(self.ris_model.placement, dtype=float)

        # Wavelength and wave number
        wavelength = self.channel_model.c / input_params['frequency']
        k = 2 * np.pi / wavelength

        # TX/RX antenna boresights: if null/"auto", point at RIS center so RIS is on boresight
        def _resolve_boresight(param_value, from_pos, to_pos, default_fallback):
            auto = param_value is None or param_value == "auto" or (isinstance(param_value, (list, tuple)) and len(param_value) == 0)
            if auto:
                vec = to_pos - from_pos
                n = np.linalg.norm(vec)
                if n < 1e-9:
                    d = np.array(default_fallback, dtype=float)
                    return d / (np.linalg.norm(d) + 1e-30)
                return vec / n
            v = np.array(param_value, dtype=float)
            n = np.linalg.norm(v)
            return v / (n + 1e-30)

        tx_bor = input_params.get('tx_boresight')
        rx_bor = input_params.get('rx_boresight')
        tx_boresight = _resolve_boresight(tx_bor, tx_position, placement, [1.0, 0.0, 0.0])
        rx_boresight = _resolve_boresight(rx_bor, rx_position, placement, [-1.0, 0.0, 0.0])
        pattern_exponent = float(input_params.get('pattern_exponent', 1.0))

        # Mutual coupling: strength alpha, decay distance d0 (meters)
        use_coupling = input_params.get('use_mutual_coupling', True)
        coupling_strength = float(input_params.get('coupling_strength', 0.05))
        coupling_decay = float(input_params.get('coupling_decay', 1.0))

        def antenna_gain(unit_direction, boresight, expo):
            cos_theta = np.abs(np.dot(np.asarray(unit_direction), np.asarray(boresight)))
            return np.maximum(1e-6, cos_theta) ** expo

        for _ in range(num_simulations):
            num_elements = self.ris_model.num_elements

            # Direct TX->RX path
            d_tx_rx = np.linalg.norm(rx_position - tx_position)
            direct_path_loss_exponent = input_params.get('direct_path_loss_exponent') or input_params['path_loss_exponent']
            pl_tx_rx = self.channel_model.calculate_path_loss_linear(
                input_params['frequency'], d_tx_rx, direct_path_loss_exponent
            )
            direct_path_loss_factor = input_params.get('direct_path_loss_factor', 1.0)
            pl_tx_rx = pl_tx_rx * direct_path_loss_factor
            amp_tx_rx = 1 / np.sqrt(pl_tx_rx)
            fading_tx_rx = self.channel_model.generate_fading(
                input_params['fading_type'], 1
            )[0]
            phase_tx_rx = np.exp(-1j * k * d_tx_rx)
            direct_signal = amp_tx_rx * fading_tx_rx * phase_tx_rx
            # TX/RX antenna patterns on direct path
            if input_params.get('use_tx_rx_pattern', True):
                dir_tx_to_rx = (rx_position - tx_position) / (d_tx_rx + 1e-30)
                dir_rx_from_tx = (tx_position - rx_position) / (d_tx_rx + 1e-30)
                g_tx_d = antenna_gain(dir_tx_to_rx, tx_boresight, pattern_exponent)
                g_rx_d = antenna_gain(dir_rx_from_tx, rx_boresight, pattern_exponent)
                direct_signal = direct_signal * np.sqrt(g_tx_d * g_rx_d)

            # Distances from TX to each element and from each element to RX
            d_tx_ris = np.linalg.norm(element_positions - tx_position, axis=1)
            d_ris_rx = np.linalg.norm(element_positions - rx_position, axis=1)

            pl_tx_ris = self.channel_model.calculate_path_loss_linear(
                input_params['frequency'], d_tx_ris, input_params['path_loss_exponent']
            )
            pl_ris_rx = self.channel_model.calculate_path_loss_linear(
                input_params['frequency'], d_ris_rx, input_params['path_loss_exponent']
            )
            amp_tx_ris = 1 / np.sqrt(pl_tx_ris)
            amp_ris_rx = 1 / np.sqrt(pl_ris_rx)

            fading_tx_ris = self.channel_model.generate_fading(
                input_params['fading_type'], num_elements
            )
            fading_ris_rx = self.channel_model.generate_fading(
                input_params['fading_type'], num_elements
            )
            total_distance = d_tx_ris + d_ris_rx
            propagation_phase = np.exp(-1j * k * total_distance)
            h_elements = (
                amp_tx_ris * fading_tx_ris * amp_ris_rx * fading_ris_rx * propagation_phase
            )

            # Incident/reflected directions and cos(θ) w.r.t. RIS normal (+z) for each element
            vec_inc = (element_positions - tx_position) / (d_tx_ris[:, np.newaxis] + 1e-30)
            vec_refl = (rx_position - element_positions) / (d_ris_rx[:, np.newaxis] + 1e-30)
            normal = np.array([0.0, 0.0, 1.0])
            cos_inc = np.dot(vec_inc, normal)
            cos_refl = np.dot(vec_refl, normal)

            # Simple element pattern (cosine): power ∝ |cos(θ_inc)|*|cos(θ_refl)|, RIS normal +z
            if input_params.get('use_element_pattern', True):
                pattern = np.sqrt(np.maximum(1e-6, np.abs(cos_inc) * np.abs(cos_refl)))
                h_elements = h_elements * pattern

            # TX/RX antenna patterns on RIS path (per element)
            if input_params.get('use_tx_rx_pattern', True):
                dir_tx_to_ris = (element_positions - tx_position) / (d_tx_ris[:, np.newaxis] + 1e-30)
                dir_rx_from_ris = (element_positions - rx_position) / (d_ris_rx[:, np.newaxis] + 1e-30)
                g_tx_ris = np.maximum(1e-6, np.abs(np.dot(dir_tx_to_ris, tx_boresight))) ** pattern_exponent
                g_rx_ris = np.maximum(1e-6, np.abs(np.dot(dir_rx_from_ris, rx_boresight))) ** pattern_exponent
                h_elements = h_elements * np.sqrt(g_tx_ris * g_rx_ris)

            # Phase optimisation at the RIS
            optimal_phases = -np.angle(h_elements)
            self.ris_model.set_phase_reflections(optimal_phases)
            reflection_coeffs = self.ris_model.calculate_reflection_coefficient(
                cos_incident=cos_inc, cos_reflected=cos_refl
            )
            element_signals = h_elements * reflection_coeffs

            # Mutual coupling between RIS elements: s_coupled = (I + alpha*K) @ s, K_ij = exp(-d_ij/d0)
            if use_coupling and num_elements > 1 and coupling_strength != 0:
                d_ij = np.linalg.norm(element_positions[:, np.newaxis, :] - element_positions[np.newaxis, :, :], axis=2)
                np.fill_diagonal(d_ij, np.inf)
                K = np.exp(-d_ij / (coupling_decay + 1e-30))
                coupling_matrix = np.eye(num_elements) + coupling_strength * K
                element_signals = coupling_matrix @ element_signals

            combined_signal = direct_signal + np.sum(element_signals)

            # Received power is the squared magnitude of the combined complex signal
            # scaled by transmit power
            received_power = (np.abs(combined_signal) ** 2) * tx_power_w
            snr_linear = received_power / noise_power_linear
            snr_list.append(snr_linear)
        # Average in linear then convert to dB for a stable, smooth curve
        return 10 * np.log10(np.mean(snr_list) + 1e-30)

    def analyze_results(self, results):
        snr_values = results.get('snr_values', [])
        if len(snr_values) < 2:
            print("Not enough SNR points to compute difference (need at least 2).")
            return
        snr_difference = snr_values[-1] - snr_values[0]
        print(f"SNR Difference: {snr_difference:.2f} dB")
