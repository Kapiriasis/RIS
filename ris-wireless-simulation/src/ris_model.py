import numpy as np

class RISModel:
    def __init__(self):
        self.element_size = 0.5  # Default
        self.placement = [0, 0, 5]  # Default position
        self.num_elements = 100
        self.phases = np.zeros(self.num_elements)  # Phase shifts (radians)
        self.layout = "2d"  # "1d" = linear array along x; "2d" = rectangular grid in x-y plane
        # Non-ideal RIS parameters
        self.reflection_amplitude = 0.9  # |Γ| < 1 to model losses
        self.phase_quantization_bits = 2  # e.g. 2-bit phase shifters
        self.phase_error_std = 0.0  # random phase error (radians)
        # Angle-dependent reflection: |Γ| scaled by (|cos_inc|*|cos_refl|)^reflection_angle_exponent
        self.reflection_angle_exponent = 0.5  # 0 = angle-independent; 0.5 or 1 = common choices

    def set_element_size(self, size):
        self.element_size = size

    def set_placement(self, placement):
        self.placement = placement

    def set_layout(self, layout):
        """Set layout: '1d' (linear along x) or '2d' (rectangular grid in x-y plane, normal +z)."""
        self.layout = str(layout).lower()

    def set_num_elements(self, num):
        self.num_elements = num
        self.phases = np.zeros(num)

    def set_reflection_amplitude(self, amplitude):
        # Set common amplitude of the reflection coefficient (0 < amplitude <= 1).
        self.reflection_amplitude = float(amplitude)

    def set_phase_quantization_bits(self, bits):
        # Set number of quantization bits for the RIS phase shifters.
        self.phase_quantization_bits = int(bits) if bits is not None else None

    def set_phase_error_std(self, std_rad):
        # Set standard deviation of random phase error (in radians).
        self.phase_error_std = float(std_rad)

    def set_reflection_angle_exponent(self, exponent):
        # Exponent for angle-dependent reflection amplitude: (|cos_inc|*|cos_refl|)^exponent.
        self.reflection_angle_exponent = float(exponent)

    def set_phase_reflections(self, phases):
        self.phases = np.array(phases)
        # self.phases = np.random.uniform(0, 2 * np.pi, self.num_elements)

    def calculate_reflection_coefficient(self, cos_incident=None, cos_reflected=None):
        """
        Calculate complex reflection coefficients including non-ideal RIS effects:
        - Common amplitude |Γ| < 1 (reflection_amplitude)
        - Angle-dependent amplitude: (|cos_inc|*|cos_refl|)^reflection_angle_exponent when angles given
        - Phase quantization to finite bits
        - Optional random phase error

        cos_incident, cos_reflected: arrays of shape (num_elements,) or None.
        If both provided, amplitude is scaled per element by (|cos_inc|*|cos_refl|)^reflection_angle_exponent.
        """
        phases = np.array(self.phases, copy=True)

        # Phase quantization
        if self.phase_quantization_bits is not None and self.phase_quantization_bits > 0:
            levels = 2 ** self.phase_quantization_bits
            step = 2 * np.pi / levels
            phases = np.round(phases / step) * step

        # Random phase error
        if self.phase_error_std > 0:
            phases = phases + np.random.normal(0.0, self.phase_error_std, size=self.num_elements)

        amplitude = self.reflection_amplitude * np.ones(self.num_elements, dtype=float)
        if (
            self.reflection_angle_exponent != 0
            and cos_incident is not None
            and cos_reflected is not None
        ):
            cos_inc = np.asarray(cos_incident, dtype=float).ravel()
            cos_refl = np.asarray(cos_reflected, dtype=float).ravel()
            product = np.maximum(1e-30, np.abs(cos_inc) * np.abs(cos_refl))
            amplitude = amplitude * (product ** self.reflection_angle_exponent)

        reflection_coeff = amplitude * np.exp(1j * phases)
        return reflection_coeff

    def get_element_positions(self):
        """
        Return Cartesian positions of all RIS elements.

        - If layout == "1d": uniform linear array along the x-axis, centered at placement.
        - If layout == "2d": rectangular grid in the x-y plane (z = placement[2]),
          normal +z; grid as square as possible, row-major order, centered at placement.
        """
        placement = np.asarray(self.placement, dtype=float)
        n = self.num_elements
        d = self.element_size

        if self.layout == "2d":
            n_cols = max(1, int(np.ceil(np.sqrt(n))))
            n_rows = max(1, int(np.ceil(n / n_cols)))
            positions = np.zeros((n, 3))
            idx = 0
            for row in range(n_rows):
                for col in range(n_cols):
                    if idx >= n:
                        break
                    # Center grid: (0,0) at center of grid
                    x = placement[0] + (col - (n_cols - 1) / 2.0) * d
                    y = placement[1] + (row - (n_rows - 1) / 2.0) * d
                    positions[idx, 0] = x
                    positions[idx, 1] = y
                    positions[idx, 2] = placement[2]
                    idx += 1
            return positions

        # 1d: linear array along x
        positions = np.zeros((n, 3))
        indices = np.arange(n)
        offsets_x = (indices - (n - 1) / 2.0) * d
        positions[:, 0] = placement[0] + offsets_x
        positions[:, 1] = placement[1]
        positions[:, 2] = placement[2]
        return positions
