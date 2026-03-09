import numpy as np

class ChannelModel:
    def __init__(self):
        self.c = 3e8  # Speed of light

    def calculate_path_loss(self, frequency, distance, path_loss_exponent):
        # Path loss in dB for scalar or array distances (array-safe).
        lambda_ = self.c / frequency
        distance = np.asarray(distance, dtype=float)
        base_loss_db = 20 * np.log10(np.maximum((4 * np.pi * distance / lambda_) ** 2, 1e-30))
        additional_loss_db = np.where(
            distance > 1,
            10 * (path_loss_exponent - 2) * np.log10(distance),
            0.0
        )
        return base_loss_db + additional_loss_db  # dB

    def calculate_path_loss_linear(self, frequency, distance, path_loss_exponent):
        """
        Linear path loss for scalar or array distances.

        Uses a generalized Friis-like model:
            PL(d) = (4 * pi * d / lambda)^2 * d^(path_loss_exponent - 2),  d > 1
            PL(d) = (4 * pi * d / lambda)^2,                               d <= 1
        """
        lambda_ = self.c / frequency
        distance = np.asarray(distance)
        base_pl = (4 * np.pi * distance / lambda_) ** 2
        # Handle both scalar and array distances without ambiguous truth value
        additional_loss = np.where(
            distance > 1, distance ** (path_loss_exponent - 2), 1.0
        )
        return base_pl * additional_loss  # linear

    def generate_fading(self, fading_type, num_samples):
        """
        Generate complex fading coefficients (unit average power E[|h|^2]=1).
        Standard Rayleigh: complex Gaussian, so random amplitude and random phase.
        """
        if fading_type == "Rayleigh":
            # Complex Rayleigh: (X + j*Y)/sqrt(2), X,Y i.i.d. N(0,1) => E[|h|^2]=1
            return (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2)
        elif fading_type == "Rician":
            # Complex Rician (K=1): deterministic + scattered, E[|h|^2]=1
            K = 1.0
            los = np.sqrt(K / (1 + K))
            scatter = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2 * (1 + K))
            return los + scatter
        else:
            return np.ones(num_samples, dtype=complex)  # No fading
