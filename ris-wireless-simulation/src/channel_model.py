import numpy as np

class ChannelModel:
    def __init__(self):
        self.c = 3e8  # Speed of light

    def calculate_path_loss(self, frequency, distance, path_loss_exponent):
        # Friis path loss formula in dB, adjusted for path_loss_exponent
        lambda_ = self.c / frequency
        base_loss_db = 20 * np.log10( (4 * np.pi * distance / lambda_)**2 )
        additional_loss_db = 10 * (path_loss_exponent - 2) * np.log10(distance) if distance > 1 else 0
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
        if fading_type == "Rayleigh":
            # Rayleigh fading
            return np.random.rayleigh(1, num_samples)
        elif fading_type == "Rician":
            # Rician fading (K=1)
            return np.random.rice(1, num_samples)
        else:
            return np.ones(num_samples)  # No fading
