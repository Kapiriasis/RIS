import numpy as np


class User:
    """Vehicle/UE moving at constant speed between two base stations."""

    def __init__(self, total_distance: float, speed: float = None, rng=None):
        rng = rng or np.random.default_rng()
        self.total_distance = float(total_distance)
        # Random start position uniformly in [0, total_distance]
        self.position = float(rng.uniform(0.0, total_distance))
        # Random speed in [5, 30] m/s (~18–108 km/h) if not provided
        self.speed = float(speed) if speed is not None else float(rng.uniform(5.0, 30.0))
        # Direction: +1 (toward BS_B) or -1 (toward BS_A)
        self.direction = float(rng.choice([-1.0, 1.0]))

    def step(self, dt: float) -> float:
        """Advance by dt seconds; bounce at boundaries. Returns new position."""
        self.position += self.direction * self.speed * dt
        if self.position >= self.total_distance:
            self.position = self.total_distance
            self.direction = -1.0
        elif self.position <= 0.0:
            self.position = 0.0
            self.direction = 1.0
        return self.position

    @property
    def distance_to_bs_a(self) -> float:
        return self.position

    @property
    def distance_to_bs_b(self) -> float:
        return self.total_distance - self.position
