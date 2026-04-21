import numpy as np


class User:
    """
    2-D random waypoint user inside a rectangular area [0,W] x [0,H].
    Picks a random destination, travels at constant speed, picks the next
    destination on arrival.  The wall obstacle is LoS-only — it does not
    restrict movement, so no avoidance logic is needed.
    """

    def __init__(
        self,
        area_width: float,
        area_height: float,
        speed: float,
        rng: np.random.Generator = None,
    ):
        self.W     = float(area_width)
        self.H     = float(area_height)
        self.speed = float(speed)
        self._rng  = rng if rng is not None else np.random.default_rng()

        self.x, self.y = self._random_pos()
        self._dest_x, self._dest_y = self._random_pos()

    def _random_pos(self) -> tuple:
        return (
            float(self._rng.uniform(0.0, self.W)),
            float(self._rng.uniform(0.0, self.H)),
        )

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def step(self, dt: float) -> np.ndarray:
        dx = self._dest_x - self.x
        dy = self._dest_y - self.y
        dist   = np.hypot(dx, dy)
        travel = self.speed * dt

        if travel >= dist:
            self.x, self.y = self._dest_x, self._dest_y
            self._dest_x, self._dest_y = self._random_pos()
        else:
            ratio   = travel / dist
            self.x += ratio * dx
            self.y += ratio * dy

        return self.position.copy()
