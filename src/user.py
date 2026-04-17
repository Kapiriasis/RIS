import numpy as np

class User:
    """
    2-D random waypoint user moving inside a rectangular area [0,W] x [0,H].
    Picks a random destination, travels at constant speed, picks next destination
    on arrival.  Boundary reflections are handled automatically because destinations
    are always inside the area.
    """
    def __init__(
        self,
        area_width: float,
        area_height: float,
        speed: float,
        rng: np.random.Generator = None,
    ):
        self.W = float(area_width)
        self.H = float(area_height)
        self.speed = float(speed)
        self._rng = rng if rng is not None else np.random.default_rng()

        # Start at a random position
        self.x = float(self._rng.uniform(0.0, self.W))
        self.y = float(self._rng.uniform(0.0, self.H))

        # Pick first waypoint
        self._dest_x, self._dest_y = self._new_waypoint()

    def _new_waypoint(self):
        return (
            float(self._rng.uniform(0.0, self.W)),
            float(self._rng.uniform(0.0, self.H)),
        )

    @property
    def position(self):
        return np.array([self.x, self.y])

    def step(self, dt: float) -> np.ndarray:
        dx = self._dest_x - self.x
        dy = self._dest_y - self.y
        dist = np.hypot(dx, dy)

        travel = self.speed * dt

        if travel >= dist:
            # Arrived at waypoint — consume remaining time toward next one
            self.x, self.y = self._dest_x, self._dest_y
            self._dest_x, self._dest_y = self._new_waypoint()
        else:
            ratio = travel / dist
            self.x += ratio * dx
            self.y += ratio * dy

        return self.position.copy()
