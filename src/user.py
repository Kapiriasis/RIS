import numpy as np


class User:
    """
    2-D random waypoint user inside a rectangular area [0,W] x [0,H].
    An optional axis-aligned square obstacle is excluded from valid positions
    (rejection sampling for waypoints and start position).
    """

    def __init__(
        self,
        area_width: float,
        area_height: float,
        speed: float,
        rng: np.random.Generator = None,
        obstacle: tuple = None,
    ):
        """
        obstacle : (cx, cy, half_side) or None
        """
        self.W = float(area_width)
        self.H = float(area_height)
        self.speed = float(speed)
        self._rng = rng if rng is not None else np.random.default_rng()
        self._obstacle = obstacle  # (cx, cy, half)

        self.x, self.y = self._random_pos()
        self._dest_x, self._dest_y = self._new_waypoint()

    def _in_obstacle(self, x: float, y: float) -> bool:
        if self._obstacle is None:
            return False
        cx, cy, half = self._obstacle
        return abs(x - cx) <= half and abs(y - cy) <= half

    def _random_pos(self) -> tuple:
        while True:
            x = float(self._rng.uniform(0.0, self.W))
            y = float(self._rng.uniform(0.0, self.H))
            if not self._in_obstacle(x, y):
                return x, y

    def _new_waypoint(self) -> tuple:
        return self._random_pos()

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def step(self, dt: float) -> np.ndarray:
        dx = self._dest_x - self.x
        dy = self._dest_y - self.y
        dist = np.hypot(dx, dy)
        travel = self.speed * dt

        if travel >= dist:
            self.x, self.y = self._dest_x, self._dest_y
            self._dest_x, self._dest_y = self._new_waypoint()
        else:
            ratio = travel / dist
            self.x += ratio * dx
            self.y += ratio * dy

        return self.position.copy()
