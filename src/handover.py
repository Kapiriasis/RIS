"""
Handover model for a 2-D RIS-assisted mmWave network.

Signal model follows Wei & Zhang (2025) equations (1)-(4):
  - LoS direct path  : S = P * K_L * d^{-alpha_L}
  - NLoS direct path : S = P * K_N * d^{-alpha_N}
  - IRS-aided path   : S_ris = P * K_L^2 * G_bf * r^{-alpha_L} * d'^{-alpha_L}
    added on top of direct (LoS or NLoS) for the serving BS only.

LoS determination: a straight line between two points is blocked if it
intersects the square obstacle (axis-aligned, given by centre and half-side).
"""
import numpy as np

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def los_blocked(p1: np.ndarray, p2: np.ndarray,
                sq_cx: float, sq_cy: float, sq_half: float) -> bool:
    """
    Return True if the segment p1→p2 intersects the axis-aligned square
    [sq_cx-sq_half, sq_cx+sq_half] x [sq_cy-sq_half, sq_cy+sq_half].

    Uses the Liang-Barsky / slab method.
    """
    x0, y0 = float(p1[0]), float(p1[1])
    x1, y1 = float(p2[0]), float(p2[1])

    xmin = sq_cx - sq_half
    xmax = sq_cx + sq_half
    ymin = sq_cy - sq_half
    ymax = sq_cy + sq_half

    dx = x1 - x0
    dy = y1 - y0

    t0, t1 = 0.0, 1.0

    for p, q in [(-dx, x0 - xmin), (dx, xmax - x0),
                 (-dy, y0 - ymin), (dy, ymax - y0)]:
        if p == 0.0:
            if q < 0.0:
                return False   # parallel and outside
        else:
            t = q / p
            if p < 0.0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
        if t0 > t1:
            return False

    return True

# ---------------------------------------------------------------------------
# RSRP computation (Wei & Zhang Eqs 1-4)
# ---------------------------------------------------------------------------

def rsrp(
    P_tx: float,
    ue_pos: np.ndarray,
    bs_pos: np.ndarray,
    K_L: float,
    K_N: float,
    alpha_L: float,
    alpha_N: float,
    sq_cx: float,
    sq_cy: float,
    sq_half: float,
    # IRS parameters — None means no IRS for this BS
    ris_pos: np.ndarray = None,
    G_bf: float = 0.0,
    chi_lin: float = 0.0,
) -> float:
    """
    Compute instantaneous (large-scale) RSRP at the UE from one BS.

    Direct path is LoS or NLoS depending on obstacle intersection.
    If ris_pos is given, the IRS-aided component is added when its
    additional gain exceeds chi_lin (linear ratio).
    """
    d = float(np.linalg.norm(ue_pos - bs_pos))
    d = max(d, 1.0)

    blocked = los_blocked(ue_pos, bs_pos, sq_cx, sq_cy, sq_half)

    if blocked:
        S_direct = P_tx * K_N * (d ** (-alpha_N))
    else:
        S_direct = P_tx * K_L * (d ** (-alpha_L))

    S_ris = 0.0
    if ris_pos is not None:
        r = float(np.linalg.norm(ris_pos - bs_pos))
        d_prime = float(np.linalg.norm(ue_pos - ris_pos))
        r = max(r, 1.0)
        d_prime = max(d_prime, 1.0)

        # IRS path requires LoS from BS→RIS and RIS→UE
        bs_ris_blocked = los_blocked(bs_pos, ris_pos, sq_cx, sq_cy, sq_half)
        ris_ue_blocked = los_blocked(ris_pos, ue_pos, sq_cx, sq_cy, sq_half)

        if not bs_ris_blocked and not ris_ue_blocked:
            S_ris_candidate = (
                P_tx * (K_L ** 2) * G_bf
                * (r ** (-alpha_L))
                * (d_prime ** (-alpha_L))
            )
            # Schedule IRS if it improves total RSRP by at least chi_lin factor:
            # (S_direct + S_ris) / S_direct > chi_lin  →  S_ris > (chi_lin-1)*S_direct
            if S_ris_candidate > max(0.0, chi_lin - 1.0) * S_direct:
                S_ris = S_ris_candidate

    return S_direct + S_ris

# ---------------------------------------------------------------------------
# Handover finite-state machine
# ---------------------------------------------------------------------------

class HandoverFSM:
    """
    3GPP A3-event handover with TTT, HOF (Q_out), and ping-pong detection.

    States
    ------
    'connected_a' / 'connected_b' : UE served by BS-A or BS-B
    'in_ttt'                       : A3 condition met, counting down TTT
    'in_hof'                       : HOF condition detected, counting down recovery
    """

    def __init__(
        self,
        ttt: float = 0.04,          # time-to-trigger [s]
        hyst_db: float = 3.0,       # A3 hysteresis [dB]
        q_out_db: float = -8.0,     # HOF threshold: S_serving / S_neighbor [dB]
        tp: float = 1.0,            # ping-pong time [s]
        hof_recovery: float = 0.5,  # time to recover from HOF [s]
    ):
        self.ttt = ttt
        self.hyst = 10.0 ** (hyst_db / 10.0)   # linear
        self.q_out = 10.0 ** (q_out_db / 10.0)  # linear
        self.tp = tp
        self.hof_recovery = hof_recovery

        self.serving = "a"          # 'a' or 'b'
        self.state = "connected_a"
        self._ttt_elapsed = 0.0
        self._last_ho_time = -np.inf
        self._hof_elapsed = 0.0

        self.handover_count = 0
        self.hof_count = 0
        self.pp_count = 0

    def step(self, dt: float, S_a: float, S_b: float, t: float):
        """
        Advance FSM by dt seconds given current RSRP values S_a, S_b.

        Parameters
        ----------
        dt  : time step [s]
        S_a : RSRP from BS-A [linear power]
        S_b : RSRP from BS-B [linear power]
        t   : current simulation time [s]
        """
        if self.state.startswith("connected") or self.state == "in_ttt":
            self._step_normal(dt, S_a, S_b, t)
        elif self.state == "in_hof":
            self._step_hof(dt, S_a, S_b)

    def _step_normal(self, dt, S_a, S_b, t):
        serving_rsrp = S_a if self.serving == "a" else S_b
        neigh_rsrp   = S_b if self.serving == "a" else S_a

        # HOF check: serving signal collapsed
        if serving_rsrp > 0 and neigh_rsrp > 0:
            if serving_rsrp / neigh_rsrp < self.q_out:
                self.hof_count += 1
                self.state = "in_hof"
                self._hof_elapsed = 0.0
                self._ttt_elapsed = 0.0
                return

        # A3 trigger: neighbour > serving * hysteresis
        a3_met = (neigh_rsrp > self.hyst * serving_rsrp)

        if a3_met:
            self._ttt_elapsed += dt
            self.state = "in_ttt"
            if self._ttt_elapsed >= self.ttt:
                # Execute handover
                self.handover_count += 1
                if t - self._last_ho_time < self.tp:
                    self.pp_count += 1
                self._last_ho_time = t
                self.serving = "b" if self.serving == "a" else "a"
                self.state = "connected_b" if self.serving == "b" else "connected_a"
                self._ttt_elapsed = 0.0
        else:
            self._ttt_elapsed = 0.0
            self.state = "connected_a" if self.serving == "a" else "connected_b"

    def _step_hof(self, dt, S_a, S_b):
        self._hof_elapsed += dt
        if self._hof_elapsed >= self.hof_recovery:
            # Re-attach to stronger BS
            self.serving = "a" if S_a >= S_b else "b"
            self.state = "connected_a" if self.serving == "a" else "connected_b"
            self._hof_elapsed = 0.0
