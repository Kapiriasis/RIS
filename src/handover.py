"""
Handover model for a 2-D RIS-assisted multi-BS network.

Signal model follows Wei & Zhang (2025):
  - LoS direct path  : S = P * K_L * d^{-alpha_L}
  - NLoS direct path : S = P * K_N * d^{-alpha_N}
  - IRS-aided path   : S_ris = P * K_L^2 * G_bf * r^{-alpha_L} * d'^{-alpha_L}

LoS is blocked when the UE-BS segment strictly crosses ANY wall in the list.
The serving BS selects the best RIS from a list (one per wall).

HOF uses an absolute serving-link SNR threshold (radio link failure).
"""
import numpy as np
from typing import List, Tuple

Wall = Tuple[np.ndarray, np.ndarray]   # (endpoint_1, endpoint_2)

# Geometry: strict line-segment intersection
def los_blocked(
    p1: np.ndarray,
    p2: np.ndarray,
    obs_p1: np.ndarray,
    obs_p2: np.ndarray,
) -> bool:
    """
    True if segment p1->p2 strictly crosses wall obs_p1->obs_p2.
    Endpoint touches (t or u == 0 or 1) are NOT counted as blocked,
    so a RIS sitting exactly at a wall tip still sees both sides.
    """
    d = p2 - p1
    e = obs_p2 - obs_p1
    cross = float(d[0] * e[1] - d[1] * e[0])
    if abs(cross) < 1e-10:
        return False          # parallel or collinear
    diff = obs_p1 - p1
    t = float(diff[0] * e[1] - diff[1] * e[0]) / cross
    u = float(diff[0] * d[1]  - diff[1] * d[0]) / cross
    return 0.0 < t < 1.0 and 0.0 < u < 1.0

def _any_blocked(p1: np.ndarray, p2: np.ndarray, walls: List[Wall]) -> bool:
    return any(los_blocked(p1, p2, w[0], w[1]) for w in walls)

# RSRP for one BS (optionally with one RIS candidate)
def rsrp(
    P_tx: float,
    ue_pos: np.ndarray,
    bs_pos: np.ndarray,
    K_L: float,
    K_N: float,
    alpha_L: float,
    alpha_N: float,
    walls: List[Wall],
    ris_pos: np.ndarray = None,
    G_bf: float = 0.0,
    chi_lin: float = 0.0,
) -> float:
    """
    Large-scale RSRP using LoS or NLoS exponent based on wall intersection.
    Adds the IRS-aided component when both BS->RIS and RIS->UE are clear.
    """
    d = max(float(np.linalg.norm(ue_pos - bs_pos)), 1.0)
    blocked = _any_blocked(ue_pos, bs_pos, walls)

    S_direct = (P_tx * K_N * d ** (-alpha_N) if blocked
                else P_tx * K_L * d ** (-alpha_L))

    if ris_pos is None:
        return S_direct

    r  = max(float(np.linalg.norm(ris_pos - bs_pos)), 1.0)
    dp = max(float(np.linalg.norm(ue_pos  - ris_pos)), 1.0)

    if (not _any_blocked(bs_pos, ris_pos, walls) and
            not _any_blocked(ris_pos, ue_pos, walls)):
        S_cand = P_tx * (K_L ** 2) * G_bf * r ** (-alpha_L) * dp ** (-alpha_L)
        if S_cand > max(0.0, chi_lin - 1.0) * S_direct:
            return S_direct + S_cand

    return S_direct


def best_rsrp(
    P_tx: float,
    ue_pos: np.ndarray,
    bs_pos: np.ndarray,
    K_L: float,
    K_N: float,
    alpha_L: float,
    alpha_N: float,
    walls: List[Wall],
    ris_list: List[np.ndarray],
    G_bf: float,
    chi_lin: float,
) -> float:
    # RSRP for one BS, using the best RIS from ris_list (or direct if none helps)
    S = rsrp(P_tx, ue_pos, bs_pos, K_L, K_N, alpha_L, alpha_N, walls)
    for ris_pos in ris_list:
        S_try = rsrp(P_tx, ue_pos, bs_pos, K_L, K_N, alpha_L, alpha_N,
                     walls, ris_pos=ris_pos, G_bf=G_bf, chi_lin=chi_lin)
        if S_try > S:
            S = S_try
    return S

# Multi-BS handover FSM
class HandoverFSM:
    """
    3GPP A3-event handover for N base stations, with TTT, absolute-SNR HOF,
    and ping-pong detection.

    step() receives rsrp_arr (shape n_bs) at each time step.
    Call initialise_serving() before the first step().
    """

    def __init__(
        self,
        n_bs: int,
        noise_power: float,
        ttt: float = 0.04,
        hyst_db: float = 3.0,
        q_out_snr_db: float = -20.0,
        tp: float = 1.0,
        hof_recovery: float = 0.5,
    ):
        self.n_bs         = n_bs
        self.noise_power  = noise_power
        self.ttt          = ttt
        self.hyst         = 10.0 ** (hyst_db      / 10.0)
        self.q_out_snr    = 10.0 ** (q_out_snr_db / 10.0)
        self.tp           = tp
        self.hof_recovery = hof_recovery

        self.serving       = 0
        self.state         = "connected"
        self._prev_serving = None
        self._ttt_elapsed  = 0.0
        self._last_ho_time = -np.inf
        self._hof_elapsed  = 0.0

        self.handover_count = 0
        self.hof_count      = 0
        self.pp_count       = 0

    def initialise_serving(self, rsrp_arr: np.ndarray):
        self.serving = int(np.argmax(rsrp_arr))

    def step(self, dt: float, rsrp_arr: np.ndarray, t: float):
        if self.state in ("connected", "in_ttt"):
            self._step_normal(dt, rsrp_arr, t)
        elif self.state == "in_hof":
            self._step_hof(dt, rsrp_arr)

    def _step_normal(self, dt, rsrp_arr, t):
        serving_rsrp = rsrp_arr[self.serving]

        if serving_rsrp / self.noise_power < self.q_out_snr:
            self.hof_count += 1
            self.state        = "in_hof"
            self._hof_elapsed = 0.0
            self._ttt_elapsed = 0.0
            return

        best_idx, best_val = self.serving, serving_rsrp
        for j in range(self.n_bs):
            if j != self.serving and rsrp_arr[j] > best_val:
                best_val = rsrp_arr[j]
                best_idx = j

        if best_idx != self.serving and best_val > self.hyst * serving_rsrp:
            self._ttt_elapsed += dt
            self.state = "in_ttt"
            if self._ttt_elapsed >= self.ttt:
                self.handover_count += 1
                if (t - self._last_ho_time < self.tp and
                        best_idx == self._prev_serving):
                    self.pp_count += 1
                self._prev_serving = self.serving
                self._last_ho_time = t
                self.serving       = best_idx
                self.state         = "connected"
                self._ttt_elapsed  = 0.0
        else:
            self._ttt_elapsed = 0.0
            self.state = "connected"

    def _step_hof(self, dt, rsrp_arr):
        self._hof_elapsed += dt
        if self._hof_elapsed >= self.hof_recovery:
            self.serving      = int(np.argmax(rsrp_arr))
            self.state        = "connected"
            self._hof_elapsed = 0.0
