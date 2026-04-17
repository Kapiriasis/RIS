import numpy as np
from src.utils import db2lin


# ---------------------------------------------------------------------------
# RSRP models  (Wei & Zhang 2025, Eqs. 1–4)
# ---------------------------------------------------------------------------

def rsrp_without_ris(P_tx, K_L, distance, alpha_L):
    """S = P_tx * K_L * d^(-alpha_L)"""
    return P_tx * K_L * (distance ** -alpha_L)


def rsrp_with_ris(P_tx, K_L, d_bs, alpha_L, G_bf, r_g, d_ris_ue, alpha_ris=None):
    """
    S = P_tx*K_L*d^(-alpha_L) + P_tx*K_L^2 * G_bf * r_g^(-alpha_L) * d_ris_ue^(-alpha_L)

    Parameters
    ----------
    d_bs       : distance from serving BS to UE
    r_g        : distance from serving BS to RIS
    d_ris_ue   : distance from RIS to UE
    alpha_ris  : path-loss exponent for RIS hops (defaults to alpha_L)
    """
    if alpha_ris is None:
        alpha_ris = alpha_L
    direct = rsrp_without_ris(P_tx, K_L, d_bs, alpha_L)
    reflected = P_tx * (K_L ** 2) * G_bf * (r_g ** -alpha_ris) * (d_ris_ue ** -alpha_ris)
    return direct + reflected


def ris_beamforming_gain(F_g, M_g):
    """G_bf = F_g * M_g^2"""
    return F_g * (M_g ** 2)


# ---------------------------------------------------------------------------
# Large-scale path-loss constant K_L from 3GPP / free-space reference
# ---------------------------------------------------------------------------

def path_loss_constant(frequency, d_ref=1.0):
    """
    K_L = (c / (4*pi*f*d_ref))^2  scaled so that PL(d_ref)=1 when alpha_L=2.
    This is the intercept used in  PL(d) = K_L * d^{-alpha_L}.
    """
    c = 3e8
    wavelength = c / frequency
    return (wavelength / (4.0 * np.pi * d_ref)) ** 2


# ---------------------------------------------------------------------------
# A3-event handover state machine
# ---------------------------------------------------------------------------

class HandoverFSM:
    """
    Implements 3GPP A3 event: handover triggers when
        RSRP_neighbour - RSRP_serving > gamma_HO
    sustained for TTT seconds.

    HOF  : serving RSRP / neighbour RSRP < Q_out during TTT window.
    PP   : handover back to original BS within sojourn time T_p.
    """

    def __init__(
        self,
        gamma_HO_dB: float = 3.0,
        TTT: float = 0.04,          # 40 ms
        Q_out_dB: float = -8.0,
        T_p: float = 1.0,           # ping-pong sojourn window [s]
        dt: float = 0.01,
    ):
        self.gamma_HO = db2lin(gamma_HO_dB)   # linear ratio
        self.Q_out = db2lin(Q_out_dB)
        self.TTT = TTT
        self.T_p = T_p
        self.dt = dt

        # Runtime state
        self.serving_bs: int = 0          # index of current serving BS (0 = BS_A)
        self.ttt_elapsed: float = 0.0     # time TTT condition has been active
        self.ttt_active: bool = False
        self.last_ho_time: float = -np.inf
        self.last_ho_from: int = -1

        # Counters
        self.handover_count: int = 0
        self.hof_count: int = 0
        self.pp_count: int = 0

    # ------------------------------------------------------------------
    def step(self, t: float, rsrp_a: float, rsrp_b: float):
        """
        Update FSM for one time step.

        Parameters
        ----------
        t       : current simulation time [s]
        rsrp_a  : RSRP from BS_A
        rsrp_b  : RSRP from BS_B
        """
        rsrp = [rsrp_a, rsrp_b]
        neighbour = 1 - self.serving_bs
        S_s = rsrp[self.serving_bs]
        S_n = rsrp[neighbour]

        a3_condition = (S_n / S_s) > self.gamma_HO if S_s > 0 else False
        hof_condition = (S_s / S_n) < self.Q_out if S_n > 0 else False

        if a3_condition:
            if not self.ttt_active:
                self.ttt_active = True
                self.ttt_elapsed = 0.0
            self.ttt_elapsed += self.dt

            # HOF check during TTT window
            if hof_condition:
                self.hof_count += 1
                self._execute_handover(t, neighbour)
                return

            if self.ttt_elapsed >= self.TTT:
                self._execute_handover(t, neighbour)
        else:
            self.ttt_active = False
            self.ttt_elapsed = 0.0

    # ------------------------------------------------------------------
    def _execute_handover(self, t: float, target: int):
        self.ttt_active = False
        self.ttt_elapsed = 0.0

        # Ping-pong: handover back to where we came from within T_p
        if (
            self.last_ho_from == target
            and (t - self.last_ho_time) <= self.T_p
        ):
            self.pp_count += 1

        self.last_ho_from = self.serving_bs
        self.last_ho_time = t
        self.serving_bs = target
        self.handover_count += 1

    # ------------------------------------------------------------------
    @property
    def summary(self):
        return {
            "handover_count": self.handover_count,
            "hof_count": self.hof_count,
            "pp_count": self.pp_count,
        }
