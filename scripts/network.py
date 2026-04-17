"""
RIS-aided handover simulation.

Two runs per call:
  1. Without RIS  — RSRP from each BS is pure direct path.
  2. With RIS     — RSRP from serving BS includes IRS reflected component.

Results are collected along the vehicle trajectory and written to results/.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from src.user import User
from src.handover import HandoverFSM, path_loss_constant, rsrp_without_ris, rsrp_with_ris, ris_beamforming_gain
from src.utils import lin2db

# ---------------------------------------------------------------------------
# Default simulation parameters
# ---------------------------------------------------------------------------
DEFAULT_NET_PARAMS = {
    # Geometry
    "total_distance": 200.0,      # BS_A at 0 m, BS_B at 200 m
    "ris_position_from_a": 100.0, # RIS midpoint along the link
    # Radio
    "P_tx": 10.0,                 # 10 W per BS
    "frequency": 28e9,
    "alpha_L": 2.5,               # LOS path-loss exponent
    # RIS
    "ris_elements": 400,
    "F_g": 1.0,                   # element directivity factor
    # A3 / TTT
    "gamma_HO_dB": 3.0,
    "TTT": 0.04,                  # 40 ms
    "Q_out_dB": -8.0,
    "T_p": 1.0,                   # ping-pong window [s]
    # Trajectory
    "sim_duration": 60.0,         # seconds
    "dt": 0.01,                   # time step [s]
    "speed": None,                # None → random per User.__init__
}


def _results_dir():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "results")


def run_network(params: dict = None):
    p = DEFAULT_NET_PARAMS.copy()
    if params:
        p.update(params)

    D = p["total_distance"]
    r_g = p["ris_position_from_a"]           # BS_A ↔ RIS distance
    alpha = p["alpha_L"]
    P_tx = p["P_tx"]
    dt = p["dt"]
    T = p["sim_duration"]
    steps = int(T / dt)

    K_L = path_loss_constant(p["frequency"])
    G_bf = ris_beamforming_gain(p["F_g"], p["ris_elements"])

    rng = np.random.default_rng()

    results = {}
    for use_ris in (False, True):
        label = "with_ris" if use_ris else "without_ris"
        user = User(D, speed=p["speed"], rng=rng)
        fsm = HandoverFSM(
            gamma_HO_dB=p["gamma_HO_dB"],
            TTT=p["TTT"],
            Q_out_dB=p["Q_out_dB"],
            T_p=p["T_p"],
            dt=dt,
        )

        times = np.arange(steps) * dt
        positions = np.empty(steps)
        rsrp_serving_log = np.empty(steps)
        rsrp_neighbor_log = np.empty(steps)
        serving_log = np.empty(steps, dtype=int)

        for i in range(steps):
            pos = user.position
            positions[i] = pos

            d_a = max(pos, 1.0)           # distance to BS_A (guard against 0)
            d_b = max(D - pos, 1.0)       # distance to BS_B

            if use_ris:
                d_ris_ue = abs(pos - r_g)
                d_ris_ue = max(d_ris_ue, 1.0)
                s_a = rsrp_with_ris(P_tx, K_L, d_a, alpha, G_bf, r_g, d_ris_ue)
                # BS_B side: RIS is also at r_g from BS_A → distance from BS_B to RIS
                r_g_b = abs(D - r_g)
                s_b = rsrp_with_ris(P_tx, K_L, d_b, alpha, G_bf, r_g_b, d_ris_ue)
            else:
                s_a = rsrp_without_ris(P_tx, K_L, d_a, alpha)
                s_b = rsrp_without_ris(P_tx, K_L, d_b, alpha)

            fsm.step(times[i], s_a, s_b)
            user.step(dt)

            rsrp = [s_a, s_b]
            rsrp_serving_log[i] = lin2db(rsrp[fsm.serving_bs])
            rsrp_neighbor_log[i] = lin2db(rsrp[1 - fsm.serving_bs])
            serving_log[i] = fsm.serving_bs

        results[label] = {
            "times": times,
            "positions": positions,
            "rsrp_serving_dB": rsrp_serving_log,
            "rsrp_neighbor_dB": rsrp_neighbor_log,
            "serving_bs": serving_log,
            **fsm.summary,
        }

    _plot_results(results)
    _print_summary(results)
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_results(results):
    out_dir = _results_dir()
    os.makedirs(out_dir, exist_ok=True)

    _, axes = plt.subplots(2, 2, figsize=(12, 8))
    titles = ["Without RIS", "With RIS"]
    keys = ["without_ris", "with_ris"]

    for col, (key, title) in enumerate(zip(keys, titles)):
        r = results[key]
        ax_rsrp = axes[0, col]
        ax_pos = axes[1, col]

        ax_rsrp.plot(r["times"], r["rsrp_serving_dB"], label="Serving BS")
        ax_rsrp.plot(r["times"], r["rsrp_neighbor_dB"], label="Neighbour BS", linestyle="--")
        ax_rsrp.set_xlabel("Time [s]")
        ax_rsrp.set_ylabel("RSRP [dBW]")
        ax_rsrp.set_title(f"RSRP — {title}")
        ax_rsrp.legend()
        ax_rsrp.grid(True, linestyle="--", alpha=0.5)

        ax_pos.plot(r["times"], r["positions"])
        ax_pos.set_xlabel("Time [s]")
        ax_pos.set_ylabel("UE position [m]")
        ax_pos.set_title(f"Trajectory — {title}")
        ax_pos.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "handover_simulation.png"))
    plt.close()

    # Serving BS over time (both on one plot)
    _, ax = plt.subplots(figsize=(10, 3))
    for key, title in zip(keys, titles):
        r = results[key]
        ax.plot(r["times"], r["serving_bs"], label=title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Serving BS index")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["BS_A", "BS_B"])
    ax.set_title("Serving BS over time")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "handover_serving_bs.png"))
    plt.close()


def _print_summary(results):
    for label, r in results.items():
        print(f"\n=== {label} ===")
        print(f"  Handovers : {r['handover_count']}")
        print(f"  HOFs      : {r['hof_count']}")
        print(f"  Ping-pongs: {r['pp_count']}")
