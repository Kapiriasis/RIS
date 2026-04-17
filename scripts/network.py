"""
2-D RIS-assisted handover simulation.

Topology
--------
  Area  : W x H metres (default 500 x 500)
  BS-A  : left edge,  centre-height
  BS-B  : right edge, centre-height
  RIS   : near the centre, offset toward BS-A side so it has clear LoS to both
  Obstacle : axis-aligned square at area centre, blocks direct LoS

Two conditions are compared over N_runs independent Monte-Carlo trajectories:
  'no_ris'  - both BSs use direct path only
  'ris'     - serving BS adds IRS-aided component when gain threshold met

Parameters follow Wei & Zhang (2025):
  K_L = 10^{-10.38}, K_N = 10^{-14.54}
  alpha_L = 2.09, alpha_N = 3.75
  F_g = 0.76, M_g = 200, G_bf = F_g * M_g^2
  chi = 6 dB  (IRS connectivity threshold)
  Q_out = -8 dB (HOF threshold)
  T_p = 1 s   (ping-pong window)
  TTT = 40 ms
  hysteresis = 3 dB
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

from src.user import User
from src.handover import rsrp, HandoverFSM
from src.utils import db2lin

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

DEFAULT_NET_PARAMS: Dict[str, Any] = {
    # Area
    "area_width": 500.0,
    "area_height": 500.0,
    # BS positions  [x, y]
    "bs_a_pos": [0.0, 250.0],
    "bs_b_pos": [500.0, 250.0],
    # RIS position [x, y]  — left of centre, clear LoS to both BSs
    "ris_pos": [200.0, 150.0],
    # Square obstacle: centre [x,y] and half-side
    "obstacle_cx": 250.0,
    "obstacle_cy": 250.0,
    "obstacle_half": 60.0,
    # Path-loss parameters (Wei & Zhang large-scale model)
    "K_L": 10 ** (-10.38),
    "K_N": 10 ** (-14.54),
    "alpha_L": 2.09,
    "alpha_N": 3.75,
    # IRS beamforming
    "F_g": 0.76,
    "M_g": 200,
    # Transmit power [W]
    "P_tx": db2lin(10.0) / 1000.0,  # 10 dBm
    # Handover thresholds
    "chi_dB": 6.0,       # IRS gain threshold over direct
    "q_out_dB": -8.0,    # HOF threshold
    "hyst_dB": 3.0,      # A3 hysteresis
    "ttt": 0.04,         # time-to-trigger [s]
    "tp": 1.0,           # ping-pong window [s]
    # Mobility
    "speed": 10.0,       # m/s
    # Simulation
    "dt": 0.01,          # time step [s]
    "T_sim": 60.0,       # trajectory duration [s]
    "N_runs": 500,       # Monte-Carlo runs
}

# ---------------------------------------------------------------------------
# Single-trajectory simulation
# ---------------------------------------------------------------------------

def _simulate_one(
    params: Dict[str, Any],
    use_ris: bool,
    rng: np.random.Generator,
) -> Dict[str, int]:
    W = params["area_width"]
    H = params["area_height"]
    bs_a = np.array(params["bs_a_pos"], dtype=float)
    bs_b = np.array(params["bs_b_pos"], dtype=float)
    ris  = np.array(params["ris_pos"],  dtype=float)

    cx   = params["obstacle_cx"]
    cy   = params["obstacle_cy"]
    half = params["obstacle_half"]

    K_L     = params["K_L"]
    K_N     = params["K_N"]
    aL      = params["alpha_L"]
    aN      = params["alpha_N"]
    P_tx    = params["P_tx"]

    G_bf    = params["F_g"] * (params["M_g"] ** 2)
    chi_lin = db2lin(params["chi_dB"])

    fsm = HandoverFSM(
        ttt=params["ttt"],
        hyst_db=params["hyst_dB"],
        q_out_db=params["q_out_dB"],
        tp=params["tp"],
    )

    user = User(W, H, params["speed"], rng=rng)
    dt   = params["dt"]
    T    = params["T_sim"]
    t    = 0.0

    while t < T:
        pos = user.step(dt)

        # Serving BS gets RIS when use_ris=True; neighbour never does
        if use_ris:
            ris_a = ris if fsm.serving == "a" else None
            ris_b = ris if fsm.serving == "b" else None
        else:
            ris_a = None
            ris_b = None

        S_a = rsrp(P_tx, pos, bs_a, K_L, K_N, aL, aN, cx, cy, half,
                   ris_pos=ris_a, G_bf=G_bf, chi_lin=chi_lin)
        S_b = rsrp(P_tx, pos, bs_b, K_L, K_N, aL, aN, cx, cy, half,
                   ris_pos=ris_b, G_bf=G_bf, chi_lin=chi_lin)

        fsm.step(dt, S_a, S_b, t)
        t += dt

    return {
        "handovers": fsm.handover_count,
        "hofs":      fsm.hof_count,
        "pps":       fsm.pp_count,
    }

# ---------------------------------------------------------------------------
# Monte-Carlo runner
# ---------------------------------------------------------------------------

def run_network(
    params: Optional[Dict[str, Any]] = None,
    results_dir: Optional[str] = None,
) -> Dict[str, Any]:
    if params is None:
        params = DEFAULT_NET_PARAMS
    if results_dir is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(root, "results")
    os.makedirs(results_dir, exist_ok=True)

    N = int(params["N_runs"])
    rng = np.random.default_rng(42)

    ho_no_ris,  hof_no_ris,  pp_no_ris  = [], [], []
    ho_ris,     hof_ris,     pp_ris     = [], [], []

    t0 = time.time()
    for i in range(N):
        seed = rng.integers(0, 2**31)
        r_no = _simulate_one(params, use_ris=False, rng=np.random.default_rng(seed))
        r_ri = _simulate_one(params, use_ris=True,  rng=np.random.default_rng(seed))

        ho_no_ris.append(r_no["handovers"]); hof_no_ris.append(r_no["hofs"]); pp_no_ris.append(r_no["pps"])
        ho_ris.append(r_ri["handovers"]);    hof_ris.append(r_ri["hofs"]);    pp_ris.append(r_ri["pps"])

        if (i + 1) % max(1, N // 10) == 0:
            elapsed = time.time() - t0
            print(f"  Run {i+1}/{N}  ({elapsed:.1f}s elapsed)")

    results = {
        "no_ris": {
            "handovers": np.array(ho_no_ris),
            "hofs":      np.array(hof_no_ris),
            "pps":       np.array(pp_no_ris),
        },
        "ris": {
            "handovers": np.array(ho_ris),
            "hofs":      np.array(hof_ris),
            "pps":       np.array(pp_ris),
        },
    }

    _print_summary(results, N)
    _plot_results(results, params, results_dir)
    return results

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_summary(results: Dict, N: int):
    print(f"\nMonte Carlo results (N={N} runs)")
    print(f"{'Metric':<20} {'No RIS':>20} {'With RIS':>20}")
    print("-" * 62)
    for key, label in [("handovers", "Handovers"), ("hofs", "HOFs"), ("pps", "Ping-pongs")]:
        a = results["no_ris"][key]
        b = results["ris"][key]
        print(f"{label:<20} {np.mean(a):>8.2f} ± {np.std(a):<8.2f}   {np.mean(b):>8.2f} ± {np.std(b):<8.2f}")
    print()

def _plot_results(results: Dict, params: Dict, results_dir: str):
    metrics = [("handovers", "Handovers per run"),
               ("hofs",      "HOFs per run"),
               ("pps",       "Ping-pongs per run")]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (key, title) in zip(axes, metrics):
        data_no = results["no_ris"][key]
        data_ri = results["ris"][key]
        max_val = max(data_no.max(), data_ri.max(), 1)
        bins = np.arange(0, max_val + 2) - 0.5
        ax.hist(data_no, bins=bins, alpha=0.6, label="No RIS", color="steelblue")
        ax.hist(data_ri, bins=bins, alpha=0.6, label="With RIS", color="darkorange")
        ax.set_title(title)
        ax.set_xlabel("Count")
        ax.set_ylabel("Frequency")
        ax.legend()
    fig.suptitle(
        f"2-D RIS handover simulation  —  area {params['area_width']:.0f}×{params['area_height']:.0f} m, "
        f"obstacle ±{params['obstacle_half']:.0f} m, N={params['N_runs']}",
        fontsize=10,
    )
    fig.tight_layout()
    out = os.path.join(results_dir, "network_handover_stats.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {out}")

    _plot_topology(params, results_dir)

def _plot_topology(params: Dict, results_dir: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    W, H = params["area_width"], params["area_height"]
    cx, cy, half = params["obstacle_cx"], params["obstacle_cy"], params["obstacle_half"]

    # Obstacle
    sq = plt.Rectangle((cx - half, cy - half), 2 * half, 2 * half,
                        color="gray", alpha=0.5, label="Obstacle (NLoS)")
    ax.add_patch(sq)

    # BSs and RIS
    bsa = params["bs_a_pos"]
    bsb = params["bs_b_pos"]
    ris = params["ris_pos"]
    ax.plot(*bsa, "bs", markersize=12, label="BS-A")
    ax.plot(*bsb, "rs", markersize=12, label="BS-B")
    ax.plot(*ris, "g^", markersize=12, label="RIS")

    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Network topology")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = os.path.join(results_dir, "network_topology.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Topology plot saved → {out}")
