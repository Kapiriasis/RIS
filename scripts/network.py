"""
2-D RIS-assisted handover simulation.

Topology — equal counts of BSs, walls, and RIS (controlled by n_nodes):
  Area   : 1000 x 1000 m
  BSs    : n_nodes stations, uniform-random placement
  Walls  : n_nodes line-segment obstacles, random centre and orientation,
           fixed half-length; each wall represents a building facade
  RIS    : n_nodes elements, one per wall — placed just past the wall's
           near endpoint so it has clear LoS to both sides of the obstacle

The serving BS selects the best RIS from the full list.

Path-loss: Wei & Zhang (2025) 3.5 GHz experiment
  K_L = K_N = 10^{-4.33}, alpha_L = 1.73, alpha_N = 3.19
  G_bf = F_g * M_g^2  (M_g = 500 for 1000 m cells)

HOF: serving SNR < q_out_snr_db = -20 dB  (absolute threshold).
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional

from src.user import User
from src.handover import rsrp, best_rsrp, HandoverFSM, Wall
from src.utils import db2lin
from src.channel import noise_power as compute_noise_power

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

DEFAULT_NET_PARAMS: Dict[str, Any] = {
    # Area [m]
    "area_width":  1000.0,
    "area_height": 1000.0,
    # Equal counts: BSs = walls = RIS
    "n_nodes":      10,
    # BS placement
    "bs_min_sep":   300.0,
    # Wall geometry
    "wall_half_len":  150.0,   # half-length of each wall [m]
    "ris_offset":      25.0,   # RIS sits this far past the wall tip [m]
    # Shared topology seed (BSs + walls placed once, fixed across all runs)
    "topo_seed": 42,
    # Path-loss (Wei & Zhang 3.5 GHz)
    "K_L":     10 ** (-4.33),
    "K_N":     10 ** (-4.33),
    "alpha_L": 1.73,
    "alpha_N": 3.19,
    # IRS beamforming
    "F_g": 0.76,
    "M_g": 500,
    # Transmit power [W]  (10 dBm)
    "P_tx": db2lin(10.0) / 1000.0,
    # Noise
    "bandwidth":       20e6,
    "noise_figure_dB": 7.0,
    # Handover thresholds
    "chi_dB":       0.0,
    "q_out_snr_db": -20.0,
    "hyst_dB":       3.0,
    "ttt":           0.04,
    "tp":            1.0,
    # Mobility
    "speed": 10.0,
    # Simulation
    "dt":     0.01,
    "T_sim":  60.0,
    "N_runs": 500,
}

# ---------------------------------------------------------------------------
# Topology builders
# ---------------------------------------------------------------------------

def _place_bs(
    n: int,
    width: float,
    height: float,
    min_sep: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniform-random BS placement with minimum inter-site separation."""
    positions: List[np.ndarray] = []
    for _ in range(200_000):
        if len(positions) == n:
            break
        p = rng.uniform([0.0, 0.0], [width, height])
        if all(np.linalg.norm(p - q) >= min_sep for q in positions):
            positions.append(p)
    if len(positions) < n:
        raise RuntimeError(
            f"Could not place {n} BSs with min separation {min_sep} m."
        )
    return np.array(positions)


def _place_walls(
    n: int,
    width: float,
    height: float,
    half_len: float,
    rng: np.random.Generator,
) -> List[Wall]:
    """
    n walls with random centre position and random orientation.
    Centres are kept at least half_len + 10 m from each area boundary so
    walls do not protrude outside.
    """
    margin = half_len + 10.0
    walls: List[Wall] = []
    for _ in range(n):
        cx = rng.uniform(margin, width  - margin)
        cy = rng.uniform(margin, height - margin)
        theta = rng.uniform(0.0, np.pi)          # orientation in [0, π)
        dx = half_len * np.cos(theta)
        dy = half_len * np.sin(theta)
        p1 = np.array([cx - dx, cy - dy])
        p2 = np.array([cx + dx, cy + dy])
        walls.append((p1, p2))
    return walls


def _place_ris(walls: List[Wall], offset: float) -> List[np.ndarray]:
    """
    One RIS per wall, placed just past the p1 endpoint in the direction
    away from p2.  This gives it clear LoS to both sides of the wall
    (the strict-intersection check excludes t = 0, so starting exactly at
    the endpoint does not count as blocked).
    """
    ris_list: List[np.ndarray] = []
    for p1, p2 in walls:
        wall_dir = (p2 - p1) / np.linalg.norm(p2 - p1)
        ris = p1 - offset * wall_dir          # step past the p1 end
        ris_list.append(ris)
    return ris_list

# ---------------------------------------------------------------------------
# Single-trajectory simulation
# ---------------------------------------------------------------------------

def _simulate_one(
    params: Dict[str, Any],
    bs_positions: np.ndarray,
    walls: List[Wall],
    ris_list: List[np.ndarray],
    use_ris: bool,
    rng: np.random.Generator,
    P_noise: float,
) -> Dict[str, int]:
    W  = params["area_width"];  H = params["area_height"]
    K_L  = params["K_L"];  K_N  = params["K_N"]
    aL   = params["alpha_L"]; aN = params["alpha_N"]
    P_tx = params["P_tx"]
    G_bf    = params["F_g"] * (params["M_g"] ** 2)
    chi_lin = db2lin(params["chi_dB"])
    n_bs    = len(bs_positions)

    fsm = HandoverFSM(
        n_bs         = n_bs,
        noise_power  = P_noise,
        ttt          = params["ttt"],
        hyst_db      = params["hyst_dB"],
        q_out_snr_db = params["q_out_snr_db"],
        tp           = params["tp"],
    )

    user = User(W, H, params["speed"], rng=rng)

    init_pos = user.position
    init_rsrp = np.array([
        rsrp(P_tx, init_pos, bs_positions[i], K_L, K_N, aL, aN, walls)
        for i in range(n_bs)
    ])
    fsm.initialise_serving(init_rsrp)

    dt = params["dt"]
    T  = params["T_sim"]
    t  = 0.0

    while t < T:
        pos = user.step(dt)

        rsrp_arr = np.empty(n_bs)
        for i in range(n_bs):
            if use_ris and i == fsm.serving:
                rsrp_arr[i] = best_rsrp(
                    P_tx, pos, bs_positions[i], K_L, K_N, aL, aN,
                    walls, ris_list, G_bf, chi_lin,
                )
            else:
                rsrp_arr[i] = rsrp(
                    P_tx, pos, bs_positions[i], K_L, K_N, aL, aN, walls,
                )

        fsm.step(dt, rsrp_arr, t)
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
    params = dict(params)   # don't mutate the caller's dict

    # Scale area so each BS covers ~250 000 m² regardless of n_nodes.
    # Baseline: n_nodes=4 → 1000×1000 m (side = 500*sqrt(4) = 1000).
    n = int(params["n_nodes"])
    scale = np.sqrt(n / 4.0)
    params["area_width"]    = 1000.0 * scale
    params["area_height"]   = 1000.0 * scale
    params["bs_min_sep"]    = 300.0  * scale
    params["wall_half_len"] = 150.0  * scale

    if results_dir is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(root, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Fixed topology (same across all N_runs)
    topo_rng = np.random.default_rng(int(params["topo_seed"]))
    n = int(params["n_nodes"])
    bs_positions = _place_bs(n, params["area_width"], params["area_height"],
                             params["bs_min_sep"], topo_rng)
    walls   = _place_walls(n, params["area_width"], params["area_height"],
                           params["wall_half_len"], topo_rng)
    ris_list = _place_ris(walls, params["ris_offset"])
    P_noise = compute_noise_power(params["bandwidth"], params["noise_figure_dB"])

    print(f"Topology: {n} BSs, {n} walls, {n} RIS")
    for i, p in enumerate(bs_positions):
        print(f"  BS-{i}: ({p[0]:.0f}, {p[1]:.0f})")

    N   = int(params["N_runs"])
    rng = np.random.default_rng(0)

    ho_no, hof_no, pp_no = [], [], []
    ho_ri, hof_ri, pp_ri = [], [], []

    t0 = time.time()
    for i in range(N):
        seed = rng.integers(0, 2 ** 31)
        r_no = _simulate_one(params, bs_positions, walls, ris_list, False,
                             np.random.default_rng(seed), P_noise)
        r_ri = _simulate_one(params, bs_positions, walls, ris_list, True,
                             np.random.default_rng(seed), P_noise)

        ho_no.append(r_no["handovers"]); hof_no.append(r_no["hofs"]); pp_no.append(r_no["pps"])
        ho_ri.append(r_ri["handovers"]); hof_ri.append(r_ri["hofs"]); pp_ri.append(r_ri["pps"])

        if (i + 1) % max(1, N // 10) == 0:
            print(f"  Run {i+1}/{N}  ({time.time()-t0:.1f}s)")

    results = {
        "no_ris": {
            "handovers": np.array(ho_no),
            "hofs":      np.array(hof_no),
            "pps":       np.array(pp_no),
        },
        "ris": {
            "handovers": np.array(ho_ri),
            "hofs":      np.array(hof_ri),
            "pps":       np.array(pp_ri),
        },
        "bs_positions": bs_positions,
        "walls":        walls,
        "ris_list":     ris_list,
    }
    _print_summary(results, N, n)
    _plot_results(results, params, results_dir)
    return results

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_summary(results: Dict, N: int, n_nodes: int):
    print(f"\nMonte Carlo results (N={N} runs, {n_nodes} BSs / walls / RIS)")
    print(f"{'Metric':<20} {'No RIS':>22} {'With RIS':>22}")
    print("-" * 66)
    for key, label in [("handovers","Handovers"), ("hofs","HOFs"), ("pps","Ping-pongs")]:
        a = results["no_ris"][key]; b = results["ris"][key]
        print(f"{label:<20} {np.mean(a):>8.2f} +/- {np.std(a):<9.2f}"
              f" {np.mean(b):>8.2f} +/- {np.std(b):.2f}")
    print()


def _plot_results(results: Dict, params: Dict, results_dir: str):
    metrics = [("handovers", "Handovers per run"),
               ("hofs",      "HOFs per run"),
               ("pps",       "Ping-pongs per run")]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (key, title) in zip(axes, metrics):
        a = results["no_ris"][key]; b = results["ris"][key]
        max_val = max(a.max(), b.max(), 1)
        bins = np.arange(0, max_val + 2) - 0.5
        ax.hist(a, bins=bins, alpha=0.6, label="No RIS",   color="steelblue")
        ax.hist(b, bins=bins, alpha=0.6, label="With RIS", color="darkorange")
        ax.set_title(title)
        ax.set_xlabel("Count")
        ax.set_ylabel("Frequency")
        ax.legend()

    n = params["n_nodes"]
    fig.suptitle(
        f"RIS-assisted handover  |  {params['area_width']:.0f}x{params['area_height']:.0f} m  "
        f"|  {n} BSs, {n} walls, {n} RIS  |  N={params['N_runs']}",
        fontsize=9,
    )
    fig.tight_layout()
    out = os.path.join(results_dir, "network_handover_stats.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {out}")

    _plot_topology(params, results["bs_positions"],
                  results["walls"], results["ris_list"], results_dir)


def _plot_topology(
    params: Dict,
    bs_positions: np.ndarray,
    walls: List[Wall],
    ris_list: List[np.ndarray],
    results_dir: str,
):
    fig, ax = plt.subplots(figsize=(7, 7))
    W, H = params["area_width"], params["area_height"]

    # Walls
    for i, (p1, p2) in enumerate(walls):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", lw=3,
                solid_capstyle="round",
                label="Wall" if i == 0 else None)

    # RIS
    ris_arr = np.array(ris_list)
    ax.scatter(ris_arr[:, 0], ris_arr[:, 1], marker="^", s=100,
               color="limegreen", zorder=4, label="RIS")

    # BSs
    colors = ["royalblue", "firebrick", "darkorange", "purple",
              "teal", "deeppink", "olive", "navy"]
    for i, pos in enumerate(bs_positions):
        ax.scatter(*pos, marker="s", s=160, color=colors[i % len(colors)],
                   zorder=5, label=f"BS-{i}")

    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    n = params["n_nodes"]
    ax.set_title(f"Network topology  ({n} BSs, {n} walls, {n} RIS)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(results_dir, "network_topology.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Topology saved: {out}")
