"""
2-D RIS-assisted handover simulation.

Topology (controlled by n_nodes):
  Area   : scales so each BS covers ~250 000 m²
  BSs    : n_nodes stations, uniform-random placement
  Walls  : one per unique nearest-neighbour BS pair, placed 20-40% of the way
           from BS_i toward its neighbour, random orientation
  RIS    : two per wall, one past each endpoint — best endpoint selected per step

The serving BS selects the best RIS from the full list.

Path-loss: Wei & Zhang (2025) 3.5 GHz experiment
  K_L = K_N = 10^{-4.33}, alpha_L = 1.73, alpha_N = 3.19
  G_bf = F_g * M_g^2  (M_g = 500 for 1000 m cells)

HOF: serving SNR < q_out_snr_db = -10 dB  (absolute threshold).
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, List, Optional, Tuple
from src.user import User
from src.handover import HandoverFSM, Wall
from src.utils import db2lin
from src.channel import noise_power as compute_noise_power

DEFAULT_NET_PARAMS: Dict[str, Any] = {
    # Area [m]
    "area_width":  1000.0,
    "area_height": 1000.0,
    # one wall + one RIS per unique nearest-neighbour BS pair
    "n_nodes":      50,
    # BS placement
    "bs_min_sep":   300.0,
    # Wall geometry
    "wall_half_len":    150.0,   # half-length of each wall [m]
    "ris_offset":        25.0,   # RIS placed this far past the wall tip [m]
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
    "q_out_snr_db": -10.0,
    "hyst_dB":       3.0,
    "ttt":           0.04,
    "tp":            1.0,
    # Mobility
    "speed": 30.0,
    # Simulation
    "dt":     0.05,
    "T_sim":  60.0,
    "N_runs": 500,
}

# Topology builders
def _place_bs(
    n: int,
    width: float,
    height: float,
    min_sep: float,
    rng: np.random.Generator,
) -> np.ndarray:
    # Uniform-random BS placement with minimum inter-site separation
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
    bs_positions: np.ndarray,
    rng: np.random.Generator,
) -> List[Wall]:
    """
    Place up to n walls, one per unique nearest-neighbour BS pair.
    Each wall centre sits 20–40% of the way from BS_i toward its neighbour
    (inside the serving cell, enabling HOFs), with a random orientation so
    the RIS endpoint geometry is favourable for the cascaded signal path.
    """
    n_bs = len(bs_positions)

    dists = np.linalg.norm(
        bs_positions[:, None, :] - bs_positions[None, :, :], axis=2
    )
    np.fill_diagonal(dists, np.inf)
    nn = np.argmin(dists, axis=1)   # shape (n_bs,)

    placed_pairs: set = set()
    walls: List[Wall] = []

    for i in rng.permutation(n_bs):
        if len(walls) >= n:
            break
        j    = int(nn[i])
        pair = frozenset((int(i), j))
        if pair in placed_pairs:
            continue
        placed_pairs.add(pair)

        bs_a, bs_b = bs_positions[i], bs_positions[j]
        t      = rng.uniform(0.2, 0.4)
        center = bs_a + t * (bs_b - bs_a)

        theta = rng.uniform(0.0, np.pi)
        dx    = half_len * np.cos(theta)
        dy    = half_len * np.sin(theta)

        p1 = np.clip(center + np.array([ dx,  dy]), [0.0, 0.0], [width, height])
        p2 = np.clip(center + np.array([-dx, -dy]), [0.0, 0.0], [width, height])
        walls.append((p1, p2))
    return walls


def _place_ris(walls: List[Wall], offset: float) -> List[np.ndarray]:
    """
    Two RIS per wall — one just past each endpoint.
    Both endpoints are geometrically valid relay points (the strict
    intersection check excludes t = 0 and t = 1, so a RIS sitting exactly
    at a tip sees both sides of the wall).  Placing at both endpoints means
    _ris_boost automatically selects whichever has the shorter cascaded path
    for the UE's current position, regardless of which side it entered from.
    """
    ris_list: List[np.ndarray] = []
    for p1, p2 in walls:
        wall_dir = (p2 - p1) / np.linalg.norm(p2 - p1)
        ris_list.append(p1 - offset * wall_dir)   # past p1 end
        ris_list.append(p2 + offset * wall_dir)   # past p2 end
    return ris_list

# Vectorised signal helpers
def _blocked_mask(src: np.ndarray, targets: np.ndarray,
                  walls: List[Wall]) -> np.ndarray:
    # True where the segment src -> targets[i] is strictly blocked by a wall
    d       = targets - src                                  # (n, 2)
    blocked = np.zeros(len(targets), dtype=bool)
    for w_p1, w_p2 in walls:
        e     = w_p2 - w_p1                                 # (2,)
        diff  = w_p1 - src                                  # (2,)
        cross = d[:, 0] * e[1] - d[:, 1] * e[0]            # (n,)
        t_num = float(diff[0] * e[1] - diff[1] * e[0])     # scalar
        u_num = diff[0] * d[:, 1] - diff[1] * d[:, 0]      # (n,)
        valid = np.abs(cross) > 1e-10
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(valid, t_num / cross, 0.0)
            u = np.where(valid, u_num / cross, 0.0)
        blocked |= valid & (t > 0.0) & (t < 1.0) & (u > 0.0) & (u < 1.0)
    return blocked

def _rsrp_vec(P_tx: float, ue_pos: np.ndarray, bs_positions: np.ndarray,
              K_L: float, K_N: float, alpha_L: float, alpha_N: float,
              walls: List[Wall]) -> np.ndarray:
    # Direct RSRP for all BSs in one vectorised call. Returns shape (n_bs,)
    d       = np.maximum(np.linalg.norm(bs_positions - ue_pos, axis=1), 1.0)
    blocked = _blocked_mask(ue_pos, bs_positions, walls)
    return np.where(blocked,
                    P_tx * K_N * d ** (-alpha_N),
                    P_tx * K_L * d ** (-alpha_L))

def _ris_boost(P_tx: float, ue_pos: np.ndarray, bs_pos: np.ndarray,
               S_direct: float, K_L: float, alpha_L: float,
               walls: List[Wall], ris_arr: np.ndarray,
               G_bf: float, chi_lin: float) -> float:
    # Return best (direct + RIS) RSRP for one BS. ris_arr shape (n_ris, 2)
    r   = np.maximum(np.linalg.norm(ris_arr - bs_pos,  axis=1), 1.0)
    dp  = np.maximum(np.linalg.norm(ris_arr - ue_pos,  axis=1), 1.0)
    clear = (~_blocked_mask(bs_pos, ris_arr, walls) &
             ~_blocked_mask(ue_pos, ris_arr, walls))
    S_cand = np.where(clear,
                      P_tx * (K_L ** 2) * G_bf * r ** (-alpha_L) * dp ** (-alpha_L),
                      0.0)
    best = float(np.max(S_cand))
    if best > max(0.0, chi_lin - 1.0) * S_direct:
        return S_direct + best
    return S_direct

# Single-trajectory simulation
def _simulate_one(
    params: Dict[str, Any],
    bs_positions: np.ndarray,
    walls: List[Wall],
    ris_list: List[np.ndarray],
    use_ris: bool,
    rng: np.random.Generator,
    P_noise: float,
) -> Dict[str, int]:
    W    = params["area_width"];  H = params["area_height"]
    K_L  = params["K_L"];  K_N  = params["K_N"]
    aL   = params["alpha_L"]; aN = params["alpha_N"]
    P_tx = params["P_tx"]
    G_bf    = params["F_g"] * (params["M_g"] ** 2)
    chi_lin = db2lin(params["chi_dB"])

    ris_arr = np.array(ris_list)   # (n_ris, 2) — pre-stacked once

    fsm = HandoverFSM(
        n_bs         = len(bs_positions),
        noise_power  = P_noise,
        ttt          = params["ttt"],
        hyst_db      = params["hyst_dB"],
        q_out_snr_db = params["q_out_snr_db"],
        tp           = params["tp"],
    )

    user = User(W, H, params["speed"], rng=rng)
    fsm.initialise_serving(
        _rsrp_vec(P_tx, user.position, bs_positions, K_L, K_N, aL, aN, walls)
    )

    dt = params["dt"]
    T  = params["T_sim"]
    t  = 0.0

    while t < T:
        pos      = user.step(dt)
        rsrp_arr = _rsrp_vec(P_tx, pos, bs_positions, K_L, K_N, aL, aN, walls)

        if use_ris:
            s = fsm.serving
            rsrp_arr[s] = _ris_boost(
                P_tx, pos, bs_positions[s], rsrp_arr[s],
                K_L, aL, walls, ris_arr, G_bf, chi_lin,
            )

        fsm.step(dt, rsrp_arr, t)
        t += dt

    return {
        "handovers": fsm.handover_count,
        "hofs":      fsm.hof_count,
    }

# Parallel worker
def _run_pair(
    args: Tuple,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Run one no-RIS / with-RIS pair from the same seed. Called in workers."""
    params, bs_positions, walls, ris_list, seed, P_noise = args
    r_no = _simulate_one(params, bs_positions, walls, ris_list, False,
                         np.random.default_rng(seed), P_noise)
    r_ri = _simulate_one(params, bs_positions, walls, ris_list, True,
                         np.random.default_rng(seed), P_noise)
    return r_no, r_ri

# Monte-Carlo run
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
    params["bs_min_sep"]    = 300.0          # fixed — cell radius is always ~500 m
    params["wall_half_len"] = 60.0   * scale

    if results_dir is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(root, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Fixed topology (same across all N_runs)
    topo_rng = np.random.default_rng(int(params["topo_seed"]))
    n     = int(params["n_nodes"])
    n_obs = n   # one wall per unique nearest-neighbour BS pair (natural limit)
    bs_positions = _place_bs(n, params["area_width"], params["area_height"],
                             params["bs_min_sep"], topo_rng)
    walls    = _place_walls(n_obs, params["area_width"], params["area_height"],
                            params["wall_half_len"], bs_positions, topo_rng)
    ris_list = _place_ris(walls, params["ris_offset"])
    P_noise = compute_noise_power(params["bandwidth"], params["noise_figure_dB"])

    print(f"Topology: {n} BSs, {n_obs} walls, {n_obs} RIS")
    for i, p in enumerate(bs_positions):
        print(f"  BS-{i}: ({p[0]:.0f}, {p[1]:.0f})")

    N    = int(params["N_runs"])
    rng  = np.random.default_rng(0)
    seeds = [int(rng.integers(0, 2 ** 31)) for _ in range(N)]
    args_list = [
        (params, bs_positions, walls, ris_list, s, P_noise) for s in seeds
    ]

    ho_no, hof_no = [], []
    ho_ri, hof_ri = [], []

    n_workers = os.cpu_count() or 1
    print(f"Running {N} pairs across {n_workers} workers …")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for done, (r_no, r_ri) in enumerate(pool.map(_run_pair, args_list), 1):
            ho_no.append(r_no["handovers"]); hof_no.append(r_no["hofs"])
            ho_ri.append(r_ri["handovers"]); hof_ri.append(r_ri["hofs"])
            if done % max(1, N // 10) == 0:
                print(f"  {done}/{N}  ({time.time()-t0:.1f}s)")

    results = {
        "no_ris": {
            "handovers": np.array(ho_no),
            "hofs":      np.array(hof_no),
        },
        "ris": {
            "handovers": np.array(ho_ri),
            "hofs":      np.array(hof_ri),
        },
        "bs_positions": bs_positions,
        "walls":        walls,
        "ris_list":     ris_list,
    }
    _print_summary(results, N, n)
    _plot_results(results, params, results_dir)
    return results

# Output helpers
def _print_summary(results: Dict, N: int, n_nodes: int):
    n_obs = len(results["walls"])
    print(f"\nMonte Carlo results (N={N} runs, {n_nodes} BSs, {n_obs} walls / RIS)")
    print(f"{'Metric':<20} {'No RIS':>22} {'With RIS':>22}")
    print("-" * 66)
    for key, label in [("handovers","Handovers"), ("hofs","HOFs")]:
        a = results["no_ris"][key]; b = results["ris"][key]
        print(f"{label:<20} {np.mean(a):>8.2f} +/- {np.std(a):<9.2f}"
              f" {np.mean(b):>8.2f} +/- {np.std(b):.2f}")
    print()

def _plot_results(results: Dict, params: Dict, results_dir: str):
    metrics = [("handovers", "Handovers per run"),
               ("hofs",      "HOFs per run")]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
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

    n_bs  = len(results["bs_positions"])
    n_obs = len(results["walls"])
    fig.suptitle(
        f"RIS-assisted handover  |  {params['area_width']:.0f}x{params['area_height']:.0f} m  "
        f"|  {n_bs} BSs, {n_obs} walls, {n_obs} RIS  |  N={params['N_runs']}",
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
    ax.scatter(ris_arr[:, 0], ris_arr[:, 1], marker="^", s=40,
               color="limegreen", zorder=4, label="RIS")

    # BSs — single legend entry for all
    colors = ["royalblue", "firebrick", "darkorange", "purple",
              "teal", "deeppink", "olive", "navy"]
    for i, pos in enumerate(bs_positions):
        ax.scatter(*pos, marker="s", s=60, color=colors[i % len(colors)],
                   zorder=5, label="BS" if i == 0 else None)

    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(
        f"Network topology  ({len(bs_positions)} BSs, "
        f"{len(walls)} walls, {len(ris_list)} RIS)"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(results_dir, "network_topology.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Topology saved: {out}")
