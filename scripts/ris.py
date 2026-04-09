import os
import numpy as np
from typing import Any, Dict, Optional
from src.plot import plot_snr_vs_elements
from src.ris_channel import simulate_ris_link

def _default_results_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "results")

def run_ris(params: Dict[str, Any], results_dir: Optional[str] = None) -> Dict[str, Any]:
    # Run RIS simulation and generate standard output plots.
    if results_dir is None:
        results_dir = _default_results_dir()

    sim_out = simulate_ris_link(params)
    cap = sim_out["capacity_bits_per_s"]

    # Sweep RIS size from 1..M and track mean SNR.
    max_elements = int(params["ris_array_size"])
    element_counts = np.arange(1, max_elements + 1, dtype=int)
    mean_snr_db_curve = []
    for m in element_counts:
        sweep_params = dict(params)
        sweep_params["ris_array_size"] = int(m)
        sweep_out = simulate_ris_link(sweep_params, include_direct=False)
        mean_snr_db_curve.append(sweep_out["metrics"]["mean_snr_db"])

    snr_vs_elements_path = os.path.join(results_dir, "ris_snr_vs_elements.png")
    plot_snr_vs_elements(
        num_elements=element_counts,
        mean_snr_db=mean_snr_db_curve,
        out_path=snr_vs_elements_path,
        label="RIS",
    )

    metrics = dict(sim_out["metrics"])
    metrics["snr_vs_elements"] = {
        "num_elements": element_counts.tolist(),
        "mean_snr_db": list(mean_snr_db_curve),
    }
    metrics["snr_linear"] = sim_out["snr_linear"]
    metrics["capacity"] = cap
    return metrics
