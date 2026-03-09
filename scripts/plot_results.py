"""Plot baseline results (outage vs SNR or similar). Re-run run_baseline with sweeps to generate data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.run import run_baseline


def main() -> None:
    # Example: sweep Tx power (SNR) and plot outage
    tx_power_dbm_list = np.linspace(0, 30, 16)
    outage_d = []
    outage_r = []
    for tx in tx_power_dbm_list:
        res = run_baseline(
            tx_power_dbm=float(tx),
            n_trials=5_000,
            seed=42,
        )
        outage_d.append(res["outage_prob_direct"])
        outage_r.append(res["outage_prob_relay"])

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.semilogy(tx_power_dbm_list, outage_d, "o-", label="Direct")
    plt.semilogy(tx_power_dbm_list, outage_r, "s-", label="Relay (AF)")
    plt.xlabel("Tx power (dBm)")
    plt.ylabel("Outage probability")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "outage_vs_tx_power.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "outage_vs_tx_power.png", bbox_inches="tight")
    plt.close()
    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
