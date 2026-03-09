"""Entry point: run baseline simulation (direct + relay) and save results."""

import sys
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.run import run_baseline

try:
    from config.params import (
        FREQ_HZ,
        BANDWIDTH_HZ,
        NOISE_FIGURE_DB,
        D_SD_M,
        D_SR_M,
        D_RD_M,
        PATH_LOSS_EXPONENT,
        TX_POWER_DBM,
        SNR_THRESHOLD_DB,
        N_TRIALS,
        SEED,
        OUTPUT_DIR,
    )
except ImportError:
    # Defaults if config not used
    FREQ_HZ = 2.4e9
    BANDWIDTH_HZ = 1e6
    NOISE_FIGURE_DB = 0.0
    D_SD_M, D_SR_M, D_RD_M = 100.0, 50.0, 50.0
    PATH_LOSS_EXPONENT = 2.0
    TX_POWER_DBM = 20.0
    SNR_THRESHOLD_DB = 0.0
    N_TRIALS = 10_000
    SEED = 42
    OUTPUT_DIR = "results"


def main() -> None:
    results = run_baseline(
        freq_hz=FREQ_HZ,
        bandwidth_hz=BANDWIDTH_HZ,
        noise_figure_db=NOISE_FIGURE_DB,
        d_sd_m=D_SD_M,
        d_sr_m=D_SR_M,
        d_rd_m=D_RD_M,
        path_loss_exponent=PATH_LOSS_EXPONENT,
        tx_power_dbm=TX_POWER_DBM,
        snr_threshold_db=SNR_THRESHOLD_DB,
        n_trials=N_TRIALS,
        seed=SEED,
        output_dir=OUTPUT_DIR,
    )
    # Save summary (full arrays can be saved here or in run_baseline)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "baseline_summary.txt", "w") as f:
        f.write(f"Outage prob (direct): {results['outage_prob_direct']:.6f}\n")
        f.write(f"Outage prob (relay):   {results['outage_prob_relay']:.6f}\n")
        f.write(f"Params: {results['params']}\n")
    print(f"Outage (direct): {results['outage_prob_direct']:.4f}")
    print(f"Outage (relay):  {results['outage_prob_relay']:.4f}")
    print(f"Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
