import json
import os
from scripts.direct import run_direct
from scripts.ris import run_ris
from scripts.relay import run_relay_df
from scripts.ber import run_ber
from src.datagen import DEFAULT_PARAMS, PARAMS_PATH, generate_params
from src.plot import plot_snr_cdf_comparison

def load_params() -> dict:
    # Ensure params.json exists and is non-empty, then load it.
    exists = os.path.exists(PARAMS_PATH)
    size = os.path.getsize(PARAMS_PATH) if exists else 0

    # If file does not exist or is empty, (re)generate it.
    if (not exists) or size == 0:
        generate_params()

    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    merged = DEFAULT_PARAMS.copy()
    merged.update(loaded)
    return merged

def main() -> None:
    params = load_params()

    # Run direct-link baseline
    direct_metrics = run_direct(params)

    # Run relay (decode-and-forward) baseline
    relay_metrics = run_relay_df(params)

    # Run BER vs. SNR comparison (direct, RIS-only, RIS+direct, theoretical)
    run_ber(params)

    # Run RIS-assisted link
    ris_metrics = run_ris(params)
    ris_summary = {
        "mean_snr_db": ris_metrics["mean_snr_db"],
        "outage_prob_5dB": ris_metrics["outage_prob_5dB"],
        "mean_capacity_bits_per_s": ris_metrics["mean_capacity_bits_per_s"],
    }

    # Combined SNR CDF comparison across all three configurations
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    plot_snr_cdf_comparison(
        snr_linear_list=[
            direct_metrics["snr_linear"],
            relay_metrics["snr_linear"],
            ris_metrics["snr_linear"],
        ],
        labels=["Direct", "Relay (DF)", "RIS"],
        out_path=os.path.join(results_dir, "snr_cdf_comparison.png"),
    )

    print("=== Direct link metrics ===")
    print(direct_metrics)
    print("\n=== Relay (DF) metrics ===")
    print(relay_metrics)
    print("\n=== RIS metrics ===")
    print(ris_summary)

if __name__ == "__main__":
    main()
