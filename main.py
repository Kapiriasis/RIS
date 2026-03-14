import json
import os

from scripts.direct import run_direct
from scripts.relay import run_relay_df
from src.datagen import PARAMS_PATH, generate_params

def load_params() -> dict:
    # Ensure params.json exists and is non-empty, then load it.
    exists = os.path.exists(PARAMS_PATH)
    size = os.path.getsize(PARAMS_PATH) if exists else 0

    # If file does not exist or is empty, (re)generate it.
    if (not exists) or size == 0:
        generate_params()

    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def main() -> None:
    params = load_params()

    # Run direct-link baseline
    direct_metrics = run_direct(params)

    # Run relay (decode-and-forward) baseline
    relay_metrics = run_relay_df(params)

    print("=== Direct link metrics ===")
    print(direct_metrics)
    print("\n=== Relay (DF) metrics ===")
    print(relay_metrics)

if __name__ == "__main__":
    main()
