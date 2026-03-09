# Simulation and Performance Analysis of Reconfigurable Intelligent Surfaces for Next-Generation Wireless Networks

This repository contains the simulation code for the thesis. **Part 1** implements a baseline wireless system: direct link (S→D) and conventional relay (S→R→D, AF/DF). RIS will be added in a later part.

## Project structure

```
thesis_code/
├── config/
│   └── params.py          # Simulation parameters (freq, distances, trials, etc.)
├── src/
│   ├── channel/           # Path loss and fading models
│   ├── links/             # Direct and relay link models
│   ├── utils/             # dB conversions, noise power
│   └── run/               # Baseline Monte Carlo runner
├── scripts/
│   ├── run_baseline.py    # Run baseline simulation
│   └── plot_results.py   # Plot outage vs Tx power (example)
├── results/               # Outputs (gitignored)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ (for `float | np.ndarray` type hints; adjust if using an older interpreter).

## Running the baseline (Part 1)

From the project root:

```bash
python scripts/run_baseline.py
```

This runs a Monte Carlo simulation for the direct and relay (AF) links and writes a summary to `results/baseline_summary.txt`.

To generate an example outage vs Tx power plot:

```bash
python scripts/plot_results.py
```

Plots are saved under `results/` (e.g. `outage_vs_tx_power.pdf`, `.png`).

## Configuration

Edit `config/params.py` to change carrier frequency, distances (S–D, S–R, R–D), transmit power, SNR threshold, number of trials, and output directory.

## License

See repository or thesis for license and attribution.
