# Simulation and Performance Analysis of Reconfigurable Intelligent Surfaces for Next-Generation Wireless Networks

Reconfigurable Intelligent Surfaces (RIS) are a key enabler for 6G wireless systems, capable of intelligently controlling the propagation of electromagnetic waves to enhance coverage, spectral efficiency and energy efficiency. This project will model and simulate RIS-assisted wireless communication systems, focusing on how RIS can improve link quality compared to traditional wireless channels. The project will involve implementing RIS-aided channel models in MATLAB/Python, simulating different RIS configurations and analyzing system performance in terms of SNR, capacity and energy efficiency.

## Project structure

```
thesis_code/
├── config/
│   └── params.json     # Simulation Parameters
├── src/
│   └──
├── scripts/
│   └──
├── results/
├── requirements.txt    #
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

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
