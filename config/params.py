"""Simulation parameters for baseline (Part 1)."""

# Carrier and bandwidth
FREQ_HZ = 2.4e9
BANDWIDTH_HZ = 1e6
NOISE_FIGURE_DB = 0.0

# Geometry (m)
D_SD_M = 100.0   # source–destination
D_SR_M = 50.0    # source–relay
D_RD_M = 50.0    # relay–destination

# Propagation
PATH_LOSS_EXPONENT = 2.0

# Transmit power and threshold
TX_POWER_DBM = 20.0
SNR_THRESHOLD_DB = 0.0

# Monte Carlo
N_TRIALS = 10_000
SEED = 42

# Output
OUTPUT_DIR = "results"
