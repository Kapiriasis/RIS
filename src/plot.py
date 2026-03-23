import os
import matplotlib.pyplot as plt
import numpy as np

def ensure_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

def plot_snr_cdf(snr_linear, out_path, label="link"):
    snr_linear = np.asarray(snr_linear)
    snr_db = 10.0 * np.log10(snr_linear)
    snr_db_sorted = np.sort(snr_db)
    p = np.linspace(0.0, 1.0, snr_db_sorted.size, endpoint=False)
    ensure_dir(out_path)
    plt.figure()
    plt.plot(snr_db_sorted, p, label=label)
    plt.xlabel("SNR [dB]")
    plt.ylabel("Empirical CDF")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_capacity_hist(capacity_values, out_path, bins=40, label="link"):
    capacity_values = np.asarray(capacity_values)
    ensure_dir(out_path)
    plt.figure()
    plt.hist(capacity_values, bins=bins, density=True, alpha=0.7, label=label)
    plt.xlabel("Capacity [bits/s]")
    plt.ylabel("Density")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_snr_vs_elements(num_elements, mean_snr_db, out_path, label="RIS"):
    num_elements = np.asarray(num_elements)
    mean_snr_db = np.asarray(mean_snr_db)
    ensure_dir(out_path)
    plt.figure()
    plt.plot(num_elements, mean_snr_db, linewidth=2, label=label)
    plt.xlabel("Number of RIS elements")
    plt.ylabel("Mean SNR [dB]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
