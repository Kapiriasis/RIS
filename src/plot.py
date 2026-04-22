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

def plot_snr_cdf_comparison(snr_linear_list, labels, out_path):
    """
    Plot empirical SNR CDFs for multiple configurations on one axes.

    Parameters
    ----------
    snr_linear_list : list of 1-D arrays, one per configuration (linear scale)
    labels          : list of strings matching snr_linear_list
    out_path        : file path to save the figure
    """
    ensure_dir(out_path)
    plt.figure()
    for snr_lin, label in zip(snr_linear_list, labels):
        snr_db = 10.0 * np.log10(np.asarray(snr_lin))
        snr_db_sorted = np.sort(snr_db)
        p = (np.arange(snr_db_sorted.size) + 1) / snr_db_sorted.size
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

def plot_capacity_hist_comparison(capacity_list, labels, out_path, bins=40):
    """
    Plot capacity histograms for multiple configurations on one axes.

    Parameters
    ----------
    capacity_list : list of 1-D arrays, one per configuration
    labels        : list of strings matching capacity_list
    out_path      : file path to save the figure
    bins          : number of histogram bins
    """
    ensure_dir(out_path)
    plt.figure()
    for cap, label in zip(capacity_list, labels):
        plt.hist(np.asarray(cap), bins=bins, density=True, alpha=0.5, label=label)
    plt.xlabel("Capacity [bits/s]")
    plt.ylabel("Probability Density [s/bits]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_ber_vs_snr(snr_db, ber_curves, labels, out_path):
    """
    Plot BER vs. average SNR for one or more curves on a semilogy axis.

    Parameters
    ----------
    snr_db     : 1-D array of SNR values in dB (shared x-axis)
    ber_curves : list of 1-D arrays, one per curve
    labels     : list of strings matching ber_curves
    out_path   : file path to save the figure
    """
    snr_db = np.asarray(snr_db)
    ensure_dir(out_path)
    plt.figure()
    for ber, label in zip(ber_curves, labels):
        plt.semilogy(snr_db, np.asarray(ber), label=label)
    plt.xlabel("$E_b/N_0$ [dB]")
    plt.ylabel("BER")
    plt.ylim([1e-4, 1])
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
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
