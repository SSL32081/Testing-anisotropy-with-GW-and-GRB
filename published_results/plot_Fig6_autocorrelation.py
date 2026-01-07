#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
from scipy.special import lpmv  # Associated Legendre polynomials
import matplotlib.pyplot as plt
from utils import DATA_DIR, FIG_DIR, SINGLE, DPI, \
    read_grb_data, compute_correlation_function
import healpy as hp
from matplotlib.colors import Normalize

LMAX = 26
NSIDE = 256
NPIX = hp.nside2npix(NSIDE)

def plot_glade_correlation(glade_data, fig, ax):
    print("Computing angular power spectrum...")
    cl_obs = hp.anafast(glade_data, lmax=LMAX)

    # Define theta range for correlation function 
    theta_degrees = np.linspace(0.0, 180.0, 10000)
    theta_degrees = np.sort(theta_degrees)

    # Compute correlation function
    print("Computing angular correlation function...")
    C_theta_obs = compute_correlation_function(cl_obs, theta_degrees, LMAX)

    ax.plot(theta_degrees, C_theta_obs, 'C3', label='GLADE+ Galaxies')

    ax.set_xlabel(r'Angular separation $\theta$ [degrees]')
    ax.set_ylabel(r'Correlation function C($\theta$)')
    # ax.set_ylim(1e-14,1e-10) #NOTE: this may affect visual interpretation though
    ax.set_title('Angular Correlation Function: GLADE+ Galaxy Distribution')
    ax.set_yscale('log')
    ax.grid(True, which='both', ls='--', lw=0.5)
    ax.legend()


def main():
    # Read GW correlations
    gw_corr = np.load(DATA_DIR / 'congregated_synthetic_GW_correlation_stats.npz')
    # Read GRB correlations
    grb_corr = np.load(DATA_DIR / 'congregated_synthetic_grb_correlation_stats.npz')
    # Read GLADE+ data
    glade_data = np.load(DATA_DIR / 'GLADE_galactic_coords.npy')

    fig, axes = plt.subplots(3, 1, figsize=(SINGLE, 15), constrained_layout=True)

    ax = axes[0]
    

    fig.savefig(FIG_DIR / 'Fig6_autocorrelations.pdf', dpi=DPI)
    return fig


if __name__ == '__main__':
    main()