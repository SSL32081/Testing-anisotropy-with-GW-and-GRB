#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
from scipy.special import lpmv  # Associated Legendre polynomials
import matplotlib.pyplot as plt
from utils import DATA_DIR, FIG_DIR, DOUBLE, DPI, \
    read_grb_data
import healpy as hp
from matplotlib.colors import Normalize

LMAX = 26
NSIDE = 256
NPIX = hp.nside2npix(NSIDE)








def main():
    
    # Read GW skymaps
    gwtc4_skymap = np.load(DATA_DIR / 'GWTC4p0_combined_galactic_skymap.npy')
    o4a_synth_skymap = np.load(DATA_DIR / 'synthetic_O4a_combined_galactic_skymap.npy')
    # Plot with fill_between for nice shaded regions
    plt.figure(figsize=(12, 8))

    # Plot 3σ region first (so it's in the background)
    plt.fill_between(ell[1:], 
                        np.maximum(1e-16, cl_3sigma_lower[1:]),  # Avoid negative values for log scale
                        cl_3sigma_upper[1:], 
                        alpha=0.15, color='black')

    # Plot 2σ region on top
    plt.fill_between(ell[1:], 
                        np.maximum(1e-16, cl_2sigma_lower[1:]), 
                        cl_2sigma_upper[1:], 
                        alpha=0.35, color='black')

    # Plot 1σ region on top
    plt.fill_between(ell[1:], 
                        np.maximum(1e-16, cl_1sigma_lower[1:]), 
                        cl_1sigma_upper[1:], 
                        alpha=0.5, color='black')

    plt.plot(ell[1:], cl_mean[1:],  label=f'Synthetic GW')
    plt.plot(ell[1:], hp.anafast(observed_map, lmax=lmax)[1:], label='Observed GW')
    plt.yscale('log')
    plt.xlabel(r'Multipole moment $\ell$')
    plt.ylabel(r'Angular Power Spectrum $C_\ell$')
    plt.title(f'Angular Power Spectrum Comparison: Synthetic vs Observed')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()
    plt.show()
    return fig


if __name__ == "__main__":
    fig = main()