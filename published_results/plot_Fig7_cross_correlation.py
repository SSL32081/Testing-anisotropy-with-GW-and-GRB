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


def corr_func(theta, Cl):
    c = 0
    x = np.cos(theta)
    for ell in range(len(Cl)):
        c += (1 + 2 * ell) * Cl[ell] * legendre(ell)(x)
    return c / (4 * np.pi)


def main():
    # Read GW correlations
    gw_skymap = np.load(DATA_DIR / 'GWTC4p0_combined_galactic_skymap.npy')
    # Read GRB correlations
    grb_corr = np.load(DATA_DIR / 'congregated_synthetic_grb_correlation_stats.npz')

    fig, ax = plt.subplots(1, 1, figsize=(SINGLE, 2.1), constrained_layout=True)

    if list_depth(Cls) >= 1:
        if Cl_labels is None:
            Cl_labels = [None] * len(Cls)
        if colors is None:
            colors = [None] * len(Cls)
        if linestyles is None:
            linestyles = [None] * len(Cls)
        for Cl, label, color, ls in zip(Cls, Cl_labels, colors, linestyles):
            ax.plot(theta_vals, corr_func(np.radians(theta_vals), Cl),
                    label=label, color=color, linestyle=ls)
    else:
        ax.plot(theta_vals, corr_func(np.radians(theta_vals), Cls),
                label=Cl_labels, color=colors, linestyle=linestyles)

    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')

    ax.set_xlabel(r'$\theta\,/\,\textrm{deg}$')
    ax.set_ylabel(ylabel if ylabel is not None else r'Correlation, $C(\theta)$')
    ax.set_title(title if title is not None else r'Angular Correlation Function $C(\theta)$')
    ax.legend()
    ax.grid(True)

    if save:
        filename = f'{title}{save_ext}'.replace(" ", "")
        fig.savefig(filename, dpi=600)

    fig.savefig(FIG_DIR / 'Fig7_cross_correlation.pdf', dpi=DPI)
    return fig


if __name__ == '__main__':
    main()
