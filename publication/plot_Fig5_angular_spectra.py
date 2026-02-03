#!/usr/bin/env python3
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from utils import DATA_DIR, KEY, FIG_DIR, SINGLE, DPI, \
    NSIDE, CL_LMAX, CF_LMAX, \
    read_grb_data, compute_skymap_from_points
import healpy as hp

parser = argparse.ArgumentParser(
    description='Options for computing the gamma fit of synthetic data.')
parser.add_argument('--gammafit', action='store_true',
                    help='Whether to use gamma fit results for plotting.')

args = parser.parse_args()
USE_GAMMA_FIT = args.gammafit
mean_key = 'mean'
std_key = 'std'
if USE_GAMMA_FIT:
    mean_key = 'gamma_mean'
    std_key = 'gamma_std'


def shift_exponential_text(ax):
    exp = ax.yaxis.get_offset_text()
    exp.set_x(-0.05)
    return exp


def plot_gw_correlation(gw_skymap, gw_synth_stat, ells, ax):
    # Observed data
    print("Computing angular power spectrum...")
    # gw_map_ring = hp.reorder(gw_data, n2r=True)
    cl_obs = hp.anafast(gw_skymap, lmax=CF_LMAX)
    ax.plot(ells, cl_obs[1:CL_LMAX+1], c='C3',
            label='Observed GW Events', zorder=10)

    colour = 'k'
    alpha_dict = {1: 0.5, 2: 0.35, 3: 0.15}

    means = gw_synth_stat[mean_key][1:CL_LMAX+1]
    std = gw_synth_stat[std_key][1:CL_LMAX+1]
    for n_sigma in reversed(range(1, 4)):
        ax.fill_between(
            ells, means - n_sigma * std, means + n_sigma * std,
            alpha=alpha_dict[n_sigma], color=colour, linewidth=0)

    # Plot mean correlation function
    ax.plot(ells, gw_synth_stat[mean_key][1:CL_LMAX+1],
            c='C0', label='Synthetic GW')

    ax.minorticks_on()
    ax.set_ylabel(r'Power Spectrum $C_\ell$')
    ax.set_title('Synthetic vs Observed GW Skymaps')
    ax.grid(True, which='major', ls='--', lw=0.5)
    ax.legend(framealpha=0.5)
    shift_exponential_text(ax)


def plot_grb_correlation(grb_data, grb_synth_stats, ells, ax):
    obs_grb_dict = {
        'full': grb_data,
        'short': grb_data[grb_data['duration'] < 2.0],
        'long': grb_data[grb_data['duration'] >= 2.0]
    }

    colour_dict = { 'full': 'grey', 'short': 'C1', 'long': 'C2' }
    line_colour_dict = { 'full': 'k', 'short': 'darkorange', 'long': 'darkgreen' }
    alpha_dict = { 1:  0.5, 2:  0.35, 3:  0.15 }

    for idx, grb_type in enumerate(('full', 'short', 'long')):
        # Convert GRB data to HEALPix map
        print("Compute map for GRB data...")
        grb_dataset = obs_grb_dict[grb_type]
        grb_map = compute_skymap_from_points(grb_dataset['l_gal'], grb_dataset['b_gal'], NSIDE)
        print("Computing angular power spectrum...")
        cl_obs = hp.anafast(grb_map, lmax=CF_LMAX)
        print("Computing angular correlation function...")
        # Plot observed data
        ax.plot(ells, cl_obs[1:CL_LMAX+1],
                c=line_colour_dict[grb_type], zorder=10)

        # Synthetic data
        mean = grb_synth_stats[f'{grb_type}_grb_CL_gamma_fit'][mean_key][1:CL_LMAX+1]
        std = grb_synth_stats[f'{grb_type}_grb_CL_gamma_fit'][std_key][1:CL_LMAX+1]

        z_ord = idx + 1 if idx != 0 else 5

        for n_sigma in reversed(range(1, 4)):
            ax.fill_between(
                ells, mean - n_sigma * std, mean + n_sigma * std,
                alpha=alpha_dict[n_sigma], color=colour_dict[grb_type],
                linewidth=0, zorder=z_ord)

        # Plot mean correlation function
        ax.plot(ells, mean, line_colour_dict[grb_type], linestyle='--', zorder=z_ord+1)

    ax.set_yscale('log')
    ax.minorticks_on()
    ax.set_xlabel(r'Multipole moment $\ell$')
    ax.set_ylabel(r'Power Spectrum $C_\ell$')
    ax.set_title('Synthetic vs Observed GRB Skymaps')
    ax.grid(True, which='major', ls='--', lw=0.5)
    shift_exponential_text(ax)

    h1 = [
        mlines.Line2D([], [], color='k', linestyle='-', label='Observed GRBs'),
        mlines.Line2D([], [], color='grey', linestyle='--', label='Synthetic GRBs'),
    ]
    h2 = [
        mlines.Line2D([], [], color='k', linestyle='-', label='All GRBs'),
        mlines.Line2D([], [], color='darkorange', linestyle='-', label='Short GRBs'),
        mlines.Line2D([], [], color='darkgreen', linestyle='-', label='Long GRBs'),
    ]
    leg1 = ax.legend(handles=h1, loc='lower right', framealpha=0.5)
    ax.add_artist(leg1)
    ax.legend(handles=h2, loc='upper right', framealpha=0.5)

    return ax


def main():
    # Read GW correlations
    gw_skymap = np.load(DATA_DIR / 'GWTC4p0_combined_galactic_skymap.npy')
    gw_synth_fit = np.load(
        DATA_DIR / f'synthetic_gw{KEY}_correlation_CLCF_gamma_fit_n180_lmax128_windowed.npz')
    # Read GRB correlations
    grb_data = read_grb_data(DATA_DIR / 'GRB_Summary_table.txt')
    grb_synth_fit = np.load(
        DATA_DIR / 'synthetic_grb_correlation_CLCF_gamma_fit_n180_lmax128_windowed.npz')

    # Use the same ell range for all plots
    ell_range = np.arange(1, CL_LMAX + 1)

    fig, axes = plt.subplots(2, 1, figsize=(SINGLE, 4.7),
                             sharex=True, sharey=True)

    plot_gw_correlation(
        gw_skymap, gw_synth_fit['gw_CL_gamma_fit'], ell_range, axes[0]
    )
    plot_grb_correlation(grb_data, grb_synth_fit, ell_range, axes[1])
    axes[1].set_xlim(0, CL_LMAX + 0.5)

    suffix = f'{KEY}'
    if USE_GAMMA_FIT:
        suffix += '_gammafit'
    fig.savefig(FIG_DIR / f"Fig5_power_spectrum{suffix}.pdf", dpi=DPI)
    return fig, axes


if __name__ == '__main__':
    fig, axes = main()
