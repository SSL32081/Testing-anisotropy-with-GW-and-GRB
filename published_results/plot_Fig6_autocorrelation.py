#!/usr/bin/env python3
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from utils import DATA_DIR, FIG_DIR, SINGLE, DPI, NSIDE, CF_LMAX, \
    read_grb_data, compute_correlation_function, compute_skymap_from_points
import healpy as hp


parser = argparse.ArgumentParser(description='Options for computing the gamma fit of synthetic data.')
parser.add_argument('--ntheta', type=int, default=180,
                    help='Number of theta bins for correlation function computation.')
parser.add_argument('--lmax', type=int, default=CF_LMAX, 
                    help='Maximum multipole moment lmax for correlation function computation.')
parser.add_argument('--windowed', action='store_true',
                    help='Whether to apply a resolution-limited window function in the correlation function computation.')
parser.add_argument('--gammafit', action='store_true',
                    help='Whether to use gamma fit results for plotting.')


DEG2RAD = np.pi / 180.0
NPIX = hp.nside2npix(NSIDE)

args = parser.parse_args()
LMAX = args.lmax
NTHETAS = args.ntheta
WINDOWED = args.windowed
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


def plot_glade_correlation(glade_data, thetas, ax):
    print("Compute map for GLADE+ data...")
    galaxy_map = compute_skymap_from_points(
        glade_data['l_gal'], glade_data['b_gal'], NSIDE
    )

    print("Computing angular power spectrum...")
    cl_obs = hp.anafast(galaxy_map, lmax=CF_LMAX)

    # Compute correlation function
    print("Computing angular correlation function...")
    C_theta_obs = compute_correlation_function(
        cl_obs, thetas * DEG2RAD, LMAX, windowed=WINDOWED)

    ax.plot(thetas, C_theta_obs, 'C3', label='GLADE+ Galaxies')

    ax.set_xlabel(r'Angular separation $\theta$ [degrees]')
    ax.set_ylabel(r'Autocorrelation function $C(\theta)$')
    # ax.set_ylim(1e-14,1e-10) #NOTE: this may affect visual interpretation though
    ax.set_title('GLADE+ Galaxy Distribution')
    # ax.set_yscale('log')
    ax.grid(True, which='both', ls='--', lw=0.5)
    # ax.legend()
    shift_exponential_text(ax)


def plot_gw_correlation(gw_skymap, gw_synth_stat, thetas, ax):
    # Observed data
    print("Computing angular power spectrum...")
    # gw_map_ring = hp.reorder(gw_data, n2r=True)
    cl_obs = hp.anafast(gw_skymap, lmax=CF_LMAX)
    C_theta_obs = compute_correlation_function(
        cl_obs, thetas * DEG2RAD, LMAX, windowed=WINDOWED)
    ax.plot(thetas, C_theta_obs, 'C3', zorder=10, label='Observed GW Events')
    
    colour = 'k'
    alpha_dict = { 1:  0.5, 2:  0.35, 3:  0.15 }

    thetas = np.linspace(0.0, 180.0, gw_synth_stat[mean_key].size)
    for n_sigma in reversed(range(1, 4)):
        ax.fill_between(
            thetas, 
            gw_synth_stat[mean_key] - n_sigma * gw_synth_stat[std_key],
            gw_synth_stat[mean_key] + n_sigma * gw_synth_stat[std_key], 
            alpha=alpha_dict[n_sigma],
            color=colour, linewidth=0)

    # Plot mean correlation function
    ax.plot(thetas, gw_synth_stat[mean_key], 'C0', label=f'Synthetic GW')

    # ax.set_xlabel(r'Angular separation $\theta$ [degrees]')
    ax.set_ylabel(r'Autocorrelation function $C(\theta)$')
    ax.set_title('Synthetic vs Observed GW Skymaps')
    ax.grid(True, which='both', ls='--', lw=0.5)
    ax.legend(framealpha=0.5)
    shift_exponential_text(ax)


def plot_grb_correlation(grb_data, grb_synth_stats, thetas, ax):
    obs_grb_dict = {
        'full': grb_data,
        'short': grb_data[grb_data['duration'] < 2.0],
        'long': grb_data[grb_data['duration'] >= 2.0]
    }

    colour_dict = { 'full': 'grey', 'short': 'C1', 'long': 'C2' }
    line_colour_dict = { 'full': 'k', 'short': 'darkorange', 'long': 'darkgreen' }
    alpha_dict = { 1:  0.5, 2:  0.35, 3:  0.15 }

    for grb_type in ('full', 'short', 'long'):
        # Convert GRB data to HEALPix map
        print("Compute map for GRB data...")
        grb_dataset = obs_grb_dict[grb_type]
        grb_map = compute_skymap_from_points(grb_dataset['l_gal'], grb_dataset['b_gal'], NSIDE)
        print("Computing angular power spectrum...")
        cl_obs = hp.anafast(grb_map, lmax=CF_LMAX)
        print("Computing angular correlation function...")
        C_theta_obs = compute_correlation_function(
            cl_obs, thetas * DEG2RAD, LMAX, windowed=WINDOWED)
        # Plot observed data
        thetas = np.linspace(0.0, 180.0, C_theta_obs.size)
        ax.plot(thetas, C_theta_obs, line_colour_dict[grb_type], zorder=10)

        # Synthetic data
        mean, std = grb_synth_stats[f'{grb_type}_grb_gamma_fit'][mean_key], \
            grb_synth_stats[f'{grb_type}_grb_gamma_fit'][std_key]
        
        thetas = np.linspace(0.0, 180.0, mean.size)
        for n_sigma in reversed(range(1, 4)):
            ax.fill_between(
                thetas, mean - n_sigma * std, mean + n_sigma * std, 
                alpha=alpha_dict[n_sigma], color=colour_dict[grb_type], 
                linewidth=0)

        # Plot mean correlation function
        ax.plot(thetas, mean, line_colour_dict[grb_type], linestyle='--')

    # ax.set_xlabel(r'Angular separation $\theta$ [degrees]')
    ax.set_ylabel(r'Autocorrelation function $C(\theta)$')
    ax.set_title('Synthetic vs Observed GRB Skymaps')
    ax.grid(True, which='both', ls='--', lw=0.5)
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
    leg1 = ax.legend(handles=h1, loc='upper right', framealpha=0.5)
    ax.add_artist(leg1)
    ax.legend(handles=h2, loc='right', framealpha=0.5)

    return ax


def main():
    suffix = f"n{NTHETAS:d}_lmax{LMAX:d}"
    if WINDOWED:
        suffix += "_windowed"

    # Read GW correlations
    gw_skymap = np.load(DATA_DIR / 'GWTC4p0_combined_galactic_skymap.npy')
    gw_synth_fit = np.load(DATA_DIR / f'synthetic_gw_correlation_CF_gamma_fit_{suffix}.npy')
    # Read GRB correlations
    grb_data = read_grb_data(DATA_DIR / 'GRB_Summary_table.txt')
    grb_synth_fit = np.load(DATA_DIR / f'synthetic_grb_correlation_CF_gamma_fit_{suffix}.npz')
    # Read GLADE+ data
    glade_data = np.load(DATA_DIR / 'GLADE_galactic_coords.npy')

    # Use the same theta values for all plots
    theta_degs = np.linspace(0.0, 180.0, int(1e4))

    fig, axes = plt.subplots(3, 1, figsize=(SINGLE, 7), 
                             sharex=True, height_ratios=[1, 1.5, 1])

    plot_gw_correlation(gw_skymap, gw_synth_fit, theta_degs, axes[0])
    plot_grb_correlation(grb_data, grb_synth_fit, theta_degs, axes[1])
    plot_glade_correlation(glade_data, theta_degs, axes[2])
    axes[2].set_xlim(0, 180)

    if USE_GAMMA_FIT:
        suffix += '_gammafit'
    fig.savefig(FIG_DIR / f"Fig6_autocorrelations_{suffix}.pdf", dpi=DPI)
    return fig, axes


if __name__ == '__main__':
    fig, axes = main()