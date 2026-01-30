#!/usr/bin/env python3
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.transforms import \
    TransformedBbox, blended_transform_factory
from mpl_toolkits.axes_grid1.inset_locator import \
    BboxConnector, BboxPatch
from utils import DATA_DIR, KEY, FIG_DIR, SINGLE, DPI, NSIDE, CF_LMAX, \
    read_grb_data, compute_correlation_function, compute_skymap_from_points
import healpy as hp

parser = argparse.ArgumentParser(
    description='Options for computing the gamma fit of synthetic data.')
parser.add_argument('--ntheta', type=int, default=180,
                    help='Number of theta bins for correlation function computation.')
parser.add_argument('--lmax', type=int, default=CF_LMAX,
                    help='Maximum multipole moment lmax for correlation function computation.')
parser.add_argument('--nowindow', action='store_true', 
                    help='Whether to apply a resolution-limited window function in the correlation function computation.')
parser.add_argument('--gammafit', action='store_true', default=False,
                    help='Whether to use gamma fit results for plotting.')


DEG2RAD = np.pi / 180.0
NPIX = hp.nside2npix(NSIDE)

args = parser.parse_args()
LMAX = args.lmax
NTHETAS = args.ntheta
WINDOWED = not args.nowindow
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


def plot_gw_correlation(gw_skymap, gw_synth_stat, thetas, ax):
    # Observed data
    print("Computing angular power spectrum...")
    # gw_map_ring = hp.reorder(gw_data, n2r=True)
    cl_obs = hp.anafast(gw_skymap, lmax=CF_LMAX)
    C_theta_obs = compute_correlation_function(
        cl_obs, thetas * DEG2RAD, LMAX, windowed=WINDOWED)
    ax.plot(thetas, C_theta_obs, 'C3', zorder=10, label='Observed GW Events')

    colour = 'k'
    alpha_dict = {1: 0.5, 2: 0.35, 3: 0.15}

    thetas = np.linspace(0.0, 180.0, gw_synth_stat[mean_key].size)
    for n_sigma in reversed(range(1, 4)):
        ax.fill_between(
            thetas,
            gw_synth_stat[mean_key] - n_sigma * gw_synth_stat[std_key],
            gw_synth_stat[mean_key] + n_sigma * gw_synth_stat[std_key],
            alpha=alpha_dict[n_sigma],
            color=colour, linewidth=0)

    # Plot mean correlation function
    ax.plot(thetas, gw_synth_stat[mean_key], 'C0', label='Synthetic GW')

    # ax.set_xlabel(r'Angular separation $\theta$ [degrees]')
    ax.set_ylabel(r'Autocorrelation function $C(\theta)$')
    ax.set_title('Synthetic vs Observed GW Skymaps')
    ax.grid(True, which='both', ls='--', lw=0.5)
    ax.legend(framealpha=0.5)
    shift_exponential_text(ax)


# The following are copied from:
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/axes_zoom_effect.html
def connect_bbox(bbox1, bbox2,
                 loc1a, loc2a, loc1b, loc2b,
                 prop_lines, prop_patches=None):
    if prop_patches is None:
        prop_patches = {
            **prop_lines,
            "alpha": prop_lines.get("alpha", 1) * 0.2,
            "clip_on": False,
        }

    c1 = BboxConnector(
        bbox1, bbox2, loc1=loc1a, loc2=loc2a, clip_on=False, **prop_lines)
    c2 = BboxConnector(
        bbox1, bbox2, loc1=loc1b, loc2=loc2b, clip_on=False, **prop_lines)

    bbox_patch1 = BboxPatch(bbox1, **prop_patches)
    bbox_patch2 = BboxPatch(bbox2, **prop_patches)

    return c1, c2, bbox_patch1, bbox_patch2


def zoom_axis(ax1, ax2, **kwargs):
    """
    ax1 : the main Axes
    ax1 : the zoomed Axes

    Similar to zoom_effect01.  The xmin & xmax will be taken from the
    ax1.viewLim.
    """

    tt = ax1.transScale + (ax1.transLimits + ax2.transAxes)
    trans = blended_transform_factory(ax2.transData, tt)

    mybbox1 = ax1.bbox
    mybbox2 = TransformedBbox(ax1.viewLim, trans)

    prop_patches = {**kwargs, "ec": "none", "alpha": 0.2}

    c1, c2, bbox_patch1, bbox_patch2 = connect_bbox(
        mybbox1, mybbox2,
        loc1a=3, loc2a=2, loc1b=4, loc2b=1,
        prop_lines=kwargs, prop_patches=prop_patches)

    # ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    # From: https://stackoverflow.com/a/57241978
    c1.set_in_layout(False)
    c2.set_in_layout(False)
    ax2.add_patch(c1)
    ax2.add_patch(c2)

    return c1, c2, bbox_patch1, bbox_patch2


def plot_grb_correlation(grb_data, grb_synth_stats, thetas, 
                         *axes):
    obs_grb_dict = {
        'full': grb_data,
        'short': grb_data[grb_data['duration'] < 2.0],
        'long': grb_data[grb_data['duration'] >= 2.0]
    }

    colour_dict = {'full': 'grey', 'short': 'C1', 'long': 'C2'}
    line_colour_dict = {'full': 'k', 'short': 'darkorange', 'long': 'darkgreen'}
    alpha_dict = {1: 0.5, 2: 0.35, 3: 0.15}

    axes[1].set_xlim(0, 35)
    axes[2].set_xlim(90, 150)
    axes[1].set_ylim(1.50e-12, 2.6e-12)
    axes[2].set_ylim(1.585e-12, 1.645e-12)
    zoom_axis(axes[1], axes[0], alpha=0.8, color='C0')
    zoom_axis(axes[2], axes[0], alpha=0.8, color='gold')

    for ax in axes:
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
            mean, std = grb_synth_stats[f'{grb_type}_grb_CF_gamma_fit'][mean_key], \
                grb_synth_stats[f'{grb_type}_grb_CF_gamma_fit'][std_key]

            thetas = np.linspace(0.0, 180.0, mean.size)
            for n_sigma in reversed(range(1, 4)):
                ax.fill_between(
                    thetas, mean - n_sigma * std, mean + n_sigma * std,
                    alpha=alpha_dict[n_sigma], color=colour_dict[grb_type],
                    linewidth=0)

            # Plot mean correlation function
            ax.plot(thetas, mean, line_colour_dict[grb_type], linestyle='--')

        # ax.set_xlabel(r'Angular separation $\theta$ [degrees]')
        shift_exponential_text(ax)

    ax = axes[0]
    ax.set_ylabel(r'Autocorrelation function $C(\theta)$')
    ax.set_title('Synthetic vs Observed GRB Skymaps')
    h1 = [
        mlines.Line2D([], [], color='k', linestyle='-', label='Observed GRBs'),
        mlines.Line2D([], [], color='grey', linestyle='--', label='Synthetic GRBs'),
    ]
    h2 = [
        mlines.Line2D([], [], color='k', linestyle='-', label='All GRBs'),
        mlines.Line2D([], [], color='darkorange', linestyle='-', label='Short GRBs'),
        mlines.Line2D([], [], color='darkgreen', linestyle='-', label='Long GRBs'),
    ]
    leg1 = ax.legend(handles=h1, loc='upper center', framealpha=0.5)
    ax.add_artist(leg1)
    ax.legend(handles=h2, loc='upper right', framealpha=0.5)
    ax.grid(True, which='both', ls='--', lw=0.5)

    for ax in axes[1:]:
        ax.minorticks_on()
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        ax.grid(True, which='major', ls='--', lw=0.4)

    return ax


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
    ax.set_title('GLADE+ Galaxy Distribution')
    ax.grid(True, which='both', ls='--', lw=0.5)
    shift_exponential_text(ax)


def main():
    suffix = f"n{NTHETAS:d}_lmax{LMAX:d}"
    if WINDOWED:
        suffix += "_windowed"

    # Read GW correlations
    gw_skymap = np.load(DATA_DIR / 'GWTC4p0_combined_galactic_skymap.npy')
    gw_synth_fit = np.load(
        DATA_DIR / f'synthetic_gw{KEY}_correlation_CLCF_gamma_fit_{suffix}.npz')
    # Read GRB correlations
    grb_data = read_grb_data(DATA_DIR / 'GRB_Summary_table.txt')
    grb_synth_fit = np.load(
        DATA_DIR / f'synthetic_grb_correlation_CLCF_gamma_fit_{suffix}.npz')
    # Read GLADE+ data
    glade_data = np.load(DATA_DIR / 'GLADE_galactic_coords.npy')

    # Use the same theta values for all plots
    theta_degs = np.linspace(0.0, 180.0, int(1e4))

    fig = plt.figure(figsize=(SINGLE, 8))
    axes = fig.subplot_mosaic([
        ["gw", "gw"],
        ["zoom1", "zoom2"],
        ["grb", "grb"],
        ["gal", "gal"],
    ], height_ratios=[1, 0.55, 0.95, 1]
    )

    plot_gw_correlation(
        gw_skymap, gw_synth_fit['gw_CF_gamma_fit'], theta_degs, axes['gw']
    )
    plot_grb_correlation(grb_data, grb_synth_fit, theta_degs, 
                         axes['grb'], axes['zoom1'], axes['zoom2'])
    plot_glade_correlation(glade_data, theta_degs, axes['gal'])
    
    for key in ('gw', 'grb', 'gal'):
        axes[key].set_xlim(0, 180)

    fig.get_layout_engine().set(w_pad=0.01, wspace=0)
    
    if USE_GAMMA_FIT:
        suffix += '_gammafit'
    fig.savefig(FIG_DIR / f"Fig6_autocorrelations_{suffix}.pdf", dpi=DPI)
    return fig, axes


if __name__ == '__main__':
    fig, axes = main()
