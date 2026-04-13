#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from utils import DATA_DIR, FIG_DIR, SINGLE, DPI, \
    read_grb_data, add_healpy_mollweide_ax
from copy import deepcopy
import healpy as hp
from matplotlib.colors import LogNorm, Normalize
from matplotlib import ticker


## GW skyamps
def plot_gw_skymap(skymaps, ax=plt.gca(), fig=plt.gcf(), vmax=None, numticks=None):
    # Hardcode number of events here, since the read-in data has lost that info
    N_events = 85
    cmap = 'viridis'

    if not vmax:
        vmax = np.percentile(skymaps[skymaps > 0], 99)
        numticks = None

    ax = add_healpy_mollweide_ax(fig, ax)
    ax.projmap(
        skymaps, nest=False,
        xsize=2600, coord='G',
        cmap=cmap, badcolor='gray', bgcolor='white',
<<<<<<< HEAD
        # vmin=0, vmax=np.percentile(skymaps[skymaps > 0], 99)
        vmin=0, vmax=5e-6,
=======
        vmin=0,
        vmax=vmax
>>>>>>> d992909 (Fix colorbar axes (vmax and numticks))
    )
    hp.graticule(verbose=False, dpar=30, dmer=30)
    ax.set_title(f"Combined GW Skymaps ({N_events} events)")

    im = ax.get_images()[0]
    fig.colorbar(
        im, ax=ax,
        pad=0.04, shrink=0.65,
        orientation="vertical",
        label=r'Probability Density $M_{\rm GW}(\chi,\phi)$'
    )
    cbar = ax.images[0].colorbar
    cbar.ax.yaxis.set_major_locator(ticker.LinearLocator(numticks=numticks))
    return fig, ax


## GRB
def plot_grb_skymap(grb_data, ax=plt.gca(), fig=plt.gcf()):
    # Separate long and short GRBs
    short_mask = grb_data['duration'] < 2.0
    long_mask = grb_data['duration'] >= 2.0
    short_grbs = grb_data[short_mask]
    long_grbs = grb_data[long_mask]

    log_norm = LogNorm(vmin=2, vmax=350)
    sc = ax.scatter(long_grbs['l_gal'], long_grbs['b_gal'],
                    c=long_grbs['duration'], s=2, marker="o",
                    cmap="winter", norm=log_norm, alpha=0.9,
                    edgecolors="none", label="Long GRBs",
                    rasterized=True)
    ax.scatter(short_grbs['l_gal'], short_grbs['b_gal'],
               s=5, c='red', marker="+",
               linewidths=0.7, alpha=0.9,
               label="Short GRBs", rasterized=True)

    fig.colorbar(sc, ax=ax, pad=0.04, shrink=0.65,
                 orientation="vertical",
                 label="Long GRB Burst Duration (s)")

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, linestyle="--", linewidth=1, alpha=0.9)
    ax.set_xticks(np.radians(np.arange(-180, 181, 30)))
    ax.set_yticks(np.radians(np.arange(-90, 91, 30)))
    line_kws = dict(color='k', linewidth=0.8, linestyle="-", alpha=0.5)
    ax.axhline(y=0, **line_kws)  # Equator
    ax.axvline(x=0, **line_kws)  # Prime meridian

    ax.set_title("GRB Skymap")
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.17),
              ncol=2, fancybox=False, frameon=False)
    return fig, ax


## GLADE+ galaxies
def plot_glade_skymap(glade_data, ax=plt.gca(), fig=plt.gcf()):
    NSIDE = 256
    theta = np.pi / 2 - glade_data['b_gal']  # colatitude
    phi = np.remainder(np.pi + glade_data['l_gal'], 2 * np.pi)  # longitude
    NPIX = hp.nside2npix(NSIDE)
    pix_indices = hp.ang2pix(NSIDE, theta, phi)

    # Create galaxy density map
    galaxy_map = np.zeros(NPIX)
    for pix in pix_indices:
        galaxy_map[pix] += 1

    # # We want counts here
    # galaxy_map = galaxy_map / np.sum(galaxy_map)
    vmin = galaxy_map[galaxy_map > 0].min()
    vmax = galaxy_map.max()
    cmap = 'plasma'

    ax = add_healpy_mollweide_ax(fig, ax)
    ax.projmap(
        galaxy_map, nest=False,
        xsize=1000, coord='G',
        cmap=cmap, badcolor='gray', bgcolor='white',
        norm='log', vmin=vmin, vmax=vmax
    )

    im = ax.get_images()[0]
    mappable = plt.cm.ScalarMappable(
        norm=Normalize(vmin=im.norm.vmin, vmax=im.norm.vmax), cmap=cmap
    )
    fig.colorbar(
        mappable, ax=ax,
        pad=0.04, shrink=0.65,
        orientation="vertical",
        label="Galaxies Count"
    )

    hp.graticule(verbose=False, dpar=30, dmer=30)
    ax.set_title("GLADE+ Galaxy Distribution")
    return fig, ax


def relocate_healpy_axes(ax, cb_ax, ref_ax_pos, ref_cb_ax_pos, shift):
    ax_pos = ax.get_position()
    new_ax_pos = deepcopy(ref_ax_pos)
    new_ax_pos.y0 = ax_pos.y0 - shift
    new_ax_pos.y1 = ax_pos.y0 + ref_ax_pos.height
    ax.set_position(new_ax_pos)

    cb_ax_pos = cb_ax.get_position()
    new_cb_pos = deepcopy(ref_cb_ax_pos)
    new_cb_pos.y0 = cb_ax_pos.y0 - shift
    new_cb_pos.y1 = cb_ax_pos.y0 + ref_cb_ax_pos.height
    cb_ax.set_position(new_cb_pos)
    return ax, cb_ax


def main():
    # Observed GW skyloc maps
    observed_map = np.load(DATA_DIR / 'GWTC4p0_combined_galactic_skymap.npy')
    # GRB locations
    grb_data = read_grb_data(DATA_DIR / "GRB_Summary_table.txt")
    # Galaxies catalogue from GLADE+
    glade_data = np.load(DATA_DIR / 'GLADE_galactic_coords.npy')

    fig, axes = plt.subplots(3, 1, figsize=(SINGLE, 6.4),
                             subplot_kw={'projection': 'mollweide'})

    plot_gw_skymap(observed_map, ax=axes[0], fig=fig, vmax=5e-6, numticks=6)
    plot_grb_skymap(grb_data, ax=axes[1], fig=fig)
    plot_glade_skymap(glade_data, ax=axes[2], fig=fig)

    # Manual Axes Adjustment for the healpy axes
    ax_poss = [ax.get_position() for ax in fig.axes]
    # After the removal and addition of healpy axes, here is the order:
    # GRB ax, GW ax, GW cb, GRB cb, GLADE ax, GLADE cb
    # GW Healpy axis and colourbar
    relocate_healpy_axes(
        fig.axes[1], fig.axes[2],
        ax_poss[0], ax_poss[3], shift=0.00
    )
    # Glade+ Healpy axis and colourbar
    relocate_healpy_axes(
        fig.axes[4], fig.axes[5],
        ax_poss[0], ax_poss[3], shift=0.04
    )

    # Cannot shift the top row axes up, as it will squeeze against the top
    # Shift all the lower axes instead.
    for idx, ax in enumerate(fig.axes):
        if idx not in (1, 2):
            new_pos = deepcopy(ax.get_position())
            new_pos.y0 -= 0.015
            new_pos.y1 -= 0.015
            ax.set_position(new_pos)

    fig.savefig(FIG_DIR / 'Fig1_observed_data.pdf', dpi=DPI)
    return fig, axes


if __name__ == "__main__":
    fig, axes = main()
