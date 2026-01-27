#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from utils import DATA_DIR, FIG_DIR, DOUBLE, DPI, NSIDE, CL_LMAX, \
    read_grb_data, compute_skymap_from_points
import healpy as hp
from matplotlib.colors import Normalize


# The following two functions are imported from mapXmap_utils.py
# make skymap from coordinates data
# map blurring to lmax
def blur_map(skymap, lmax=CL_LMAX, nside=NSIDE,
             remove_monopole=False, tol=1e-10):
    alm = hp.map2alm_lsq(skymap, lmax=lmax, mmax=lmax, tol=tol)
    alm = alm[0]
    if remove_monopole:
        alm[0] = 0.0

    blurred_map = hp.alm2map(alm, nside=nside)
    return blurred_map


# normalize skymap
def normalize_skymap(skymap, shift_min=False):
    if shift_min:
        skymap = skymap - skymap.min()
    return skymap / skymap.sum()


def preprocess_skymaps(skymap, lmax=CL_LMAX, nside=NSIDE):
    skymap = blur_map(skymap, lmax=lmax, nside=nside)
    skymap = normalize_skymap(skymap, shift_min=True)
    return skymap


def plot_colourbar(fig, ax, im, label):
    mappable = plt.cm.ScalarMappable(
        norm=Normalize(vmin=im.norm.vmin, vmax=im.norm.vmax),
        cmap=im.get_cmap())
    cb = fig.colorbar(mappable, ax=ax,
                      orientation="horizontal",
                      location='bottom',
                      label=label, shrink=0.7, pad=0.05)
    return cb


def main():
    # Observed GW skyloc maps
    observed_map = np.load(DATA_DIR / 'GWTC4p0_combined_galactic_skymap.npy')
    gw_skymap = preprocess_skymaps(observed_map)
    # GRB locations
    grb_data = read_grb_data(DATA_DIR / "GRB_Summary_table.txt")
    grb_skymap = compute_skymap_from_points(
        l_rad=grb_data['l_gal'], b_rad=grb_data['b_gal'], nside=NSIDE)
    grb_skymap = preprocess_skymaps(grb_skymap)

    fig = plt.figure(figsize=(DOUBLE, 2.8), constrained_layout=False)
    PLOT_KWS = {
        'min': 0.0,
        'norm': 'hist',
        'graticule': True,
        'graticule_labels': False,
        'longitude_grid_spacing': 30,
        'projection_type': 'mollweide',
        'cbar': False,
    }

    im0 = hp.projview(gw_skymap, fig=fig, sub=(1, 2, 1), **PLOT_KWS)
    im1 = hp.projview(grb_skymap, fig=fig, sub=(1, 2, 2), **PLOT_KWS)

    # Set title and colorbar in post, so as to use our default style
    axes = fig.axes
    axes[0].set_title(rf'GWTC-4 GW Localization Skymap ($\ell \leq {CL_LMAX}$)')
    plot_colourbar(fig, axes[0], im0,
                   label=r'Rescaled Probability Density' + '\n' +
                           r'$M_\text{GW}\left(\chi, \phi\right)$')
    axes[1].set_title(rf'GRB Event Density Skymap ($\ell \leq {CL_LMAX}$)')
    plot_colourbar(fig, axes[1], im1,
                   label=r'Normalized Event Density' + '\n' +
                           r'$M_\text{GRB}\left(\chi, \phi\right)$')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'Fig4_GWnGRB_harmonics_lmax26.pdf',
                bbox_inches='tight', dpi=DPI)
    return fig


if __name__ == "__main__":
    fig = main()
