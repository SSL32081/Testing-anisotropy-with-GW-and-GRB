#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

from utils import DATA_DIR, FIG_DIR, DPI, SINGLE, CF_LMAX, NSIDE, \
    read_grb_data, compute_skymap_from_points, compute_correlation_function


def main():
    # Read GW correlations
    gw_skymap = np.load(DATA_DIR / 'GWTC4p0_combined_galactic_skymap.npy')
    # Read GRB correlations
    grb_data = read_grb_data(DATA_DIR / "GRB_Summary_table.txt")

    obs_grb_dict = {
        'all': (grb_data, 'k', '--'),
        'short': (grb_data[grb_data['duration'] < 2.0], 'C1', '-'),
        'long': (grb_data[grb_data['duration'] >= 2.0], 'C2', '-')
    }

    theta_degs = np.linspace(0.0, 180.0, int(1e4))

    fig, ax = plt.subplots(1, 1, figsize=(SINGLE, 3.1))
    ax2 = ax.inset_axes([15, 0.18e-12, 165, 1.3e-12], transform=ax.transData)

    for grb_type in ('all', 'short', 'long'):
        grb_dataset, colour, linestyle = obs_grb_dict[grb_type]
        grb_map = compute_skymap_from_points(
                -grb_dataset['l_gal'], grb_dataset['b_gal'], NSIDE)
        grb_map /= grb_map.sum()
        print("Computing angular power spectrum...")
        cross_cl = hp.anafast(gw_skymap, map2=grb_map, lmax=CF_LMAX)
        # Tested, this version does not make a difference:
        # cross_cl = hp.anafast(hp.alm2map(hp.map2alm(gw_skymap / gw_skymap.sum(), lmax=CF_LMAX), nside=NSIDE),
        #                       map2=hp.alm2map(hp.map2alm(grb_map, lmax=CF_LMAX), nside=NSIDE), lmax=CF_LMAX)
        cross_cf = compute_correlation_function(
            cross_cl, theta_degs * np.pi / 180, CF_LMAX, windowed=True)

        label = grb_type.capitalize() + ' GRB x GW'
        ax.plot(theta_degs, cross_cf, c=colour, linestyle=linestyle)
        ax2.plot(theta_degs, cross_cf, c=colour, linestyle=linestyle, label=label)

    ax.set_ylim(0, 1.69e-12)
    ax2.set_xlim(-5, 185)

    indicator = ax.indicate_inset_zoom(ax2, edgecolor="grey")
    # Change the connecting lines
    for vis, path in zip((False, True, True, False), indicator[1]):
        path.set_visible(vis)

    # Ticks
    ax.set_xticks(30 * np.arange(7))
    ax2.set_xticks(30 * np.arange(7))
    ax2.minorticks_on()
    ax2.tick_params(which='minor', direction='in', left=False)
    ax2.legend()

    ax.set_xlabel(r'$\theta\,/\,{\rm deg}$')
    ax.set_ylabel(r'$C_{{\rm GW}\times{\rm GRB}}(\theta)$')
    ax.set_title('Angular cross-corelation between GW and GRB events')

    # Shift the sci. notation text to the left
    exp = ax.yaxis.get_offset_text()
    exp.set_x(-0.09)

    fig.savefig(FIG_DIR / 'Fig7_cross_correlation.pdf', dpi=DPI)
    return fig


if __name__ == '__main__':
    main()
