#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use('../matplotlibrc')
FIG_DIR = Path("../figures/")
DATA_DIR = Path("../data/")


## GRB 
def read_grb_data(file_path):
    # Load the data
    keys = ('ra', 'dec', 'pos_err', 'duration', 'redshift')
    dtypes = [(key, 'f8') for key in keys]
    arr = np.loadtxt(file_path, dtype=dtypes, usecols=(3,4,5,6,11))

    # Pre-process data
    mask = np.ones_like(arr, dtype=bool)
    for key in keys:
        if key == 'redshift':
            continue
        mask &= np.isfinite(arr[key]) & (arr[key] != -999)
    arr = arr[mask]

    # Convert RA, Dec to radians for Mollweide projection
    # SL: How is this different from just np.radians?
    coords = SkyCoord(ra=arr['ra'] * u.deg, dec=arr['dec'] * u.deg, frame="icrs")
    arr['ra'] = (360.0 * u.deg - coords.ra).wrap_at(180 * u.deg).radian
    arr['dec'] = coords.dec.radian
    return arr


def read_grb_simulated_data(file_path):
    with open(file_path) as f:
        header = f.readlines()[0]
        # Strip away the leading comment and closing newline character
        keys = header[2:-2].split(' ')
        dtypes = [(key, 'f8') for key in keys]
    simulated_grbs = np.loadtxt(file_path, dtype=dtypes)
    return simulated_grbs


def plot_grb_skymap(grb_data, ax=plt.gca(), fig=plt.gcf()):
    # Separate long and short GRBs
    short_mask = grb_data['duration'] < 2.0
    long_mask = grb_data['duration'] >= 2.0
    short_grbs = grb_data[short_mask]
    long_grbs = grb_data[long_mask]
    
    log_norm = LogNorm(vmin=2, vmax=350)
    sc = ax.scatter(long_grbs['ra'], long_grbs['dec'], 
                      c=long_grbs['duration'], s=2, marker="o",
                      cmap="winter", norm=log_norm, alpha=0.9,
                      edgecolors="none", label="Long GRBs", 
                      rasterized=True)
    ax.scatter(short_grbs['ra'], short_grbs['dec'], 
                 s=5, c='red', marker="+",
                 linewidths=0.7, alpha=0.9,
                 label="Short GRBs", rasterized=True)

    fig.colorbar(sc, ax=ax, pad=0.04, shrink=0.65, 
                 orientation="vertical", 
                 label="Long GRB Burst Duration (s)")

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, linestyle="-", linewidth=1, alpha=0.5)
    ax.set_xticks(np.radians(np.linspace(-180, 180, 13)))  # Fewer longitude lines
    ax.set_yticks(np.radians(np.linspace(-90, 90, 7)))    # Fewer latitude lines
    line_kws = dict(color='k', linewidth=0.8, linestyle="-", alpha=0.5)
    ax.axhline(y=0, **line_kws)  # Equator
    ax.axvline(x=0, **line_kws)  # Prime meridian

    ax.set_title("GRB Skymap")
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.17),
              ncol=2, fancybox=False, frameon=False)
    return fig, ax


def main():

    # Observed GW skyloc maps
    observed_map = np.load(DATA_DIR / 'GWTC-4_mixed_combined_skymap.npy')
    # GRB locations
    grb_data = read_grb_data(DATA_DIR / "GRB_Summary_table.txt")
    sim_grb_data = read_grb_simulated_data(DATA_DIR / "simulated_grbs.txt")
    # Galaxies catalogue from GLADE+
    

    fig, axes = plt.subplots(3, 1, figsize=(4, 6.5), 
                             subplot_kw={'projection': 'mollweide'})

    plot_grb_skymap(grb_data, ax=axes[1], fig=fig)

    fig.savefig(FIG_DIR / 'Fig1_observed_data.pdf', dpi=300)


if __name__ == "__main__":
    main()