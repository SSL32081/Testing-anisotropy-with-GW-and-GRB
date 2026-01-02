'''
Shared utility functions and variables for published results scripts.
'''
from pathlib import Path
import numpy as np
from numpy.lib import recfunctions as rfn
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt

plt.style.use('../matplotlibrc')
FIG_DIR = Path("../figures/")
DATA_DIR = Path("../data/")

SINGLE = 4.1  # inches, single column fig width
DOUBLE = 8.3  # inches, double column fig width
DPI = 450  # figure dpi

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

    coords = SkyCoord(ra=arr['ra'] * u.deg, dec=arr['dec'] * u.deg, frame="icrs")
    l_gal = coords.galactic.l.wrap_at(180 * u.deg).radian
    b_gal = coords.galactic.b.radian
    l_gal_wrapped = np.remainder(l_gal + np.pi, 2 * np.pi) - np.pi
    l_gal_wrapped = -l_gal_wrapped 

    _arr = rfn.append_fields(arr, 'l_gal', l_gal_wrapped, dtypes='f8')
    arr = rfn.append_fields(_arr, 'b_gal', b_gal, dtypes='f8')
    return arr


def read_grb_simulated_data(file_path):
    with open(file_path) as f:
        header = f.readlines()[0]
        # Strip away the leading comment and closing newline character
        keys = header[2:-2].split(' ')
        dtypes = [(key, 'f8') for key in keys]
    simulated_grbs = np.loadtxt(file_path, dtype=dtypes)
    return simulated_grbs