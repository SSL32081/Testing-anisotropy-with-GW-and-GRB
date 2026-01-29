'''
Shared utility functions and variables for all published results scripts.
'''
import os
from pathlib import Path
import numpy as np
from numpy.lib import recfunctions as rfn
from scipy.special import lpmv  # Associated Legendre polynomials
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import healpy as hp

# Most artistic settings as here:
plt.style.use('../matplotlibrc')
FIG_DIR = Path("../figures/")
DATA_DIR = Path("../data/")
# Synthetic O4a skymap FITS files directory (replace as appropriate)
PARENT_DIR = Path(os.environ.get('HANDON_REPO', './')) / 'LVK_skyloc_samples'
GWTC4_FITS_DIR = PARENT_DIR / 'GWTC4p0_skymaps'
GW_SYN_DATA_SET = 'O4a_SNR8_SFR'  # Options: O4a, O4a_SNR8, O4a_SNR8_SFR
SYN_O4A_FITS_DIR = PARENT_DIR / f'Synthetic_{GW_SYN_DATA_SET}_skymaps'
N_SIMS, KEY = {
    'O4a': (1000, ''),
    'O4a_SNR8': (2278, '_snr8'),
    'O4a_SNR8_SFR': (1477, '_snr8_sfr'),
}[GW_SYN_DATA_SET]

SINGLE = 4.1  # inches, single column fig width
DOUBLE = 8.3  # inches, double column fig width
DPI = 450  # figure dpi

# ell max to use for different purposes
CL_LMAX = 26
CF_LMAX = 128
LMAX = max(CF_LMAX, CL_LMAX)

# This is the nside that all GW skymaps are resized to
NSIDE = 256


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


def add_healpy_mollweide_ax(fig, ax):
    # This is to mimic the healpy.mollview function behaviour
    # Replcae the original axis with a healpy axis
    left, bottom, right, top = ax.get_position().extents
    extent = (left, bottom, right - left, top - bottom)
    fig.delaxes(ax)
    ax = hp.projaxes.HpxMollweideAxes(
        fig, extent, coord='G', flipconv='astro'
    )
    fig.add_axes(ax)
    return ax


def read_synthetic_gw_skymap(idx, nside=NSIDE):
    fit_file = SYN_O4A_FITS_DIR / f'H1L1_{idx}_galactic.fits.gz'
    skymap = hp.read_map(fit_file, nest=False)
    skymap_resized = hp.ud_grade(skymap, nside, power=-2)
    return skymap_resized / np.sum(skymap_resized)


def compute_skymap_from_points(l_rad, b_rad, nside):
    """
    Build a HEALPix map from point data in Galactic coordinates.

    healpy.ang2pix default expects:
      theta = colatitude (radians),
      phi = longitude (radians) when lonlat=False.

    co-latitude: 0 to π  (N to S)
    longitude: 0 to 2π

    galactic latitude b: π/2 to -π/2 (N to S)
    galactic longitude l: 0 to 2π
    """
    npix = hp.nside2npix(nside)
    theta = np.pi / 2.0 - b_rad
    phi = l_rad

    # Ring-ordering to be compatible with anafast
    ipix = hp.ang2pix(nside, theta, phi, nest=False)
    counts = np.zeros(npix, dtype=float)
    np.add.at(counts, ipix, 1.0)

    if counts.sum() == 0:
        return counts
    return counts / counts.sum()


def compute_correlation_function(cl, thetas, lmax, lmax_res=26, windowed=True):
    """
    Compute angular correlation function C(θ) from Cℓ.

    Formula: C(θ) = 1/(4π) * Σ_{ℓ=0}^{ℓmax} (2ℓ+1) Cℓ Pℓ(cos θ)
    """
    resolution = 1 / lmax_res
    ells = np.arange(lmax + 1)
    cos_theta = np.cos(thetas)
    cf_theta = np.zeros_like(thetas, dtype=np.float64)

    for ell in ells:
        # Legendre polynomial Pℓ(cos θ)
        _cf_theta = (2 * ell + 1) * cl[ell] * lpmv(0, ell, cos_theta)
        if windowed:
            _cf_theta *= np.exp(-ell * (ell + 1) * (resolution ** 2))
        cf_theta += _cf_theta

    return cf_theta / (4 * np.pi)
