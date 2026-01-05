#!/usr/bin/env python3
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp
from scipy.optimize import curve_fit
from scipy.stats import norm

import sys
sys.path.append('../published_results/')
from utils import DATA_DIR, read_grb_data, compute_grb_skymap, \
    compute_correlation_function

# Initialize output structured array
keys = ('ra', 'dec', 'l_gal', 'b_gal', 'duration')
dtypes = [(key, 'f8') for key in keys]
dtypes.append(('flag', 'S1'))

LMAX = 26
NSIDE = 128
N_realisations = 100
thetas = np.linspace(0.0, np.pi, int(1e4))


def bimodal_log(x, A1, mu1, sigma1, A2, mu2, sigma2):
    '''Bimodal Gaussian function in log-space.

    Remark: Not normalised.
    '''
    return (A1 * norm.pdf(np.log10(x), mu1, sigma1) +
            A2 * norm.pdf(np.log10(x), mu2, sigma2))


def fit_real_grb_data(grb_data):
    grb_durations = grb_data['duration']
    short_grbs = grb_data[grb_durations < 2.0]
    long_grbs = grb_data[grb_durations >= 2.0]
    n_short = short_grbs.size
    n_long = long_grbs.size

    # Curve Fitting
    # Create log-spaced bins from the minimum to the maximum duration value
    bins = np.logspace(np.log10(grb_durations.min()), np.log10(grb_durations.max()), 50)
    # Calculate histogram
    counts, bin_edges = np.histogram(grb_durations, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initial guesses for the parameters
    # Based on the visual separation of short (<2s) and long (>2s) GRBs
    p0 = [
        n_short, np.log10(0.3), 0.5,  # A1, mu1, sigma1 for short GRBs
        n_long, np.log10(30), 0.5    # A2, mu2, sigma2 for long GRBs
    ]

    # Fit the bimodal function to the histogram data
    # We use the log of bin_centers for the x-data in the fit function
    try:
        popt, _ = curve_fit(bimodal_log, bin_centers, counts, p0=p0)
        
        print("Fit parameters (A1, mu1, sigma1, A2, mu2, sigma2):")
        print(popt)

    except RuntimeError:
        print("Fit failed. Could not find optimal parameters.")
    return popt


def generate_synthetic_grb_data(coeff, num_simulated_grbs):
    # generate the GRBs with the assumption of isotropic distribution for positions,
    # and the same redshift distribution as observed and the duration distribution as observed
    output_arr = np.zeros(num_simulated_grbs, dtype=dtypes)

    # isotropic distribution for positions on the sphere:
    # l ~ Uniform(0, 2π), sin(b) ~ Uniform(-1, 1)
    u_rand, v_rand = np.random.uniform(0.0, 1.0, (2, num_simulated_grbs))

    # galactic coordinates (in radians, as plain floats)
    l_rad = 2.0 * np.pi * v_rand                  # [0, 2π)
    b_rad = np.arcsin(2.0 * u_rand - 1.0)         # [-π/2, π/2]
    output_arr['l_gal'] = l_rad
    output_arr['b_gal'] = b_rad

    # convert to ICRS (RA, Dec) – here we attach units exactly once
    sim_coords = SkyCoord(l=l_rad * u.rad, b=b_rad * u.rad, frame="galactic").icrs

    # calculate x and y for mollweide projection
    # RA: wrap at 180 deg, in radians; Dec directly in radians
    output_arr['ra'] = sim_coords.ra.wrap_at(180 * u.deg).radian
    output_arr['dec'] = sim_coords.dec.radian

    # duration distribution from the fitted bimodal distribution
    prob_short = coeff[0] / (coeff[0] + coeff[3])
    num_short = np.random.binomial(num_simulated_grbs, prob_short)
    num_long = num_simulated_grbs - num_short

    # The bimodal fit was performed on log10(duration),
    # so we generate from a normal distribution and exponentiate base 10.
    sim_duration_short = np.random.normal(coeff[1], coeff[2], num_short)
    sim_duration_long  = np.random.normal(coeff[4], coeff[5], num_long)
    sim_duration = 10.0 ** np.concatenate([sim_duration_short, sim_duration_long])
    output_arr['duration'] = sim_duration
    # Which Gaussian do these points come from?
    sim_flags = np.concatenate([['l'] * num_short, ['r'] * num_long])
    output_arr['flag'] = sim_flags
    return output_arr


def get_grb_skymaps_and_spectra(full_grb_data):
    durations = full_grb_data['duration']
    short_grbs = full_grb_data[durations < 2.0]
    long_grbs = full_grb_data[durations >= 2.0]
    skymap_full = compute_grb_skymap(full_grb_data['l_gal'], full_grb_data['b_gal'], nside=NSIDE)
    skymap_short = compute_grb_skymap(short_grbs['l_gal'], short_grbs['b_gal'], nside=NSIDE)
    skymap_long = compute_grb_skymap(long_grbs['l_gal'], long_grbs['b_gal'], nside=NSIDE)

    # Compute angular power spectra
    cl_full = hp.anafast(skymap_full, lmax=LMAX) 
    cl_short = hp.anafast(skymap_short, lmax=LMAX)
    cl_long = hp.anafast(skymap_long, lmax=LMAX)

    # Compute auto-correlation functions
    Ctheta_full = compute_correlation_function(cl_full, thetas, LMAX)
    Ctheta_short = compute_correlation_function(cl_short, thetas, LMAX)
    Ctheta_long = compute_correlation_function(cl_long, thetas, LMAX)

    return cl_full, cl_short, cl_long, Ctheta_full, Ctheta_short, Ctheta_long

def main():
    grb_data = read_grb_data(DATA_DIR / "GRB_Summary_table.txt")
    coeff = fit_real_grb_data(grb_data)
    all_correlations = [] * 6
    for idx in range(N_realisations):
        simulated_grbs = generate_synthetic_grb_data(coeff, grb_data.size)
        correlations = get_grb_skymaps_and_spectra(simulated_grbs)
        keys = ('cl_full', 'cl_short', 'cl_long', 'Ctheta_full', 'Ctheta_short', 'Ctheta_long')
        np.savez(DATA_DIR / f'simulated_grbs/simulated_grbs_realisation_{idx:d}.npz', 
                 simulated_grbs=simulated_grbs, **dict(zip(keys, correlations)))
        for accumulator, element in zip(all_correlations, correlations):
            accumulator.append(element)
    all_correlations = [np.array(corr) for corr in all_correlations]
    keys = ('full_mulipole_spectrum', 'short_multipole_spectrum', 'long_multipole_spectrum',
            'full_angular_spectrum', 'short_angular_spectrum', 'long_angular_spectrum')
    np.savez(DATA_DIR / 'cogregrated_grb_correlation_stats.npz',
            **dict(zip(keys, all_correlations)))

if __name__ == "__main__":
    main()