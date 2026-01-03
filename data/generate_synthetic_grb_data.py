#!/usr/bin/env python3
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import curve_fit
from scipy.stats import norm

import sys
sys.path.append('../published_results/')
from utils import DATA_DIR, read_grb_data


def bimodal_log(x, A1, mu1, sigma1, A2, mu2, sigma2):
    '''Bimodal Gaussian function in log-space.

    Remark: Not normalised.
    '''
    return (A1 * norm.pdf(np.log10(x), mu1, sigma1) +
            A2 * norm.pdf(np.log10(x), mu2, sigma2))

grb_data = read_grb_data(DATA_DIR / "GRB_Summary_table.txt")
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
    popt, pcov = curve_fit(bimodal_log, bin_centers, counts, p0=p0)
    
    print("Fit parameters (A1, mu1, sigma1, A2, mu2, sigma2):")
    print(popt)

except RuntimeError:
    print("Fit failed. Could not find optimal parameters.")


# generate the GRBs with the assumption of isotropic distribution for positions,
# and the same redshift distribution as observed and the duration distribution as observed
num_simulated_grbs = grb_data.size * 1
print(f"Generating {num_simulated_grbs} simulated GRBs.")

# Initialize output structured array
keys = ('ra', 'dec', 'l_gal', 'b_gal', 'duration')
dtypes = [(key, 'f8') for key in keys]
dtypes.append(('flag', 'S1'))
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
num_short = int(num_simulated_grbs * n_short / grb_data.size)
num_long = num_simulated_grbs - num_short

# The bimodal fit was performed on log10(duration),
# so we generate from a normal distribution and exponentiate base 10.
sim_duration_short = 10.0 ** np.random.normal(popt[1], popt[2], num_short)
sim_duration_long  = 10.0 ** np.random.normal(popt[4], popt[5], num_long)
sim_duration = np.concatenate([sim_duration_short, sim_duration_long])
output_arr['duration'] = sim_duration
sim_flags = np.concatenate([['s'] * n_short, ['l'] * n_long])
output_arr['flag'] = sim_flags

# Saving the simulated GRB data
np.save(DATA_DIR / 'simulated_grbs.npy', output_arr)