#!/usr/bin/env python3
import numpy as np
import healpy as hp
from multiprocessing import Pool
import sys
sys.path.append('../published_results/')
from utils import DATA_DIR, NSIDE, read_synthetic_GW_skymap, \
    compute_correlation_function

LMAX = 128
NPIX = hp.nside2npix(NSIDE)

N_SIMS = 1000  # Total number of synthetic GW skymaps
N_EVENTS = 85  # Number of GW events to simulate per synthetic skymap


def process_one_group(idx_arr):
    if idx_arr.size != N_EVENTS:
        return None, None, None
    synthetic_map = np.zeros(NPIX)
    for idx in idx_arr:
        skymap = read_synthetic_GW_skymap(idx)
        synthetic_map += skymap
    synthetic_map /= N_EVENTS
    cl_synth = hp.anafast(synthetic_map, lmax=LMAX)  #an array of C_ell values 
    cf_synth = compute_correlation_function(cl_synth, thetas, LMAX)
    return synthetic_map, cl_synth, cf_synth


def get_synthetic_GW_correlations(n_sims=N_SIMS, n_events=N_EVENTS):
    # The following splits the n_sims skymaps into as many groups as possible
    # based on the given N_EVENTS per group.
    indices = np.arange(n_sims)
    np.random.shuffle(indices)
    split_indices = np.arange(n_events, n_sims, n_events)
    split_arrays = np.array_split(indices, split_indices)

    print(f"Processing {len(split_arrays) - 1} synthetic GW skymap groups with {n_events} events each")

    # Process each group
    with Pool(processes=20) as pool:
        results = pool.map(process_one_group, split_arrays)

    dtypes = [
        ('skymap', 'f8', (NPIX,)),
        ('multipole_spectrum', 'f8', (LMAX + 1,)),
        ('angular_spectrum', 'f8', (thetas.size,))
    ]
    return np.array([row for row in results if row[0] is not None], dtype=dtypes)


if __name__ == "__main__":
    thetas = np.linspace(0.0, np.pi, int(1000))
    gw_synth_skymap_stats = get_synthetic_GW_correlations()
    np.save(DATA_DIR / f"congregated_synthetic_GW_correlation_stats_{N_SIMS:d}_{N_EVENTS:d}_ellmax_{LMAX:d}_n{thetas.size:d}.npy",
            gw_synth_skymap_stats)
    # Note on 2025/01/05: The accumulated skymap is no longer needed.
    # Note on 2026/01/08: Update to save as npy.