#!/usr/bin/env python3
import numpy as np
import healpy as hp
import sys
sys.path.append('../published_results/')
from utils import DATA_DIR, NSIDE, read_synthetic_GW_skymap, \
    compute_correlation_function

LMAX = 26
NPIX = hp.nside2npix(NSIDE)


def get_synthetic_GW_correlations(n_sims=1000, n_events=85):
    # The following splits the n_sims skymaps into as many groups as possible
    # based on the given n_events per group.
    indices = np.arange(n_sims)
    np.random.shuffle(indices)
    split_indices = np.arange(n_events, n_sims, n_events)
    split_arrays = np.array_split(indices, split_indices)

    print(f"Processing {len(split_arrays) - 1} synthetic GW skymap groups with {n_events} events each")

    # Initialise outputs
    cl_tot_array = []
    C_theta_array = []
    all_skymap = []

    # Process each group
    for arr in split_arrays:
        if arr.size != n_events:
            continue
        synthetic_map = np.zeros(NPIX)
        for idx in arr:
            skymap = read_synthetic_GW_skymap(idx)
            synthetic_map += skymap
        synthetic_map /= n_events

        all_skymap.append(synthetic_map)

        cl_synth = hp.anafast(synthetic_map, lmax=LMAX)  #an array of C_ell values 
        cl_tot_array.append(cl_synth) 

        C_theta = compute_correlation_function(cl_synth, thetas, LMAX)
        C_theta_array.append(C_theta)

    return np.array(all_skymap), np.array(cl_tot_array), np.array(C_theta_array)


if __name__ == "__main__":
    thetas = np.linspace(0.0, np.pi, int(1e3))
    gw_synth_skymap_stats = get_synthetic_GW_correlations()
    np.savez(DATA_DIR / f"congregated_synthetic_GW_correlation_stats_n{thetas.size:d}.npz",
             all_skymap=gw_synth_skymap_stats[0],
             multipole_spectrum=gw_synth_skymap_stats[1],
             angular_spectrum=gw_synth_skymap_stats[2])
    # Note on 2025/01/05: The accumulated skymap is no longer needed.