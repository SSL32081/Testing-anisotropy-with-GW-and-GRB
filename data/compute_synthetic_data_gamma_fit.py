#!/usr/bin/env python3
import numpy as np
from scipy.stats import gamma 
from multiprocessing import Pool
import argparse
import sys
sys.path.append('../published_results/')
from utils import DATA_DIR, CF_LMAX, N_SIMS, KEY, compute_correlation_function

parser = argparse.ArgumentParser(description='Options for computing the gamma fit of synthetic data.')
parser.add_argument('--ntheta', type=int, default=180,
                    help='Number of theta bins for correlation function computation.')
parser.add_argument('--lmax', type=int, default=CF_LMAX, 
                    help='Maximum multipole moment lmax for correlation function computation.')
parser.add_argument('--nowindow', action='store_false', default=True, 
                    help='Whether to apply a resolution-limited window function in the correlation function computation.')


def _gamma_fit(synth_data_row):
    scale = 1e13  # Regularising values for easier fitting.
    fitted_params = gamma.fit(synth_data_row * scale, method='MLE')
    return gamma.mean(*fitted_params) / scale, gamma.std(*fitted_params) / scale, \
        np.mean(synth_data_row), np.std(synth_data_row)


def mp_gamma_fit(synth_data):
    with Pool(processes=20) as pool:
        results = pool.map(_gamma_fit, synth_data)
    return np.array(results, dtype=[('gamma_mean', 'f8'), ('gamma_std', 'f8'), ('mean', 'f8'), ('std', 'f8')])


def recompute_correlation_function(cl_data, cf_data, lmax, windowed=False, nthetas=500):
    if cf_data.shape[1] == nthetas:
        return cf_data.T
    print('Resizing correlation function data...')
    thetas = np.linspace(0, np.pi, nthetas)
    output_cf = []
    for spectrum in cl_data:
        cf = compute_correlation_function(
            spectrum, thetas, lmax=lmax, windowed=windowed)
        output_cf.append(cf)
    return np.array(output_cf).T


def main():
    args = parser.parse_args()
    NTHETAS = args.ntheta
    LMAX = args.lmax
    WINDOWED = not args.nowindow

    suffix = f"n{NTHETAS:d}_lmax{LMAX:d}"
    if WINDOWED:
        suffix += "_windowed"

    # Read synthetic data
    gw_synth_data = np.load(DATA_DIR / f'congregated_synthetic_gw{KEY}_correlation_stats_{N_SIMS}_85_lmax128_n1000.npy')
    print('(Multi-)Processing gamma fit for synthetic GW correlations...')
    gw_gamma_fits = {}
    gw_gamma_fits['gw_CL_gamma_fit'] = mp_gamma_fit(
        gw_synth_data['multipole_spectrum'].T
    )
    gw_gamma_fits['gw_CF_gamma_fit'] = mp_gamma_fit(
        recompute_correlation_function(
            gw_synth_data['multipole_spectrum'],
            gw_synth_data['angular_spectrum'],
            lmax=LMAX, windowed=WINDOWED, nthetas=NTHETAS 
        )
    )
    np.savez(DATA_DIR / f'synthetic_gw{KEY}_correlation_CLCF_gamma_fit_{suffix}', **gw_gamma_fits)

    grb_synth_data = np.load(DATA_DIR / 'congregated_synthetic_grb_correlation_stats_1000_lmax128_n1000.npy')
    print('(Multi-)Processing gamma fit for synthetic GRB correlations...')
    grb_gamma_fits = {}
    for grb_type in ('full', 'short', 'long'):
        print(f'  - {grb_type.capitalize()} GRB sample')
        grb_gamma_fits[f'{grb_type}_grb_CL_gamma_fit'] = mp_gamma_fit(
            grb_synth_data[f'{grb_type}_multipole_spectrum'].T
        )
        grb_gamma_fits[f'{grb_type}_grb_CF_gamma_fit'] = mp_gamma_fit(
            recompute_correlation_function(
                grb_synth_data[f'{grb_type}_multipole_spectrum'],
                grb_synth_data[f'{grb_type}_angular_spectrum'],
                lmax=LMAX, windowed=WINDOWED, nthetas=NTHETAS 
            )
        )

    np.savez(DATA_DIR / f'synthetic_grb_correlation_CLCF_gamma_fit_{suffix}', **grb_gamma_fits)


if __name__ == "__main__":
    main()
