#!/usr/bin/env python3
import numpy as np
from scipy.stats import gamma 
from multiprocessing import Pool
import sys
sys.path.append('../published_results/')
from utils import DATA_DIR, compute_correlation_function


def _gamma_fit(synth_data_row):
    scale = 1e12  # Regularising values for better fitting.
    fitted_params = gamma.fit(synth_data_row * scale, method='MLE')
    return gamma.mean(*fitted_params) / scale, gamma.std(*fitted_params) / scale, np.mean(synth_data_row), np.std(synth_data_row)


def mp_gamma_fit(synth_data):
    with Pool(processes=20) as pool:
        results = pool.map(_gamma_fit, synth_data)
    return np.array(results, dtype=[('gamma_mean', 'f8'), ('gamma_std', 'f8'), ('mean', 'f8'), ('std', 'f8')])


def recompute_correlation_function(cl_data, cf_data, nthetas=500):
    if cf_data.shape[1] == nthetas:
        return cf_data.T
    print('Resizing correlation function data...')
    thetas = np.linspace(0, np.pi, nthetas)
    output_cf = []
    for spectrum in cl_data:
        cf = compute_correlation_function(spectrum, thetas, lmax=128)
        output_cf.append(cf)
    return np.array(output_cf).T


def main():
    NTHETAS = 180
    # Read synthetic data
    gw_synth_data = np.load(DATA_DIR / 'congregated_synthetic_gw_correlation_stats_1000_85_ellmax_128_n1000.npy')
    print('(Multi-)Processing gamma fit for synthetic GW correlations...')
    gw_fit_results = mp_gamma_fit(
        recompute_correlation_function(
            gw_synth_data['multipole_spectrum'],
            gw_synth_data['angular_spectrum'],
            nthetas=NTHETAS
        )
    )
    np.save(DATA_DIR / 'synthetic_gw_correlation_CF_gamma_fit.npy', gw_fit_results)

    grb_synth_data = np.load(DATA_DIR / 'congregated_synthetic_grb_correlation_stats_100_lmax128_n1000.npy')
    print('(Multi-)Processing gamma fit for synthetic GRB correlations...')
    grb_gamma_fits = {}
    for grb_type in ('full', 'short', 'long'):
        print(f'  - {grb_type.capitalize()} GRB sample')
        grb_gamma_fits[f'{grb_type}_grb_gamma_fit'] = mp_gamma_fit(
            recompute_correlation_function(
                grb_synth_data[f'{grb_type}_multipole_spectrum'],
                grb_synth_data[f'{grb_type}_angular_spectrum'],
                nthetas=NTHETAS
            )
        )

    np.savez(DATA_DIR / 'synthetic_grb_correlation_CF_gamma_fit.npz', **grb_gamma_fits)


if __name__ == "__main__":
    main()
