#!/usr/bin/env python3
import numpy as np
from scipy.stats import gamma 
from multiprocessing import Pool
import sys
sys.path.append('../published_results/')
from utils import DATA_DIR 


def _gamma_fit(synth_data_row):
    scale = 1e12  # Regularising values for better fitting.
    mean = np.mean(synth_data_row)
    var = np.var(synth_data_row)
    scale = mean / var
    loc = mean * scale
    fitted_params = gamma.fit(synth_data_row * scale, method='MLE')
    return gamma.mean(*fitted_params) / scale, gamma.std(*fitted_params) / scale

def mp_gamma_fit(synth_data):
    with Pool(processes=20) as pool:
        results = pool.map(_gamma_fit, synth_data)
    return np.array(results, dtype=[('mean', 'f8'), ('std', 'f8')])


def main():
    # Read synthetic data
    gw_synth_data = np.load(DATA_DIR / 'congregated_synthetic_GW_correlation_stats.npz')
    grb_synth_data = np.load(DATA_DIR / 'congregated_synthetic_grb_correlation_stats_n1000.npz')

    print('(Multi-)Processing gamma fit for synthetic GW correlations...')
    gw_fit_results = mp_gamma_fit(gw_synth_data['angular_spectrum'].T)
    np.save(DATA_DIR / 'synthetic_GW_correlation_stats_gamma_fit.npy', gw_fit_results)

    print('(Multi-)Processing gamma fit for synthetic GRB correlations...')
    print('  - Full GRB sample')
    full_grb_fit_results = mp_gamma_fit(grb_synth_data['full_angular_spectrum'].T)
    print('  - Short GRB sample')
    short_grb_fit_results = mp_gamma_fit(grb_synth_data['short_angular_spectrum'].T)
    print('  - Long GRB sample')
    long_grb_fit_results = mp_gamma_fit(grb_synth_data['long_angular_spectrum'].T)

    np.savez(DATA_DIR / 'synthetic_grb_correlation_stats_gamma_fit.npz', 
             full_grb_gamma_fit=full_grb_fit_results,
             short_grb_gamma_fit=short_grb_fit_results,
             long_grb_gamma_fit=long_grb_fit_results)


if __name__ == "__main__":
    main()
