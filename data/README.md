# Data Sources

## Key data files
* `GWTC4p0_combined_galactic_skymap.npy`
  * Figure 1, 3, 4, 5, 6, 7
  * Produced from `download_and_process_gw_fits_files.py`
* `GRB_Summary_table.txt`
  * Figure 1, 4, 5, 6, 7
* `GLADE_galactic_coords.npy`
  * Figure 1, 6
  * Produced from `process_glade_plus_data.py`
* `congregated_synthetic_gw_snr8_correlation_stats_2278_85_lmax128_n1000.npy`
  * Figure 2
  * Generated from `compute_synthetic_gw_skymaps_correlations.py`
* `simulated_grbs/simulated_grbs_realisation_n1000_0.npz`
  * Figure 2
  * Generated from `generate_synthetic_grb_data.py`
* `synthetic_gw_snr8_correlation_CLCF_gamma_fit_n180_lmax128_windowed.npz`
  * Figure 5, 6
  * Generated from `compute_synthetic_data_gamma_fit.py`
* `synthetic_grb_correlation_CLCF_gamma_fit_n180_lmax128_windowed.npz`
  * Figure 5, 6
  * Generated from `compute_synthetic_data_gamma_fit.py`

## Others
* `GLADE_plus_subset.txt`
  * Intermediate file, source of `GLADE_galactic_coords.npy`
  * Downloaded by `download_glade_plus.sh`

* `congregated_synthetic_grb_correlation_stats_1000_lmax128_n1000.npy`
  * Intermediate file, source of `synthetic_*_correlation_CLCF_gamma_fit_n180_lmax128_windowed.npy`
  * Generated from `generate_synthetic_grb_data.py`

<!--
## Data sources

* Observed GRB
  * `Summary_table.txt` / `GRB_Summary_table.txt`
* Observed Galaxy
  * `GLADE_galactic_coords.txt`
* Synthetic GW skymap
  * `https://gw.phy.cuhk.edu.hk/static/O4a_simulated_skymaps/`
  * `wget + tar`
* Synthetic GRB 
  * `simulated_grbs.txt`

-->