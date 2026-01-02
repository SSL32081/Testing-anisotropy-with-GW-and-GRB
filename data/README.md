# Data Sources

* `simulated_grb.npy`
  * Generated from `generate_synthetic_grb_data.py`
* `simulated_grbs.txt` (OBSOLETE)
  * Generated from `tests/GRB_skymaps_synthetic/grb_skymap.ipynb`
  * Superseded by `simulated_grb.npy`.
* `GLADE_galactic_coords.npy`
  * Produced from `process_glade_plus_data.py`
* `GLADE_plus_subset.txt`
  * Intermediate file, source of `GLADE_galactic_coords.npy`
  * Downloaded by `download_glade_plus.sh`