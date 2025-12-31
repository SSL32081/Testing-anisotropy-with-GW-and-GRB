# Testing-anisotropy-with-GW-and-GRB
The repository for storing production scripts and data-products for the paper.

## Structure of this Repository

Let's keep this place tidy!

* `production` 
  * This is for any scripts/notebooks that produces the final results in the paper
* `tests`
  * As the name suggests, this is for any notebooks that are for testing purposes.
* `figures` 
  * Any figures that will appear in the paper should go here.
* `data`
  * Small datasets (likely from processed data) that are generated can be stored here.

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
