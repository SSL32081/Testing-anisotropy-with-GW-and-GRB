#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
DATA_DIR = Path("../data/")

print('Reading GLADE+ data...')
keys = ('ra', 'dec', 'd_L', 'z_cmb')
dtypes = [(key, 'f8') for key in keys]
glade_arr = np.genfromtxt(DATA_DIR / 'GLADE_plus_subset.txt', dtype=dtypes,
              missing_values=('None', 'null'), filling_values=np.nan)

# Pre-process data, remove NaNs
mask = ~(
    np.isnan(glade_arr['ra']) |
    np.isnan(glade_arr['dec']) |
    np.isnan(glade_arr['z_cmb'])
)
masked_data = glade_arr[mask]

print('Converting to galactic coordinates...')
coords = SkyCoord(
    ra=masked_data['ra'] * u.deg, dec=masked_data['dec'] * u.deg, frame="icrs")
l_gal = coords.galactic.l.wrap_at(180 * u.deg).radian
b_gal = coords.galactic.b.radian

l_gal_wrapped = np.remainder(l_gal + np.pi, 2 * np.pi) - np.pi
l_gal_wrapped = -l_gal_wrapped

output_arr = np.zeros_like(
    masked_data, dtype=[('l_gal', 'f8'), ('b_gal', 'f8'), ('z_cmb', 'f8')])
output_arr['l_gal'] = l_gal_wrapped
output_arr['b_gal'] = b_gal
output_arr['z_cmb'] = masked_data['z_cmb']

np.save(DATA_DIR / 'GLADE_galactic_coords.npy', output_arr)