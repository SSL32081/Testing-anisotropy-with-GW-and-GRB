#!/usr/bin/env python3
'''
This script is composed of two major parts:
1. Download skymap FITS files from online sources
2. Process the FITS files to convert them to galactic coordinates

The first part is relatively quick.
With a decent network connection, it is typically done within 10 mins.

The second part, however, is slow.
By default, it will save the converted maps as intermedidate files
to allow resuming should the program stop.
A) For GWTC-4 skymaps, it will produce about 1.4 GB of fits files
B) For Synthetic skymaps, it will produce more than 31 GB of files,
    and it will take over a day to run with 16 processes in parallel.

For these reasons, it is highly recommended to simply use the final
congregated skymap npy files instead:
* GWTC4p0_combined_galactic_skymap.npy
* synthetic_O4a_combined_galactic_skymap.npy
'''

import os
from pathlib import Path
import numpy as np
import healpy as hp
from ligo.skymap.io import read_sky_map
import multiprocessing as mp


def rotate_skymap_to_galactic(skymap, save=False, output_path=None):
    nside = hp.get_nside(skymap)
    print('Get', nside, 'and size: ', skymap.size, 'for: ', output_path.name)
    # Convert to spherical harmonics
    lmax = 3 * nside - 1  # Standard choice
    alm = hp.map2alm(skymap, lmax=lmax)
    # Create rotator and rotate alms
    r = hp.Rotator(coord=['C', 'G'])
    alm_rotated = r.rotate_alm(alm)
    # Convert back to map
    skymap_gal = hp.alm2map(alm_rotated, nside=nside)
    # Normalize
    skymap_gal = skymap_gal / np.sum(skymap_gal)

    if save:
        # Save to new file
        hp.write_map(output_path, skymap_gal.astype(np.float64),
                     coord='G', nest=False, overwrite=True)
    return skymap_gal


PARENT_DIR = Path(os.environ.get('HANDON_REPO', './')) / 'LVK_skyloc_samples'
# Synthetic O4a skymap FITS files directory (replace as appropriate)
# SYN_O4A_FITS_DIR = PARENT_DIR / 'Synthetic_O4a_skymaps'
# SYN_O4A_FITS_DIR = PARENT_DIR / 'Synthetic_O4a_SNR8_skymaps'
SYN_O4A_FITS_DIR = PARENT_DIR / 'Synthetic_O4a_SNR8_SFR_skymaps'


# 2. Process skymap fits
def process_celestrial_fits_file(filepath):
    if 'galactic' in filepath.name:
        # We want to make sure the map to be in ring-ordering
        skymap_gal = hp.read_map(filepath, nest=False)
    else:
        filename = filepath.name.split('.')[0]
        output_path = filepath.with_name(filename + "_galactic.fits.gz")
        if output_path.exists():
            return None

        print('Processing:', filepath.name)
        # Ensure all maps are in RING ordering
        skymap, _ = read_sky_map(filepath, nest=False, distances=False, moc=False)
        skymap_gal = rotate_skymap_to_galactic(
            skymap, save=True, output_path=output_path)

    return skymap_gal


NSIDE = 256
NPIX = hp.nside2npix(NSIDE)

# Process synthetic O4a skymap fits
with mp.Pool(processes=100) as pool:
    processed_maps = pool.map(
        process_celestrial_fits_file, SYN_O4A_FITS_DIR.glob("H1L1*fits.gz"))

print(len(processed_maps))
