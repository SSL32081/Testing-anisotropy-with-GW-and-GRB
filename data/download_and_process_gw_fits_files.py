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
import requests
import shutil
import tarfile
import numpy as np
import healpy as hp
from ligo.skymap.io import read_sky_map
import multiprocessing as mp


def https_download(url, dest_path: Path):
    with requests.get(url, stream=True) as req:
        req.raise_for_status()  # Raise an exception for bad status codes
        with open(dest_path, 'wb') as f:
            # Copy the response content to the file object in chunks
            shutil.copyfileobj(req.raw, f)
    print(f"Successfully Downloaded to: {dest_path}")


def rotate_skymap_to_galactic(skymap, header, save=False, output_path=None):
    nside = hp.get_nside(skymap)
    print('Get', nside, 'and size: ', skymap.size, 'for: ', output_path.name)
    ordering = 'nested' if header.get('ORDERING') == 'NESTED' else 'ring'
    # If not in RING ordering, convert to RING for rotation
    if ordering == 'nested':
        skymap = hp.reorder(skymap, n2r=True)  # NESTED to RING

    # Convert to spherical harmonics
    lmax = 3 * nside - 1  # Standard choice
    alm = hp.map2alm(skymap, lmax=lmax)

    # Create rotator and rotate alms
    r = hp.Rotator(coord=['C', 'G'])
    alm_rotated = r.rotate_alm(alm)

    # Convert back to map
    skymap_gal = hp.alm2map(alm_rotated, nside=nside)
    # Convert back to original
    if ordering == 'nested':
        skymap_gal = hp.reorder(skymap_gal, r2n=True)  # RING to NESTED
    # Normalize
    skymap_gal = skymap_gal / np.sum(skymap_gal)

    if save:
        # Save to new file
        hp.write_map(output_path, skymap_gal.astype(np.float64),
                     coord='G', nest=(ordering == 'nested'), overwrite=True)
    return skymap_gal


PARENT_DIR = Path(os.environ.get('HANDON_REPO', './')) / 'LVK_skyloc_samples'
# GWTC-4 skymap FITS files directory (replace as appropriate)
GWTC4_FITS_DIRNAME = 'GWTC4p0_skymaps'
GWTC4_FITS_DIR = PARENT_DIR / GWTC4_FITS_DIRNAME
# Synthetic O4a skymap FITS files directory (replace as appropriate)
SYN_O4A_FITS_DIR = PARENT_DIR / 'Synthetic_O4a_skymaps'

## Sources of the FITS files
### GWTC-4
GWTC4p0_Zenodo = "17014085"
skymaps_GWTC4p0 = f"https://zenodo.org/records/{GWTC4p0_Zenodo}/files/IGWN-GWTC4p0-1a206db3d_721-Archived_Skymaps.tar.gz"
GWTC4p0_tarfile = "GWTC4p0_skymaps.tar.gz"
### Synthetic O4a
synth_o4a_urls = "https://gw.phy.cuhk.edu.hk/static/O4a_simulated_skymaps/{tar_file}".format
synth_o4a_zenodo = "https://zenodo.org/records/{}/files/".format("")
synth_o4a_tarfile = "O4a_H1L1_synthetic_skymaps.tar.gz"

# 1. Check if the FITS files are available in the local directory
# if not, download them
## Check GWTC-4 skymaps
GWTC4_FITS_DIR.mkdir(parents=True, exist_ok=True)
if len(list(GWTC4_FITS_DIR.glob("*.fits.gz"))) < 370:
    # Only partial files exist, redownload
    print("Downloading GWTC-4 skymap FITS files...")
    https_download(skymaps_GWTC4p0, PARENT_DIR / GWTC4p0_tarfile)

    with tarfile.open(PARENT_DIR / GWTC4p0_tarfile, mode='r:gz') as tar:
        # Extract the files into the GWTC4_FITS_DIR
        tar.extractall(path=GWTC4_FITS_DIR)
        # Relocate the files to the desired directory
        for fits_file in (GWTC4_FITS_DIR / "parameter_estimation/skymaps").glob("*.fits.gz"):
            fits_file.rename(GWTC4_FITS_DIR / fits_file.name)
    # Remove the tar.gz file and extracted folders
    (PARENT_DIR / GWTC4p0_tarfile).unlink()
    shutil.rmtree(GWTC4_FITS_DIR / 'parameter_estimation')
else:
    print("GWTC-4 skymap FITS files already exist. Skipping download.")

## Synthetic O4a skymaps
USE_ZENODO = False
SYN_O4A_FITS_DIR.mkdir(parents=True, exist_ok=True)
if len(list(SYN_O4A_FITS_DIR.glob("*.fits.gz"))) < 1000:
    # Only partial files exist, redownload
    if USE_ZENODO:
        ### A. Zenodo record
        print(f"Downloading Synthetic O4a skymap FITS files: {synth_o4a_tarfile}...")
        https_download(synth_o4a_zenodo, PARENT_DIR / synth_o4a_tarfile)

        with tarfile.open(PARENT_DIR / synth_o4a_tarfile, mode='r:gz') as tar:
            # Extract the files into the SYN_O4A_FITS_DIR
            tar.extractall(path=SYN_O4A_FITS_DIR)
        # Remove the tar.gz file
        (PARENT_DIR / synth_o4a_tarfile).unlink()
    else:
        ### B. Old files: 10 sets
        for start_idx in range(0, 1000, 100):
            end_idx = start_idx + 99
            tar_file = f"H1L1_sets_{start_idx:d}_{end_idx:d}.tar.gz"

            print(f"Downloading Synthetic O4a skymap FITS files: {tar_file}...")
            https_download(synth_o4a_urls(tar_file=tar_file), PARENT_DIR / tar_file)

            with tarfile.open(PARENT_DIR / tar_file, mode='r:gz') as tar:
                # Extract the files into the SYN_O4A_FITS_DIR
                tar.extractall(path=SYN_O4A_FITS_DIR)
            for fits_file in (SYN_O4A_FITS_DIR / f"H1L1/sets_{start_idx:d}_{end_idx:d}").glob("*.fits.gz"):
                fits_file.rename(SYN_O4A_FITS_DIR / fits_file.name)
            # Remove the tar.gz file
            (PARENT_DIR / tar_file).unlink()
        shutil.rmtree(SYN_O4A_FITS_DIR / 'H1L1')
else:
    print("Synthetic O4a skymap FITS files already exist. Skipping download.")

print('All skymap FITS files are ready!')


# 2. Process GWTC-4 skymap fits
def process_celestrial_fits_file(filepath):
    if 'galactic' in filepath.name:
        skymap_gal = hp.read_map(filepath, verbose=False)
    else:
        skymap, header = read_sky_map(filepath, distances=False, moc=False)
        filename = filepath.name.split('.')[0]
        output_path = filepath.with_name(filename + "_galactic.fits.gz")
        if output_path.exists():
            return None

        print('Processing:', filepath.name)
        skymap_gal = rotate_skymap_to_galactic(
            skymap, header, save=True, output_path=output_path)

    # Resample to common resolution
    # with power=-2, it keeps the sum of the map invariant)
    skymap_resized = hp.ud_grade(skymap_gal, NSIDE, power=-2)
    return skymap_resized / np.sum(skymap_resized)


NSIDE = 256
NPIX = hp.nside2npix(NSIDE)
with mp.Pool(processes=16) as pool:
    processed_maps = pool.map(
        process_celestrial_fits_file, GWTC4_FITS_DIR.glob("*Mixed_*fits.gz"))

resultant_map = np.zeros(NPIX)
for skymap in processed_maps:
    if skymap is not None:
        resultant_map += skymap

resultant_map = resultant_map / np.sum(resultant_map)  # Normalize combined map
np.save("./GWTC4p0_combined_galactic_skymap.npy", resultant_map)


# 3. Process synthetic O4a skymap fits
with mp.Pool(processes=16) as pool:
    processed_maps = pool.map(
        process_celestrial_fits_file, SYN_O4A_FITS_DIR.glob("H1L1*fits.gz"))

resultant_map = np.zeros(NPIX)
for skymap in processed_maps:
    if skymap is not None:
        resultant_map += skymap

resultant_map = resultant_map / np.sum(resultant_map)  # Normalize combined map
np.save("./synthetic_O4a_combined_galactic_skymap.npy", resultant_map)
