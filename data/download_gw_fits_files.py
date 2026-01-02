#!/usr/bin/env python3
import os
from pathlib import Path
import requests
import shutil
import tarfile

def https_download(url, dest_path: Path):
    with requests.get(url, stream=True) as req:
        req.raise_for_status()  # Raise an exception for bad status codes
        with open(dest_path, 'wb') as f:
            # Copy the response content to the file object in chunks
            shutil.copyfileobj(req.raw, f)
    print(f"Successfully Downloaded to: {dest_path}")


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
synth_o4a_zenodo = "https://zenodo.org/records/{}/files/".format()
synth_o4a_tarfile = "O4a_H1L1_synthetic_skymaps.tar.gz"

# 1. Check if the FITS files are available in the local directory
# if not, download them
## Check GWTC-4 skymaps
GWTC4_FITS_DIR.mkdir(parents=True, exist_ok=True)
if len(list(GWTC4_FITS_DIR.glob("*.fits.gz"))) != 370:
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

## Synthetic O4a skymaps
USE_ZENODO = False
SYN_O4A_FITS_DIR.mkdir(parents=True, exist_ok=True)
if len(list(SYN_O4A_FITS_DIR.glob("*.fits.gz"))) != 1000:
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
        for start_idx in range(0, 1001, 100):
            end_idx = start_idx + 99
            tar_file = f"H1L1_sets_{start_idx:d}_{end_idx:d}.tar.gz"

            print(f"Downloading Synthetic O4a skymap FITS files: {tar_file}...")
            https_download(synth_o4a_urls(tar_file=tar_file), PARENT_DIR / tar_file)

            with tarfile.open(PARENT_DIR / tar_file, mode='r:gz') as tar:
                # Extract the files into the SYN_O4A_FITS_DIR
                tar.extractall(path=SYN_O4A_FITS_DIR)
            # Remove the tar.gz file
            (PARENT_DIR / tar_file).unlink()


# 2. 