import numpy as np
import pandas as pd
import healpy as hp
from pathlib import Path
from ligo.skymap.io.fits import read_sky_map

def get_max_sky_location(skymap: np.ndarray) -> tuple[float, float]:
    """
    get theta and phi of the maximum probability 
    pixel in skymap.
    """
    nside = hp.npix2nside(len(skymap))
    idx_max = np.argmax(skymap)
    theta, phi = hp.pix2ang(nside, idx_max) 
    
    return theta, phi


def load_data(csv_file, M_chirp, d_L_col) -> pd.DataFrame:
    try:
        data = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: file '{csv_file}' not found.")
        return pd.DataFrame()

    df = data[[d_L_col, M_chirp, "name"]].copy()
    df.rename(columns={M_chirp: 'M_value', d_L_col: 'd_L'}, inplace=True)
    
    return df


def get_location(event_name, base_path: Path, folders: list[str]) -> tuple[float, float]:

    skymap_files = []
    for folder in folders:
        if folder in ["GWTC2p1", "GWTC3p0"]:
            file_name = (f"{folder}_skymaps/IGWN-{folder}-v2-{event_name}_"
                "PEDataRelease_cosmo_reweight_C01:SEOBNRv4PHM.fits")
            skymap_files.append(base_path / file_name)
                
        elif folder == "GWTC4p0":
            file_name = (f"{folder}_skymaps/IGWN-GWTC4p0-1a206db3d_721-{event_name}-"
                "Mixed_Skymap_PEDataRelease.fits.gz")
            skymap_files.append(base_path / file_name)   

    found_file = next((f for f in skymap_files if f.exists()), None)
    theta, phi = np.nan, np.nan
    if found_file:
        try:
            skymap, _ = read_sky_map(str(found_file), moc=False)
            theta, phi = get_max_sky_location(skymap)
        except Exception as e:
            pass
    else: 
        print(f"No fits file: {event_name}")

    return theta, phi

def process_gw_data(
    csv_file: str, 
    base_path_str: str, 
    folders: list[str], 
    output_file: str = 'output.txt',
    M_chirp: str = "chirp_mass_source",
    d_L_col: str = "luminosity_distance"
    ) -> pd.DataFrame:
    
    df_clean = load_data(csv_file, M_chirp, d_L_col)
    base_path = Path(base_path_str)
    
    thetas, phis = [], []
    for event_name in df_clean['name']:
        theta, phi = get_location(event_name, base_path, folders)
        thetas.append(theta)
        phis.append(phi)

    df_clean['theta'] = thetas
    df_clean['phi'] = phis
    df_final = df_clean.copy() 

    # Write output file
    header = f'#event_name\t{M_chirp}[Msun]\t{d_L_col}[Mpc]\ttheta[rad]\tphi[rad]'
    columns_to_save = ['name', 'M_value', 'd_L', 'theta', 'phi']
    with open(output_file, 'w') as f:
        f.write(header + '\n')
        df_final[columns_to_save].to_csv(f, sep='\t', index=False, 
            header=False, float_format='%.6f', na_rep='nan')
    
    return df_final


if __name__ == "__main__":
    skymap_path = "/lfs/data/liting/GW_school/hands-on-school-2025/LVK_skyloc_samples/" 
    df_final = process_gw_data(
        csv_file='events.csv', # Download: https://gwosc.org/eventapi/html/GWTC/ 
        base_path_str=skymap_path, 
        folders=["GWTC2p1", "GWTC3p0", "GWTC4p0"], 
        output_file='gw_location_data.txt',
        M_chirp="chirp_mass_source", 
        d_L_col="luminosity_distance"
    )
