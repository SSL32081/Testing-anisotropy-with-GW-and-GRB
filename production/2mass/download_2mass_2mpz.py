from astroquery.vizier import Vizier
import astropy.units as u

# 1. Configure Vizier to return the full catalog (approx 1 million rows)
v = Vizier(columns=['*', '_OT'], row_limit=-1)

# 2. Query the 2MPZ catalog specifically
# VII/275 is the 2MPZ: 2MASS Photometric Redshift Catalogue
catalogs = v.get_catalogs('VII/275')

# 3. Extract the primary table
twompz_table = catalogs[0]

## Masking (NOT WORKING ATM)
#mask = twompz_table['Ksmag'] < 13.9
#clean_sample = twompz_table[mask]

print(f"Downloaded {len(twompz_table)} galaxies from 2MPZ.")


# Save
filename = "2mass/2mpz_full.fits"
twompz_table.write("2mpz_full.fits", format="fits", overwrite=True)
print('Saved 2MPZ catalog to "2mpz_full.fits".')
