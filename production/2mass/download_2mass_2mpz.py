from astroquery.vizier import Vizier
import astropy.units as u

# 1. Configure Vizier to return the full catalog (approx 1 million rows)
v = Vizier(columns=['*', '_OT'], row_limit=-1)

# 2. Query the 2MPZ catalog specifically
# VII/275 is the 2MPZ: 2MASS Photometric Redshift Catalogue
catalogs = v.get_catalogs('VII/275')

# 3. Extract the primary table
twompz_table = catalogs[0]


print(f"Downloaded {len(clean_sample)} galaxies from 2MPZ.")

