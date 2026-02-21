import scienceplots
import matplotlib.pyplot as plt
import numpy as np
import helperfunctions
plt.style.use(["science", "ieee", "bright"])

filename = "2mass_galaxy_catalog_spec.csv"

# Read in the file, including the names of the variables (defined in the header)
data = np.genfromtxt(filename, delimiter=",", names=True)
# Convert to radians
ra = np.deg2rad(data["RAJ2000"])
dec = np.deg2rad(data["DEJ2000"])

helperfunctions.plot_ra_dec(ra, dec)
plt.title("2MASS Galaxy Catalog", fontsize=14)
plt.savefig("2mass_galaxy_catalog_spec_ra_dec.pdf", bbox_inches="tight")
plt.close()

helperfunctions.plot_l_b(ra, dec)
plt.title("2MASS Galaxy Catalog", fontsize=14)
plt.savefig("2mass_galaxy_catalog_spec_l_b.pdf", bbox_inches="tight")
plt.close()
