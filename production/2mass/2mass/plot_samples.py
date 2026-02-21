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
plt.show()

helperfunctions.plot_l_b(ra, dec)
plt.show()
