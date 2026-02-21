import scienceplots
import matplotlib.pyplot as plt
import numpy as np
import os
import helperfunctions
plt.style.use(["science", "ieee", "bright"])

filename = "2mass_galaxy_catalog_spec.csv"

# Read in the file, including the names of the variables (defined in the header)
data = np.genfromtxt(filename, delimiter=",", names=True)
# Convert to radians
ra = np.deg2rad(data["RAJ2000"])
dec = np.deg2rad(data["DEJ2000"])
l, b = helperfunctions.convert_ra_dec_to_l_b(ra, dec)
n_galaxies = len(ra)

# Now generate 1,000 realisations of n_galaxies synthetic galaxies distributed across the sky, with masking
n_realisations = 1000
# If the file already exists, we can load it instead of regenerating it
if os.path.exists("synthetic_2mass_catalog.npz"):
    data = np.load("synthetic_2mass_catalog.npz")
    l_synthetic_all = data["l_synthetic_all"]
    b_synthetic_all = data["b_synthetic_all"]
else:
    l_synthetic_all = np.zeros((n_realisations, n_galaxies))
    b_synthetic_all = np.zeros((n_realisations, n_galaxies))
    for i in range(n_realisations):
        l_synthetic, b_synthetic = helperfunctions.generate_uniform_sphere_2mass(n_galaxies)
        l_synthetic_all[i] = l_synthetic
        b_synthetic_all[i] = b_synthetic
    np.savez("synthetic_2mass_catalog.npz", l_synthetic_all=l_synthetic_all, b_synthetic_all=b_synthetic_all) # Save

# Plot one realisation of the synthetic galaxy
l_synthetic = l_synthetic_all[0]
b_synthetic = b_synthetic_all[0]
helperfunctions.plot_l_b(l_synthetic, b_synthetic)
plt.title("Example synthetic 2mass data")
plt.savefig("example_synthetic_2mass_data.pdf", bbox_inches="tight")
plt.close()

# Plot the actual galaxies:
helperfunctions.plot_l_b(l, b)
plt.title("Actual 2mass data")
plt.savefig("actual_2mass_data.pdf", bbox_inches="tight")
plt.close()

# Get spectrum for real data
nside = 128
ell_real, cl_real = helperfunctions.get_angular_power_spectrum(l, b, nside=nside)
# Get spectrum for one synthetic realization
ell_synthetic, cl_synthetic = helperfunctions.get_angular_power_spectrum(l_synthetic, b_synthetic, nside=nside)

# 4. Plot the results
fig, ax = helperfunctions.plot_angular_power_spectrum(ell_real, cl_real, ax=None, label="Actual 2MASS (Clustered)")
_, _ = helperfunctions.plot_angular_power_spectrum(ell_synthetic, cl_synthetic, ax=ax, label="Synthetic (Uniform/Poisson)")
plt.legend()
plt.show()

# Now get the spectra for all 1,000 realisations of the synthetic data, and plot them together
if os.path.exists("synthetic_2mass_spectra.npz"):
    data = np.load("synthetic_2mass_spectra.npz")
    ell_synthetic = data["ell_synthetic"]
    cl_all_synthetic = data["cl_all_synthetic"]
else:
    ell_synthetic, cl_all_synthetic = helperfunctions.get_angular_power_spectra(l_synthetic_all, b_synthetic_all, nside=nside)
    # Save:
    np.savez("synthetic_2mass_spectra.npz", ell_synthetic=ell_synthetic, cl_all_synthetic=cl_all_synthetic)

fig, ax = helperfunctions.plot_angular_power_spectrum(ell_real, cl_real, ax=None, label="Actual 2MASS (Clustered)")
fig, ax = helperfunctions.plot_angular_power_spectra(ell_synthetic, cl_all_synthetic, ax=ax)
plt.legend()
plt.show()


