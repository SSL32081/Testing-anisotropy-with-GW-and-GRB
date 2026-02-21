import scienceplots
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
# Plotting
from astropy.visualization import wcsaxes
from astropy.wcs import WCS
# For converting ra, dec to l, b
from astropy.coordinates import SkyCoord
from astropy import units as u
import ligo.skymap.plot
plt.style.use(["science", "ieee", "bright"])

def convert_ra_dec_to_l_b(ra, dec):
    c = SkyCoord(ra=ra*u.rad, dec=dec*u.rad, frame='icrs')
    l = c.galactic.l.rad
    b = c.galactic.b.rad
    return l, b

def plot_ra_dec(ra, dec):
    ''' Plot RA and Dec in a Mollweide projection. RA is wrapped to [-pi, pi] for better visualization.
    '''
    fig  = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='mollweide')
    # Wrap ra and dec for plotting
    ra_plot, dec_plot = ra, dec
    ra_plot = np.remainder(ra_plot + 2*np.pi, 2*np.pi) - np.pi  # Wrap to [-pi, pi]
    ra_plot = -ra_plot  # Invert RA for correct orientation
    ax.scatter(ra_plot, dec_plot, alpha=0.5, s=1)
    ax.set_xlabel(r"RA")
    ax.set_ylabel(r"Dec")
    ax.grid(True)
    return fig, ax

def plot_l_b(l, b):
    ''' Plot l,b in the galactic coordinate system (l, b) in a Mollweide projection. l is wrapped to [-pi, pi] for better visualization.
    '''
    fig  = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='mollweide')
    # Wrap l and b for plotting
    l_plot, b_plot = l, b
    l_plot = np.remainder(l_plot + 2*np.pi, 2*np.pi) - np.pi  # Wrap to [-pi, pi]
    l_plot = -l_plot  # Invert l for correct orientation
    ax.scatter(l_plot, b_plot, alpha=0.5, s=1)
    ax.set_xlabel(r"l")
    ax.set_ylabel(r"b")
    ax.grid(True)
    return fig, ax

def mask_zone_of_avoidance(l, b, zone_of_avoidance=10):
    ''' Mask out points within zone_of_avoidance degrees of the galactic plane (|b| < zone_of_avoidance).
    '''
    mask = np.abs(b) >= np.radians(zone_of_avoidance)
    return l[mask], b[mask]

def generate_uniform_sphere_2mass(n_points, zone_of_avoidance=10):
    ''' Generate n_points uniformly distributed in galactic plane coordinate l,b, but with a zone of avoidance around the galactic plane (|b| < zone_of_avoidance degrees).
    '''
    n_points_extra = int(n_points * 2)  # Generate more points to ensure we get enough after filtering
    l = np.random.uniform(0, 2*np.pi, n_points_extra)
    b = np.arcsin(np.random.uniform(-1, 1, n_points_extra))
    l, b = mask_zone_of_avoidance(l, b, zone_of_avoidance=zone_of_avoidance)
    # Pick the first n_points after filtering
    l = l[:n_points]
    b = b[:n_points]
    return l, b

def get_angular_power_spectrum(l, b, nside=128):
    ''' Compute the angular power spectrum Cl from galactic coordinates l and b.
    '''
    # 1. Convert Galactic coordinates to colatitude (theta) and longitude (phi)
    # healpy uses theta [0, pi] and phi [0, 2pi]
    theta = np.pi/2 - b  # Convert b to colatitude
    phi = l  # l is already in the correct range for phi
    
    # 2. Project points onto a HEALPix map
    npix = hp.nside2npix(nside)
    pixel_indices = hp.ang2pix(nside, theta, phi)
    
    # Create the map (count galaxies per pixel)
    hpx_map = np.bincount(pixel_indices, minlength=npix).astype(float)
    
    # Optional: Convert to overdensity delta = (n - <n>) / <n>
    mean_n = np.mean(hpx_map)
    hpx_map = (hpx_map - mean_n) / mean_n
    
    # 3. Compute the angular power spectrum Cl
    cl = hp.anafast(hpx_map)
    ell = np.arange(len(cl))
    return ell, cl

def get_angular_power_spectra(l_all, b_all, nside=128):
    ''' Compute the angular power spectra for multiple realizations of l and b.
    l_all and b_all should be arrays of shape (n_realisations, n_galaxies).
    '''
    n_realisations = l_all.shape[0]
    cl_all = []
    for i in range(n_realisations):
        ell, cl = get_angular_power_spectrum(l_all[i], b_all[i], nside=nside)
        cl_all.append(cl)
    return np.array(ell), np.array(cl_all)

def plot_angular_power_spectrum(ell, cl, label=None, ax=None):
    ''' Plot the angular power spectrum Cl as a function of multipole moment ell.
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    #ax.plot(ell, ell * (ell + 1) * cl, label=label)
    ax.plot(ell, cl, label=label)
    ax.set_xlabel(r"$\ell$")
    #ax.set_ylabel(r"$\ell(\ell+1)C_\ell$")
    ax.set_ylabel(r"$C_\ell$")
    #ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True)
    fig = ax.get_figure()
    return fig, ax

def plot_angular_power_spectra(ell, cl_all, labels=None, ax=None):
    ''' Plot median, 1-sigma, 3-sigma bands for multiple angular power spectra.
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    cl_median = np.median(cl_all, axis=0)
    cl_1sigma = np.percentile(cl_all, [16, 84], axis=0)
    cl_3sigma = np.percentile(cl_all, [2.5, 97.5], axis=0)
    y = ell * (ell + 1) * cl_all
    y_median = np.median(y, axis=0)
    y_1sigma = np.percentile(y, [16, 84], axis=0)
    y_3sigma = np.percentile(y, [2.5, 97.5], axis=0)
    #ax.plot(ell, y_median, label=labels[0] if labels else "Median")
    #ax.fill_between(ell, y_1sigma[0], y_1sigma[1], color="gray", alpha=0.5, label=labels[1] if labels else "1-sigma")
    #ax.fill_between(ell, y_3sigma[0], y_3sigma[1], color="lightgray", alpha=0.5, label=labels[2] if labels else "3-sigma")
    ax.plot(ell, cl_median, label=labels[0] if labels else "Median")
    ax.fill_between(ell, cl_1sigma[0], cl_1sigma[1], color="gray", alpha=0.5, label=labels[1] if labels else "1-sigma")
    ax.fill_between(ell, cl_3sigma[0], cl_3sigma[1], color="lightgray", alpha=0.5, label=labels[2] if labels else "3-sigma")
    ax.set_xlabel(r"$\ell$")
    #ax.set_ylabel(r"$\ell(\ell+1)C_\ell$")
    ax.set_ylabel(r"$C_\ell$")
    #ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True)
    if labels:
        ax.legend()
    fig = ax.get_figure()
    return fig, ax
