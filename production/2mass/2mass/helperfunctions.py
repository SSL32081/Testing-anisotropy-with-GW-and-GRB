import scienceplots
import matplotlib.pyplot as plt
import numpy as np
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

def plot_l_b(ra, dec):
    ''' Plot ra, dec in the galactic coordinate system (l, b) in a Mollweide projection. l is wrapped to [-pi, pi] for better visualization.
    '''
    l, b = convert_ra_dec_to_l_b(ra, dec)
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

