import scienceplots
import matplotlib.pyplot as plt
import numpy as np
# For converting ra, dec to l, b
from astropy.coordinates import SkyCoord
from astropy import units as u
plt.style.use(["science", "ieee", "bright"])

def convert_ra_dec_to_l_b(ra, dec):
    c = SkyCoord(ra=ra*u.rad, dec=dec*u.rad, frame='icrs')
    l = c.galactic.l.rad
    b = c.galactic.b.rad
    return l, b

def plot_ra_dec(ra, dec):
    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(111, projection="mollweide")
    ax.scatter(ra, dec, s=1, alpha=0.5)
    ax.set_xlabel(r"ra")
    ax.set_ylabel(r"dec")
    plt.grid(True)
    return fig, ax

def plot_l_b(ra, dec):
    l, b = convert_ra_dec_to_l_b(ra, dec)
    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(111, projection="mollweide")
    ax.scatter(l, b, s=1, alpha=0.5)
    ax.set_xlabel(r"l")
    ax.set_ylabel(r"b")
    plt.grid(True)
    return fig, ax

