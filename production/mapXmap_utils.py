# GRB x GW utilities, for cross-correlating GRB and GW maps (or any skymaps, really)

import numpy as np
from matplotlib import pyplot as plt
import healpy as hp
from scipy.special import legendre  # Legendre functions P_l(x)

# set universal nside value
nside = 512


# two point correlation function given C_l coefficients
def corr_func(theta, Cl):
    c = 0
    x = np.cos(theta)
    for ell in range(len(Cl)):
        c += (1 + 2 * ell) * Cl[ell] * legendre(ell)(x)
    return c / (4 * np.pi)


# make skymap from coordinates data (ra_deg, dec_deg)
def make_skymap(ra_deg, dec_deg, nside=nside):
    npix = hp.nside2npix(nside) # npix = 12*nside**2
    skymap = np.zeros(npix)

    lon_rad = np.radians(ra_deg)
    lat_rad = np.pi / 2 - np.radians(dec_deg)

    ipix = hp.ang2pix(nside, lat_rad, lon_rad, nest=False)

    for i in ipix:
        skymap[i] += 1.0  # Or your actual data value

    return skymap


# map to alm, with lmax option
def map2alm(skymap, lmax=None):
    if lmax is None:
        lmax = 3 * hp.npix2nside(len(skymap)) - 1
    alm = hp.map2alm(skymap, lmax=lmax)
    return alm


# alm to Cl
def alm2cl(alm, lmax=None):
    if lmax is not None:
        Cl = hp.alm2cl(alm, lmax=lmax, mmax=lmax)
    else:
        Cl = hp.alm2cl(alm)
    return Cl


PROJ_KWS = {
    'graticule': True,
    'projection_type': 'mollweide',
}


# skymap plotting function
def plot_skymap(skymap, title='', min=None, max=None, unit='', norm=None):
    if min is None:
        min = np.min(skymap[skymap>0])
    if max is None:
        max = np.max(skymap)

    return hp.projview(
        skymap,
        title=title, graticule_labels=True,
        norm=norm, min=min, max=max,
        unit=unit, **PROJ_KWS)


# special skymap plotting function to match GW skymap style in literature
def plot_skymap_special(skymap, title='', min=None, max=None, unit='', norm=None):
    if min is None:
        min = np.min(skymap[skymap>0])
    if max is None:
        max = np.max(skymap)

    return hp.projview(
        skymap,
        title=title, graticule_color='darkgray',
        longitude_grid_spacing=30,
        latitude_grid_spacing=30,
        norm=norm, min=min, max=max,
        unit=unit, **PROJ_KWS)


# normalize skymap
def normalize_skymap(skymap, shift_min=False):
    if shift_min is True:
        skymap = skymap - skymap.min()

    return skymap / skymap.sum()


# normalize skymap, square integral
def normalize_skymap_sqrint(skymap, shift_min=False):
    if shift_min is True:
        skymap = skymap - skymap.min()

    sqr_map = [n**2 for n in skymap]
    sqr_map = np.array(sqr_map)
    sqr_intgrl = sqr_map.sum()

    return skymap / sqr_intgrl


# map blurring to lmax
def blur_map(skymap, lmax, remove_monopole=False, nside=nside, tol=1e-10):
    alm = hp.map2alm_lsq(skymap, lmax=lmax, mmax=lmax, tol=tol)
    alm = alm[0]
    if remove_monopole is True:
        alm[0] = 0.0

    blurred_map = hp.alm2map(alm, nside=nside)
    return blurred_map


# get blurred map alm
def blur_mapANDalm(skymap, lmax, remove_monopole=False, nside=nside, tol=1e-10):
    alm = hp.map2alm_lsq(skymap, lmax=lmax, mmax=lmax, tol=tol)
    alm = alm[0]
    if remove_monopole is True:
        alm[0] = 0.0

    blurred_map = hp.alm2map(alm, nside=nside)
    return blurred_map, alm


# get cross-map angular power spectrum Cl^cross
def cross_cl(map1, map2, lmax=None):
    if lmax is None:
        lmax = 3 * hp.npix2nside(len(map1)) - 1

    Cl_cross = hp.anafast(map1, map2=map2, lmax=lmax)

    return Cl_cross


# check if array contains complex values
def contains_complex_value(arr):
    for item in arr:
        if isinstance(item, complex):
            return True
    return False


# get depth of nested list
def list_depth(lst):
    if not isinstance(lst, list) or not lst:
        return 0

    max_depth = 0
    for item in lst:
        if isinstance(item, list):
            max_depth = max(max_depth, get_list_depth(item))

    return 1 + max_depth


# plot Cl function; if more than one, pass lists for Cls, Cl_labels, and colors
def plot_Cls(Cls, Cl_labels=None, lmax=None, monopole_term=True,
             colors=None, linestyles=None, title=None,
             ylabel=None, ylog=True, xlog=False,
             save=False, save_ext=".png"):
    if lmax is None:
        if list_depth(Cls) >= 1:
            lmax = len(Cls[0])
        else:
            lmax = len(Cls)

    l_vals = np.arange(lmax)
    start_idx = 0

    if monopole_term is False:
        start_idx = 1
        l_vals = l_vals[1:]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
    if list_depth(Cls) >= 1:
        if Cl_labels is None:
            Cl_labels = [None]*len(Cls)
        if colors is None:
            colors = [None]*len(Cls)
        if linestyles is None:
            linestyles = [None]*len(Cls)
        for Cl, label, color, ls in zip(Cls, Cl_labels, colors, linestyles):
            Cl = Cl[start_idx:lmax]
            ax.plot(l_vals, Cl, label=label, color=color, linestyle=ls)
    else:
        Cls = Cls[start_idx:lmax]
        ax.plot(l_vals, Cls, label=Cl_labels, color=colors, linestyle=linestyles)
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_xticks(l_vals, minor=True)

    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')

    ax.set_ylabel(ylabel if ylabel is not None else r'$C_\ell$')
    ax.set_title(title if title is not None else r'Angular Power Spectrum $C_\ell$')
    ax.legend()
    ax.grid(True)

    if save:
        filename = f'{title}{save_ext}'.replace(" ", "")
        fig.savefig(filename, dpi=600)

    plt.show()
    return fig


# plot Dl function; if more than one, pass lists for Cls, Cl_labels, and colors
def plot_Dls(Cls, Cl_labels=None, lmax=None,
             colors=None, linestyles=None, title=None,
             ylabel=None, ylog=True, xlog=False,
             save=False, save_ext='.png'):
    if lmax is None:
        if list_depth(Cls) >= 1:
            lmax = len(Cls[0])
        else:
            lmax = len(Cls)

    l_vals = np.arange(lmax)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
    if list_depth(Cls) >= 1:
        if Cl_labels is None:
            Cl_labels = [None]*len(Cls)
        if colors is None:
            colors = [None]*len(Cls)
        if linestyles is None:
            linestyles = [None]*len(Cls)
        for Cl, label, color, ls in zip(Cls, Cl_labels, colors, linestyles):
            Cl = Cl[:lmax]
            ax.plot(l_vals, Cl*l_vals*(l_vals+1)/(2*np.pi),
                    label=label, color=color, linestyle=ls)
    else:
        Cls = Cls[:lmax]
        ax.plot(l_vals, Cls*l_vals*(l_vals+1)/(2*np.pi),
                label=Cl_labels, color=colors, linestyle=linestyles)

    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')

    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_xticks(l_vals, minor=True)
    ax.set_ylabel(ylabel if ylabel is not None else r'$D_\ell = C_\ell \ell (\ell + 1)/ 2\pi$')
    ax.set_title(title if title is not None else r'Angular Power Spectrum $D_\ell$')
    ax.legend()
    ax.grid(True)

    if save:
        filename = f'{title}{save_ext}'.replace(" ", "")
        fig.savefig(filename, dpi=600)

    plt.show()
    return fig


# plot angular correlation function C(theta)
def plot_corr_func(Cls, Cl_labels=None,
                   colors=None, linestyles=None, title=None,
                   ylabel=None, ylog=False, xlog=False,
                   save=False, save_ext='.png'):
    theta_vals = np.linspace(0, 180, 180)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
    if list_depth(Cls) >= 1:
        if Cl_labels is None:
            Cl_labels = [None] * len(Cls)
        if colors is None:
            colors = [None] * len(Cls)
        if linestyles is None:
            linestyles = [None] * len(Cls)
        for Cl, label, color, ls in zip(Cls, Cl_labels, colors, linestyles):
            ax.plot(theta_vals, corr_func(np.radians(theta_vals), Cl),
                    label=label, color=color, linestyle=ls)
    else:
        ax.plot(theta_vals, corr_func(np.radians(theta_vals), Cls),
                label=Cl_labels, color=colors, linestyle=linestyles)

    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')

    ax.set_xlabel(r'$\theta\,/\,\textrm{deg}$')
    ax.set_ylabel(ylabel if ylabel is not None else r'Correlation, $C(\theta)$')
    ax.set_title(title if title is not None else r'Angular Correlation Function $C(\theta)$')
    ax.legend()
    ax.grid(True)

    if save:
        filename = f'{title}{save_ext}'.replace(" ", "")
        fig.savefig(filename, dpi=600)

    plt.show()
    return fig


# save skymap to .npy file
def save_skymap_npy(skymap, filename=None):
    if filename is None:
        filename = str(skymap) + '.npy'
    plt.savefig(
        filename, dpi=600, bbox_inches="tight", facecolor="white")


# save skymap to .npy file
def save_plot_local(plot, filename=None):
    if filename is None:
        filename = str(plot) + '.png'
    plt.savefig(
        filename, dpi=600, bbox_inches="tight", facecolor="white")


##
