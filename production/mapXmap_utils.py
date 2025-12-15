# GRB x GW utilities, for cross-correlating GRB and GW maps (or any skymaps, really)

import numpy as np
from matplotlib import pyplot as plt
import healpy as hp
from scipy.special import legendre # Legendre functions P_l(x)

# set universal nside value
nside = 512


# two point correlation function given C_l coefficients
def corr_func(theta, Cl):
    c = 0
    for l in range(len(Cl)):
        c += (1 + 2*l)*Cl[l]*legendre(l)(np.cos(theta))/(4*np.pi)
    return c


# make skymap from coordinates data (ra_deg, dec_deg)
def make_skymap(ra_deg, dec_deg, nside=nside):
    npix = hp.nside2npix(nside) # npix = 12*nside**2
    skymap = np.zeros(npix)
    
    lon_rad = np.radians(ra_deg)
    lat_rad = np.pi/2 - np.radians(dec_deg)

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


# skymap plotting function
def plot_skymap(skymap, title='', min=None, max=None, unit='', norm=None):
    if min is None:
        min = np.min(skymap[skymap>0])
    if max is None:
        max = np.max(skymap)

    return hp.projview(skymap,
                graticule = True, graticule_labels = True,
                title=title, projection_type='mollweide', norm=norm,
                min=min, max=max,
                unit=unit)


# special skymap plotting function to match GW skymap style in literature
def plot_skymap_special(skymap, title='', min=None, max=None, unit='', norm=None):
    if min is None:
        min = np.min(skymap[skymap>0])
    if max is None:
        max = np.max(skymap)
    
    return hp.projview(skymap,
                graticule = True, graticule_color='darkgray',
                title=title, projection_type='mollweide', norm=norm,
                min=min, max=max,
                longitude_grid_spacing=30, latitude_grid_spacing=30,
                unit = unit)


# normalize skymap
def normalize_skymap(skymap, shift_min=False):
    if shift_min is True:
        skymap = skymap - skymap.min()
    
    skymap = skymap / skymap.sum()

    return skymap


# normalize skymap, square integral
def normalize_skymap_sqrint(skymap, shift_min=False):
    if shift_min is True:
        skymap = skymap - skymap.min()

    sqr_map = [n**2 for n in skymap]
    sqr_map = np.array(sqr_map)
    sqr_intgrl = sqr_map.sum()

    skymap = skymap / sqr_intgrl

    return skymap


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
def plot_Cls(Cls, Cl_labels=None, colors = None, linestyles=None, lmax=None, title=None, monopole_term=True,
             ylabel=None, ylog=True, xlog=False, save=False, save_ext=".png"):
    if lmax is None:
        if list_depth(Cls)>=1:
            lmax = len(Cls[0])
        else:
            lmax = len(Cls)
    
    l_vals = np.arange(0, lmax, 1)
    start_idx = 0

    if monopole_term is False:
        start_idx = 1
        l_vals = l_vals[1:]

    plt.figure(figsize=(8, 5))
    if list_depth(Cls)>=1:
        if Cl_labels is None:
            Cl_labels = [None]*len(Cls)
        if colors is None:
            colors = [None]*len(Cls)
        if linestyles is None:
            linestyles = [None]*len(Cls)
        for Cl, label, color, ls in zip(Cls, Cl_labels, colors, linestyles):
            Cl = Cl[start_idx:lmax]
            plt.plot(l_vals, Cl, label=label, color=color, linestyle=ls)
    else:
        Cls = Cls[start_idx:lmax]
        plt.plot(l_vals, Cls, label=Cl_labels, color=colors, linestyle=linestyles)
    plt.xlabel(r'Multipole $\ell$')
    plt.xticks(l_vals, minor=True)
    
    if xlog is True:
        plt.xscale('log')
    else:
        plt.xscale('linear')

    if ylabel is None:
        plt.ylabel(r'$C_\ell$')
    else:
        plt.ylabel(ylabel)

    if ylog is True:
        plt.yscale('log')
    else:
        plt.yscale('linear')
    
    if title is not None:
        plt.title(title)
    else:
        plt.title(r'Angular Power Spectrum $C_\ell$')
    plt.legend()
    plt.grid(True)

    if save is True:
        plt.savefig(str(title).replace(" ","") + save_ext, dpi=600, bbox_inches="tight")
    
    plt.show()


# plot Dl function; if more than one, pass lists for Cls, Cl_labels, and colors
def plot_Dls(Cls, Cl_labels=None, colors=None, linestyles=None, lmax=None, title=None,
             ylabel=None, ylog=True, xlog=False, save=False, save_ext='.png'):
    if lmax is None:
        if list_depth(Cls)>=1:
            lmax = len(Cls[0])
        else:
            lmax = len(Cls)
    
    l_vals = np.arange(lmax)

    plt.figure(figsize=(8, 5))
    if list_depth(Cls)>=1:
        if Cl_labels is None:
            Cl_labels = [None]*len(Cls)
        if colors is None:
            colors = [None]*len(Cls)
        if linestyles is None:
            linestyles = [None]*len(Cls)
        for Cl, label, color, ls in zip(Cls, Cl_labels, colors, linestyles):
            Cl = Cl[:lmax]
            plt.plot(l_vals, Cl*l_vals*(l_vals+1)/(2*np.pi), label=label, color=color, linestyle=ls)
    else:
        Cls = Cls[:lmax]
        plt.plot(l_vals, Cls*l_vals*(l_vals+1)/(2*np.pi), label=Cl_labels, color=colors, linestyle=linestyles)
    plt.xlabel(r'Multipole $\ell$')
    plt.xticks(l_vals, minor=True)

    if xlog is True:
        plt.xscale('log')
    else:
        plt.xscale('linear')

    if ylabel is None:
        plt.ylabel(r'$D_\ell = C_\ell \ell (\ell + 1)/ 2\pi$')
    else:
        plt.ylabel(ylabel)

    if ylog is True:
        plt.yscale('log')
    else:
        plt.yscale('linear')

    if title is not None:
        plt.title(title)
    else:
        plt.title(r'Angular Power Spectrum $D_\ell$')
    plt.legend()
    plt.grid(True)

    if save is True:
        plt.savefig(str(title).replace(" ","") + save_ext, dpi=600, bbox_inches="tight")

    plt.show()


# plot angular correlation function C(theta)
def plot_corr_func(Cls, Cl_labels=None, colors=None, linestyles=None, title=None, ylabel=None,
                   ylog=False, xlog=False, save=False, save_ext='.png'):
    theta_vals = np.linspace(0, 180, 180)

    plt.figure(figsize=(8, 5))
    if list_depth(Cls)>=1:
        if Cl_labels is None:
            Cl_labels = [None]*len(Cls)
        if colors is None:
            colors = [None]*len(Cls)
        if linestyles is None:
            linestyles = [None]*len(Cls)
        for Cl, label, color, ls in zip(Cls, Cl_labels, colors, linestyles):
            plt.plot(theta_vals, corr_func(np.radians(theta_vals), Cl),
                     label=label, color=color, linestyle=ls)
    else:
        plt.plot(theta_vals, corr_func(np.radians(theta_vals), Cls),
                 label=Cl_labels, color=colors, linestyle=linestyles)
    plt.xlabel(r'$\theta$ [deg]')
    
    if xlog is True:
        plt.xscale('log')
    else:
        plt.xscale('linear')
    
    if ylabel is None:
        plt.ylabel(r'Correlation, $C(\theta)$')
    else:
        plt.ylabel(ylabel)

    if ylog is True:
        plt.yscale('log')
    else:
        plt.yscale('linear')

    if title is not None:
        plt.title(title)
    else:
        plt.title(r'Angular Correlation Function $C(\theta)$')

    plt.legend()
    plt.grid(True)

    if save is True:
        plt.savefig(str(title).replace(" ","") + save_ext, dpi=600, bbox_inches="tight")

    plt.show()


# save skymap to .npy file
def save_skymap_npy(skymap, filename=None):
    if filename is None:
        filename = str(skymap) + '.npy'
    plt.savefig(filename,
                dpi=600, bbox_inches="tight", facecolor="white")
    
# save skymap to .npy file
def save_plot_local(plot, filename=None):
    if filename is None:
        filename = str(plot) + '.png'
    plt.savefig(filename,
                dpi=600, bbox_inches="tight", facecolor="white")


##