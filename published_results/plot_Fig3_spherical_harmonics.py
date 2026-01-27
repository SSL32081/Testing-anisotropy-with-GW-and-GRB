#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from utils import DATA_DIR, FIG_DIR, DOUBLE, DPI, \
    add_healpy_mollweide_ax
from copy import deepcopy
import healpy as hp
from matplotlib.colors import Normalize

LMAX = 40  # Maximum ℓ to compute
combined_map = np.load(DATA_DIR / 'GWTC4p0_combined_galactic_skymap.npy')
NSIDE = hp.get_nside(combined_map)

# Compute spherical harmonic coefficients
alm = hp.map2alm(combined_map, lmax=LMAX, iter=3)


def extract_ell_range(alm, ell_min, ell_max):
    """
    Reconstruct map using [ell_min, ell_max]
    """
    # Initialise the output
    alm_filtered = np.zeros_like(alm)
    # Insert only the relevant alm
    for ell in range(ell_min, ell_max + 1):
        for em in range(0, ell + 1):
            idx = hp.Alm.getidx(LMAX, ell, em)
            alm_filtered[idx] = alm[idx]
    # Reconstruct map
    map_reconstructed = hp.alm2map(alm_filtered, NSIDE, lmax=LMAX)
    return map_reconstructed, alm_filtered


all_max_maps = []
all_maps = []
for ell in range(1, 7):
    ell_map, _ = extract_ell_range(alm, ell, ell)
    all_maps.append(ell_map)
    max_map = np.max(np.abs(ell_map))
    all_max_maps.append(max_map)

fig, axes = plt.subplots(2, 3, figsize=(DOUBLE, 3.9),
                         constrained_layout=False)
# Healpy axis does not work well with constrained_layout

max_map = np.max(all_max_maps) * 0.98
for idx, ax in enumerate(axes.flatten()):
    ax = add_healpy_mollweide_ax(fig, ax)
    ax.projmap(
        all_maps[idx], nest=False,
        xsize=1300, coord='G',
        cmap='RdBu_r', badcolor='gray', bgcolor='white',
        vmin=-max_map, vmax=max_map
    )
    hp.graticule(dpar=30, dmer=30)
    ax.set_title(fr"$\ell={idx+1}$")

mappable = plt.cm.ScalarMappable(
    norm=Normalize(vmin=-max_map, vmax=max_map),
    cmap='RdBu_r',
)
fig.colorbar(mappable, ax=fig.axes,
             orientation="horizontal",
             location='bottom',
             label=r"$C_\ell^{\rm GW}(\chi,\ \phi)$",
             shrink=0.5, fraction=0.03)

# Need to redefine since the healpy axis replaced the old ones
axes = fig.axes
axes[0].set_title(axes[0].get_title() + ' (Dipole)')
axes[1].set_title(axes[1].get_title() + ' (Quadrupole)')
axes[2].set_title(axes[2].get_title() + ' (Octopole)')

width = axes[0].get_position().width * 1.25
height = axes[0].get_position().height * 1.4

# Adjust height across rows
for ax in axes[0:3]:
    pos = deepcopy(ax.get_position())
    pos.y0 = pos.y1 - height
    ax.set_position(pos)
for ax in axes[3:]:
    pos = deepcopy(ax.get_position())
    pos.y1 += 0.01
    pos.y0 = pos.y1 - height
    ax.set_position(pos)

# Adjust width across columns
for ax in (axes[0], axes[3]):
    pos = deepcopy(ax.get_position())
    pos.x1 = pos.x0 + width
    ax.set_position(pos)
for ax in (axes[2], axes[5]):
    pos = deepcopy(ax.get_position())
    pos.x0 = pos.x1 - width
    ax.set_position(pos)
for ax in (axes[1], axes[4]):
    pos = deepcopy(ax.get_position())
    mid_x = 0.5 * (pos.x0 + pos.x1)
    pos.x0 = mid_x - width / 2
    pos.x1 = mid_x + width / 2
    ax.set_position(pos)

fig.savefig(FIG_DIR / 'Fig3_spherical_decomposition.pdf',
            dpi=DPI, bbox_inches='tight')
