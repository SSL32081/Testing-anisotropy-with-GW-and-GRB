#!/usr/bin/env python3
from h5py import File
import argparse
import numpy as np
import numpy.lib.recfunctions as rfn
from scipy.stats import chi2
import matplotlib.pyplot as plt

from utils import PARENT_DIR, FIG_DIR, DPI, SINGLE

parser = argparse.ArgumentParser(
    description='Options for computing the gamma fit of synthetic data.')
parser.add_argument('-n', type=int, default=500,
                    help='Number of realisations to draw.')
parser.add_argument('--errors', action='store_true', 
                    help='Whether to plot error bars on the histogram.')
parser.add_argument('--nocosmo', action='store_false',
                    help='Whether to use cosmology-corrected mass samples.')

args = parser.parse_args()
N_realisations = args.n
N_samples = 1000

def read_skyloc_mass_samples(skyloc_file, mass_file):
    subkeys = ['ra', 'dec', 'chirp_mass']
    all_samples = np.array([[]] * N_realisations, dtype=[(key, 'f8') for key in subkeys])
    with File(skyloc_file, 'r') as f_skyloc, File(mass_file, 'r') as f_mass:
        for event in f_skyloc.keys():
            try:
                skyloc_group = f_skyloc[event + '/C01:Mixed/skyloc_samples']
                mass_group = f_mass[event + '/C01:Mixed/mass_samples']
            except KeyError:
                try:
                    skyloc_group = f_skyloc[event + '/C00:Mixed/skyloc_samples']
                    mass_group = f_mass[event + '/C00:Mixed/mass_samples']
                except KeyError:
                    print(f"Missing C00 or C01 group for event {event}")
                    continue

            if skyloc_group.size < N_samples:
                print(f"Not enough samples for event {event}: only {skyloc_group.size} samples")
                continue

            names = list(skyloc_group.dtype.names)
            names.remove('redshift')
            if np.all(skyloc_group['redshift'] == mass_group['redshift']):
                skyloc = skyloc_group[tuple(names)]
                merged = rfn.merge_arrays(
                    [skyloc, mass_group], usemask=False, flatten=True)
                merged['chirp_mass'] /= (1 + merged['redshift'])
                subsamples = np.vstack([
                    np.random.choice(merged[subkeys], size=N_samples, replace=False) 
                        for _ in range(N_realisations)
                ])
                all_samples = np.concatenate([all_samples, subsamples], axis=1)
            else:
                print(f"Redshift mismatch for event {event}")
                continue

    return all_samples


def assign_hemispheres(df, dec_d, ra_d):
    cos_gamma = np.cos(dec_d) * np.cos(df['dec']) + \
        np.sin(dec_d) * np.sin(df['dec']) * np.cos(df['ra'] - ra_d)

    return np.where(cos_gamma > 0, +1, -1)


def get_poisson_err(counts, alpha=0.10):
    low = np.where(counts > 0, 0.5 * chi2.ppf(alpha / 2, 2 * counts), 0)
    high = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (counts + 1))
    err = np.array([counts - low, high - counts])
    return np.where(counts > 0, err, np.nan)


def plot_hist(samples, ax_dec, ax_ra, plot_err=False):
    # Mc = samples['chirp_mass']
    # bins = np.linspace(np.min(Mc) - 3, np.max(Mc) + 3, 51)
    bins = np.arange(0, 121, 5)
    centres = (bins[:-1] + bins[1:]) / 2

    pos_counts, neg_counts = [], []
    for realisation in samples:
        labels = assign_hemispheres(realisation, ax_dec, ax_ra)
        pos_count, _ = np.histogram(realisation['chirp_mass'][labels == +1], bins=bins)
        neg_count, _ = np.histogram(realisation['chirp_mass'][labels == -1], bins=bins)
        pos_counts.append(pos_count / N_samples)
        neg_counts.append(neg_count / N_samples)

    fig, ax = plt.subplots(figsize=(SINGLE, 3.1))

    pos_counts_stats = np.percentile(pos_counts, [5, 50, 95], axis=0)
    neg_counts_stats = np.percentile(neg_counts, [5, 50, 95], axis=0)

    pos_count = np.sum(pos_counts) / N_realisations
    neg_count = np.sum(neg_counts) / N_realisations

    ax.stairs(pos_counts_stats[1], bins, hatch='//', color='C0',
            label=fr'Forward hemisphere ($N^{{\rm F}}={pos_count:.2f}$)',
            zorder=5)
    ax.stairs(neg_counts_stats[1], bins, ls='-', hatch='\\\\', color='C1',
            label=fr'Backward hemisphere ($N^{{\rm B}}={neg_count:.2f}$)')

    ax.fill_between(
            centres, pos_counts_stats[0], pos_counts_stats[2],
            alpha=0.5, color='C0', step='mid', lw=0,
        )
    ax.fill_between(
            centres, neg_counts_stats[0], neg_counts_stats[2],
            alpha=0.5, color='C1', step='mid', lw=0,
        )
    if plot_err:
        err_kws = dict(fmt='none', capsize=2, zorder=10, alpha=0.7, linewidth=0.8)
        ax.errorbar(centres - 0.5, pos_counts_stats[1], 
                    yerr=get_poisson_err(pos_counts_stats[1]),
                    color='C0', **err_kws)
        ax.errorbar(centres + 0.5, neg_counts_stats[1], 
                    yerr=get_poisson_err(neg_counts_stats[1]),
                    color='C1', **err_kws)
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, 120)

    ax.set(xlabel=r'Source Frame Chirp Mass, ${\cal M}_{\rm c}^{\rm src}\,/\,M_{\odot}$',
           ylabel='Effective Event Counts',
           title=r'Dipole Axis (${\rm RA}=167.9^\circ, {\rm DEC}={-}6.9^\circ$)')

    return fig


def main():
    cosmo_txt = 'cosmo' if args.nocosmo else 'nocosmo'
    errorbars = args.errors
    RA_D, DEC_D = 167.942, -6.944
    ax_dec, ax_ra = np.deg2rad(90.0 - DEC_D), np.deg2rad(RA_D)

    gwtc2p1_samples = read_skyloc_mass_samples(
        PARENT_DIR / f'GWTC2p1_{cosmo_txt}_skyloc_samples.h5',
        PARENT_DIR / f'../LVK_mass_samples/GWTC2p1_{cosmo_txt}_mass_samples.h5'
    )
    gwtc3p0_samples = read_skyloc_mass_samples(
        PARENT_DIR / f'GWTC3p0_{cosmo_txt}_skyloc_samples.h5',
        PARENT_DIR / f'../LVK_mass_samples/GWTC3p0_{cosmo_txt}_mass_samples.h5'
    )
    gwtc4p0_samples = read_skyloc_mass_samples(
        PARENT_DIR / 'GWTC4p0_skyloc_samples.h5',
        PARENT_DIR / '../LVK_mass_samples/GWTC4p0_mass_samples.h5'
    )
    all_samples = np.concatenate([
        gwtc2p1_samples, gwtc3p0_samples, gwtc4p0_samples
    ], axis=1)
    print(all_samples.shape)

    fig = plot_hist(all_samples, ax_dec, ax_ra, plot_err=errorbars)
    suffix = '_with_err' if errorbars else ''
    fig.savefig(FIG_DIR / f'Fig8_dipole_mass_{cosmo_txt}_distribution{suffix}_N{N_samples:d}.pdf', dpi=DPI)
    return fig


if __name__ == '__main__':
    main()
