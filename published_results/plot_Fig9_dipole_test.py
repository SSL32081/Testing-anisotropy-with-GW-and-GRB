#!/usr/bin/env python3
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

from utils import FIG_DIR, DPI, DOUBLE

N_F = 68
N_B = 104
nsamp = 200_000

print(np.__version__)


def compute_beta_samples(lambda_range, forward_posterior, backward_posterior):
    # Sample lambda_F and lambda_B from their posteriors
    lambda_F_samp = np.random.choice(lambda_range, size=nsamp, p=forward_posterior)
    lambda_B_samp = np.random.choice(lambda_range, size=nsamp, p=backward_posterior)

    # Ratio distribution r = lambda_F / lambda_B
    r_samp = lambda_F_samp / lambda_B_samp

    # Define dipole parameter beta via the usual parametrization
    #   lambda_F = lambda * (1 + beta/2),  lambda_B = lambda * (1 - beta/2)
    # so that
    #   r = lambda_F / lambda_B = (1 + beta/2) / (1 - beta/2)
    # which implies
    #   beta = 2 * (r - 1) / (r + 1)
    return 2.0 * (r_samp - 1.0) / (r_samp + 1.0)


def compute_posteriors():
    # Plot the integrands as a function of lambda
    lambda_range = np.arange(1, 201, 1)

    # Compute integrands for plotting
    integrand_I_vals, integrand_A_F_vals, integrand_A_B_vals = [], [], []
    for lam in lambda_range:
        dist = poisson(mu=lam)
        pmf_F, pmf_B = dist.pmf([N_F, N_B])
        integrand_I_vals.append(pmf_F * pmf_B)
        integrand_A_F_vals.append(pmf_F)
        integrand_A_B_vals.append(pmf_B)
    integrand_I_vals = np.array(integrand_I_vals)
    integrand_A_F_vals = np.array(integrand_A_F_vals)
    integrand_A_B_vals = np.array(integrand_A_B_vals)

    posterior_I_vals = integrand_I_vals / np.trapz(integrand_I_vals, lambda_range)
    posterior_A_F_vals = integrand_A_F_vals / np.trapz(integrand_A_F_vals, lambda_range)
    posterior_A_B_vals = integrand_A_B_vals / np.trapz(integrand_A_B_vals, lambda_range)

    return lambda_range, posterior_I_vals, posterior_A_F_vals, posterior_A_B_vals


def main():
    # Color scheme
    color_iso = '#40b0a6'   # teal for isotropic posterior
    color_F   = '#e66101'   # warm orange for forward hemisphere
    color_B   = '#5e3c99'   # deep violet for backward hemisphere
    color_cmb = '#d62728'   # red for CMB dipole
    color_beta = '#2c3e50'   # midnight blue - sophisticated, distinct

    # CMB dipole from motion: v ≈ 370 km/s → β ≈ 0.00123
    beta_cmb = 370 / 299792.458  # v/c

    lambdas, post_I, post_A_F, post_A_B = compute_posteriors()
    beta_samp = compute_beta_samples(lambdas, post_A_F, post_A_B)

    # Calculate quantiles from your beta samples
    q_05, q_50, q_95 = np.percentile(beta_samp, [5, 50, 95])    # 5th percentile (lower bound)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 3.1))

    # Left panel: Lambda posteriors
    ax1.plot(lambdas, post_I, color=color_iso,
             label=r'$p(\lambda \mid N_{\rm F}, N_{\rm B})$')
    ax1.plot(lambdas, post_A_F, color=color_F,
             label=r'$p(\lambda_{\rm F} \mid N_{\rm F})$')
    ax1.plot(lambdas, post_A_B, color=color_B,
             label=r'$p(\lambda_{\rm B} \mid N_{\rm B})$')

    ax1.set_xlim(40, 150)
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel(r'$\lambda$')
    ax1.set_ylabel('PDF')
    ax1.legend(frameon=True, edgecolor='black')

    # Right panel: Beta posterior with CMB dipole
    ax2.hist(beta_samp, bins=80, density=True, color=color_beta, alpha=0.65,
             edgecolor='none', label=r'$p(\beta \mid N_{\rm F}, N_{\rm B})$')
    ax2.axvline(beta_cmb, color=color_cmb, linewidth=1.5, linestyle='--',
                label=f'$\\beta = {beta_cmb:.4f}$')

    # Just median + shaded region (cleaner look)
    ax2.axvspan(q_05, q_95, alpha=0.15, color=color_beta, zorder=0)
    ax2.axvline(q_50, color='#1a252f', linewidth=1.2, linestyle='-')

    ax2.set_xlabel(r'$\beta$')
    ax2.set_ylabel(r'$p(\beta \mid N_{\rm F}, N_{\rm B})$')
    ax2.legend(frameon=True, edgecolor='black')#
    ax2.set_xlim(-1.00, 0.4)
    fig.savefig(FIG_DIR / 'Fig9_dipole_beta_posterior.pdf', dpi=DPI)


if __name__ == '__main__':
    main()

