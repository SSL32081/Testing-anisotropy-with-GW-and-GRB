import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2 
import scienceplots 

def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df.columns = df.columns.str.strip().str.replace('#', '')
    
    rename_map = {
        'event_name': 'name',
        'chirp_mass_source[Msun]': 'M_value',
        'theta[rad]': 'theta',
        'phi[rad]': 'phi'
    }
    df = df.rename(columns=rename_map)
    df[['M_value', 'theta', 'phi']] = df[['M_value', 'theta', 'phi']].apply(pd.to_numeric, errors='coerce')
    return df.drop_duplicates(subset='name').dropna(subset=['M_value', 'theta', 'phi'])

def assign_hemispheres(df, theta_d, phi_d):
    cos_gamma = (np.cos(theta_d) * np.cos(df['theta']) + 
                 np.sin(theta_d) * np.sin(df['theta']) * np.cos(df['phi'] - phi_d))
    
    df['hemisphere'] = np.where(cos_gamma > 0, 'Forward', 'Backward')
    counts = df['hemisphere'].value_counts()
    label_map = {
        'Forward': f'Forward hemisphere ($N^F={counts.get("Forward", 0)}$)',
        'Backward': f'Backward hemisphere ($N^B={counts.get("Backward", 0)}$)'
    }
    df['hemisphere_label'] = df['hemisphere'].map(label_map)
    return df

def get_poisson_err(counts, alpha=0.10):
    low = np.where(counts > 0, 0.5 * chi2.ppf(alpha / 2, 2 * counts), 0)
    high = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (counts + 1))
    err = np.array([counts - low, high - counts])
    return np.where(counts > 0, err, np.nan)

def plot_hist(df, output_file):
    plt.style.use(['science', 'ieee', 'bright'])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    m_vals = df['M_value']
    bins = np.arange(np.floor(m_vals.min()/5)*5, np.ceil(m_vals.max()/5)*5 + 5.001, 5)
    
    sns.histplot(data=df, x='M_value', hue='hemisphere_label', bins=bins, 
                 multiple='stack', ax=ax, legend=True)

    f_counts, edges = np.histogram(df[df['hemisphere'] == 'Forward']['M_value'], bins=bins)
    b_counts, _ = np.histogram(df[df['hemisphere'] == 'Backward']['M_value'], bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.errorbar(centers, b_counts, yerr=get_poisson_err(b_counts), 
                fmt='none', capsize=3, color='C1', zorder=10)
    ax.errorbar(centers, f_counts + b_counts, yerr=get_poisson_err(f_counts), 
                fmt='none', capsize=3, color='C0', zorder=10)
    ax.set(xlabel=r'Source Frame Chirp Mass $[M_{\odot}]$', ylabel='Number Counts',
           title=r'Dipole Axis (RA=167.9$^\circ$, DEC=-6.9$^\circ$)')
    
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    return 
 
if __name__ == "__main__":
    RA_D, DEC_D = 167.942, -6.944
    THETA_D, PHI_D = np.deg2rad(90.0 - DEC_D), np.deg2rad(RA_D)
    
    df_clean = load_data("gw_location_data.txt")
    data = assign_hemispheres(df_clean, THETA_D, PHI_D)
    plot_hist(data, "chirpmass_dipole_hist.pdf")
