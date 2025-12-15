import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import matplotlib.patches as mpatches 

THETA_DIPOLE_DEG = 90.0 - (-6.944)
PHI_DIPOLE_DEG = 167.942
THETA_DIPOLE_RAD = np.deg2rad(THETA_DIPOLE_DEG)
PHI_DIPOLE_RAD = np.deg2rad(PHI_DIPOLE_DEG)

df = pd.read_csv("gw_location_data.txt", sep='\t')
df.columns = df.columns.str.strip()
if df.columns[0].startswith('#'):
    df.rename(columns={df.columns[0]: df.columns[0].lstrip('#')}, inplace=True)

df.rename(columns={
    'event_name': 'name',
    'chirp_mass_source[Msun]': 'M_value',
    'luminosity_distance[Mpc]': 'd_L',
    'theta[rad]': 'theta',
    'phi[rad]': 'phi'
}, inplace=True)

for col in ['M_value', 'd_L', 'theta', 'phi']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.drop_duplicates(subset='name', keep='first', inplace=True)
df_clean = df.dropna(subset=['M_value', 'theta', 'phi']).copy()

# Dipole Hemispheres Calculation 
cos_gamma = (
    np.cos(THETA_DIPOLE_RAD) * np.cos(df_clean['theta']) +
    np.sin(THETA_DIPOLE_RAD) * np.sin(df_clean['theta']) * np.cos(df_clean['phi'] - PHI_DIPOLE_RAD)
)
df_clean['hemisphere'] = np.where(cos_gamma > 0, 'Forward', 'Backward')

mc_forward = df_clean[df_clean['hemisphere'] == 'Forward']['M_value'].values
mc_backward = df_clean[df_clean['hemisphere'] == 'Backward']['M_value'].values
N_forward = len(mc_forward)
N_backward = len(mc_backward)


M_min = df_clean['M_value'].min()
M_max = df_clean['M_value'].max()
BIN_WIDTH = 5.0
bin_start = np.floor(M_min / BIN_WIDTH) * BIN_WIDTH
bin_end = np.ceil(M_max / BIN_WIDTH) * BIN_WIDTH
bins = np.arange(bin_start, bin_end + 1e-6, BIN_WIDTH)

plt.style.use(['science','ieee', 'bright'])
plt.figure(figsize=(8, 5))

# Plot histogram
sns.histplot(
    data=df_clean,
    x='M_value',
    hue='hemisphere',
    bins=bins,
    stat='count',
    multiple='stack',
    palette={'Forward': 'red', 'Backward': 'blue'},
    edgecolor='black',
    linewidth=0.5,
    alpha=0.6,
    kde=False,
    legend=False 
)

plt.xlabel(r'Source Frame Chirp Mass $[M_{\odot}]$', fontsize=12)
plt.ylabel('Number Counts', fontsize=12)
plt.title(f'Chirp Mass Number Counts Split by Dipole Axis ($\Theta_D={THETA_DIPOLE_DEG:.1f}^\circ$, $\Phi_D={PHI_DIPOLE_DEG:.1f}^\circ$)')


# legend
forward_label = f'Forward hemisphere (N={N_forward})'
backward_label = f'Backward hemisphere (N={N_backward})'
forward_patch = mpatches.Patch(facecolor='red', label=forward_label, alpha=0.6)
backward_patch = mpatches.Patch(facecolor='blue', label=backward_label, alpha=0.6)
plt.legend(handles=[forward_patch, backward_patch], loc='upper right') 

plt.tight_layout()
plt.savefig("chirpmass_dipole_hist.pdf", dpi=300)
plt.savefig("chirpmass_dipole_hist.png", dpi=300)
