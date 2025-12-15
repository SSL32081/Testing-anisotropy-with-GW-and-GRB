import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

THETA_DIPOLE_DEG = 90.0 - (-6.944)
PHI_DIPOLE_DEG = 167.942
THETA_DIPOLE_RAD = np.deg2rad(THETA_DIPOLE_DEG)
PHI_DIPOLE_RAD = np.deg2rad(PHI_DIPOLE_DEG)

# load data
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

# calcuate dipole hemisphere
cos_gamma = (
    np.cos(THETA_DIPOLE_RAD) * np.cos(df_clean['theta']) +
    np.sin(THETA_DIPOLE_RAD) * np.sin(df_clean['theta']) * np.cos(df_clean['phi'] - PHI_DIPOLE_RAD)
)
df_clean['hemisphere_short'] = np.where(cos_gamma > 0, 'Forward', 'Backward')

N_forward = len(df_clean[df_clean['hemisphere_short'] == 'Forward'])
N_backward = len(df_clean[df_clean['hemisphere_short'] == 'Backward'])

forward_label = f'Forward hemisphere (N={N_forward})'
backward_label = f'Backward hemisphere (N={N_backward})'

# for label
df_clean['hemisphere_label'] = df_clean['hemisphere_short'].map({
    'Forward': forward_label,
    'Backward': backward_label
})

M_min = df_clean['M_value'].min()
M_max = df_clean['M_value'].max()
BIN_WIDTH = 5.0
bin_start = np.floor(M_min / BIN_WIDTH) * BIN_WIDTH
bin_end = np.ceil(M_max / BIN_WIDTH) * BIN_WIDTH
bins = np.arange(bin_start, bin_end + 1e-6, BIN_WIDTH)

# plot histogram
plt.style.use(['science','ieee', 'bright'])

sns.histplot(
    data=df_clean,
    x='M_value',
    hue='hemisphere_label', 
    bins=bins,
    stat='count',
    multiple='stack',
    kde=False,
)

plt.xlabel(r'Source Frame Chirp Mass $[M_{\odot}]$')
plt.ylabel('Number Counts')
plt.title(r'Dipole Axis ($\theta_D=$'+f'${THETA_DIPOLE_DEG:.1f}^\circ$, $\phi_D={PHI_DIPOLE_DEG:.1f}^\circ$)')

plt.tight_layout()
plt.savefig("chirpmass_dipole_hist.pdf", dpi=300)
