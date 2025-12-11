import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

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

# remove duplcated events
N_initial = len(df)
df.drop_duplicates(subset='name', keep='first', inplace=True)
N_unique = len(df)

# remove nan data
df_clean = df.dropna(subset=['M_value', 'theta']).copy()
N_total = len(df_clean)

# seperate event by Hemisphere
equator_rad = np.pi / 2.0
df_clean['hemisphere'] = np.where(df_clean['theta'] < equator_rad, 'North', 'South')

mc_north = df_clean[df_clean['hemisphere'] == 'North']['M_value'].values
mc_south = df_clean[df_clean['hemisphere'] == 'South']['M_value'].values

N_north = len(mc_north)
N_south = len(mc_south)
median_north = np.median(mc_north)
median_south = np.median(mc_south)
ks_statistic, p_value = stats.ks_2samp(mc_north, mc_south)
print("K-S test:")
print("max abs diff = ", ks_statistic)
print("p value = ", p_value)

plt.figure(figsize=(8, 6))
north_label = f'North Hemisphere (N={N_north})'
south_label = f'South Hemisphere (N={N_south})'

def custom_kde_plot(data, x, hue, **kwargs):
    
    color_map = {'North': 'blue', 'South': 'red'} 
    for name, group in data.groupby(hue):
        label = north_label if name == 'North' else south_label
        current_color = color_map.get(name, 'gray') 
        
        sns.kdeplot(data=group, x=x, 
            label=label, color=current_color, 
            fill=True, linewidth=2, 
            alpha=0.5, **{k: v for k, v in kwargs.items() if k != 'palette'})

custom_kde_plot(data=df_clean, x='M_value', hue='hemisphere',
                palette={'North': 'blue', 'South': 'red'}, bw_adjust=0.5)


plt.axvline(median_north, color='blue', linestyle=':', alpha=0.9, linewidth=1.5, label=f'NH Median: ${median_north:.1f}$ $M_{{\\odot}}$')
plt.axvline(median_south, color='red', linestyle=':', alpha=0.9, linewidth=1.5, label=f'SH Median: ${median_south:.1f}$ $M_{{\\odot}}$')

plt.xlabel(r'Source Frame Chirp Mass $[M_{\odot}]$', fontsize=12)
plt.ylabel('Observed Density', fontsize=12)
plt.legend(loc='upper right')
plt.tight_layout()

plt.savefig("chirpmass_dis.png", dpi=300)
