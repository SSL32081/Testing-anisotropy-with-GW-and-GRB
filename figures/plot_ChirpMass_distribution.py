import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('../matplotlibrc')

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

ns_label_colour = {
    'North': ('blue', f'North Hemisphere (N={N_north})'),
    'South': ('red', f'South Hemisphere (N={N_south})'),
}


def custom_kde_plot(data, x, hue, ax=plt.cla(), **kwargs):

    for name, group in data.groupby(hue):
        label = ns_label_colour.get(name, '')
        current_color = ns_label_colour.get(name, 'gray')

        sns.kdeplot(data=group, x=x,
                    label=label, color=current_color,
                    fill=True, linewidth=2,
                    alpha=0.5, ax=ax, **kwargs)


fig, ax = plt.subplots(1, 1, constrained_layout=True)
custom_kde_plot(data=df_clean, x='M_value', ax=ax, hue='hemisphere',
                palette={'North': 'blue', 'South': 'red'}, bw_adjust=0.5)

linekws = dict(color='blue', linestyle=':', alpha=0.9, linewidth=1.5)
ax.axvline(median_north, label=fr'NH Median: ${median_north:.1f}\,M_\odot$', **linekws)
ax.axvline(median_south, label=fr'SH Median: ${median_south:.1f}\,M_\odot$', **linekws)

ax.set_xlabel(r'Source Frame Chirp Mass $[M_{\odot}]$', fontsize=12)
ax.set_ylabel('Observed Density', fontsize=12)
ax.legend(loc='upper right')

fig.savefig("chirpmass_dis.png", dpi=300)
