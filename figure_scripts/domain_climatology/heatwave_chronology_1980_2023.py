# ── Consolidated into figure_scripts/ on 2026-08-03 ─────────────────────────
# Original location: Heat_waves_bolzano/scripts/chronology_events_comparison_20260218_grid_only.py
# Paper figure:       HW_fixed_threshold_1980_2023_grid_only.pdf
# Note: Unrelated to the valley-lapse-rate methodology -- included only because it feeds a paper figure.
# UPDATE (2026-08-04): larger fonts + a "(b)" panel-label box added (this figure is stacked
# with aws_temperature_heatmap.py's "(a)" as one two-panel LaTeX figure).
# ─────────────────────────────────────────────────────────────────────────────

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 15,
    'axes.labelsize': 17,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})


def add_panel_label(ax, letter):
    ax.text(0.0, 1.1, f'({letter})',
            transform=ax.transAxes,
            fontsize=17, fontweight='bold',
            va='top', ha='left',
            bbox=dict(facecolor='white', edgecolor='black',
                      boxstyle='round,pad=0.3', linewidth=1.0),
            zorder=20)

file_path = "/mnt/CEPH_PROJECTS/RETURN/DS_CLIMATE/BOLZANO_URBAN_CASE/TASMAX_HEATWAVES/Heatwaves95p_fixed_1980_2025_Bolzano_Grid.csv"

df = pd.read_csv(file_path)
df['start_date'] = pd.to_datetime(df['start_date'], format='%d.%m.%y')
df['year'] = df['start_date'].dt.year

events_per_year = df.groupby('year').size()
mean_intensity = df.groupby('year')['Intensity'].mean()

fig, ax = plt.subplots(figsize=(12, 5))
ax_twin = ax.twinx()

# Bars
ax.bar(events_per_year.index, events_per_year.values,
       width=0.85, color='#999999', alpha=0.8,
       edgecolor='#555555', linewidth=0.5,
       label='Heatwave events')

# Intensity scatter + mean line
intensity_color = '#b03020'
ax_twin.scatter(df['year'], df['Intensity'],
                color=intensity_color, alpha=0.45, s=22, zorder=2,
                label='Individual events intensity')
ax_twin.plot(mean_intensity.index, mean_intensity.values,
             color=intensity_color, linewidth=2.0, zorder=3,
             label='Mean intensity')

# Axis labels
ax.set_xlabel('Year', labelpad=8)
ax.set_ylabel('Number of heatwave events', color='#222222', labelpad=8)
ax_twin.set_ylabel('Intensity (°C)', color=intensity_color, labelpad=8)

# Ticks
ax.set_xlim(1979, 2027)
ax.set_xticks(range(1980, 2027, 5))
ax.tick_params(axis='x', rotation=0)
ax.tick_params(axis='y', labelcolor='#222222')
ax_twin.tick_params(axis='y', labelcolor=intensity_color)
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Subtle horizontal gridlines (bars axis only)
ax.yaxis.grid(True, linestyle=':', linewidth=0.6, color='#bbbbbb')
ax.set_axisbelow(True)

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax_twin.spines['top'].set_visible(False)

# Combined legend
lines1, lbls1 = ax.get_legend_handles_labels()
lines2, lbls2 = ax_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, lbls1 + lbls2,
          frameon=True, framealpha=0.95, edgecolor='#cccccc',
          loc='upper left', ncol=1)

add_panel_label(ax, 'b')

plt.tight_layout()
os.makedirs('./PLOTS', exist_ok=True)
plt.savefig('./PLOTS/HW_fixed_threshold_1980_2023_grid_only.pdf', dpi=300, bbox_inches='tight')
plt.show()
