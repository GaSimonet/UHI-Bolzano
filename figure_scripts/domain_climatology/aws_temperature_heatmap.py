# ── Consolidated into figure_scripts/ on 2026-08-03 ─────────────────────────
# Original location: /mnt/CEPH_PROJECTS/RETURN/DS_CLIMATE/BOLZANO_URBAN_CASE/AWS_alto_Adige/scripts/01_process/bolzano_heatmap_only.py
# Paper figure:       AWS_temperature_heatmap_hour_month.pdf
# Note: Unrelated to the valley-lapse-rate methodology (plain AWS station climatology, no urban/rural UHI correction) -- included only because it feeds a paper figure. Original output filename had a '07_' numeric prefix and no 'AWS_' prefix; renamed when copied into the paper's figures/ folder.
# UPDATE (2026-08-04): larger fonts + an "(a)" panel-label box added (this figure is stacked
# with heatwave_chronology_1980_2023.py's "(b)" as one two-panel LaTeX figure).
# ─────────────────────────────────────────────────────────────────────────────

#%% Setup and Data Loading
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os

file_path = '/home/gsimonet/Desktop/AWS_alto_Adige/CSV/organized/8320LTV0010A_Bozen_20030101_20240224.csv'
invalid_value = -777
output_dir = 'weather_plots'
os.makedirs(output_dir, exist_ok=True)

#%% Read and Clean Data
df = pd.read_csv(file_path, skiprows=12, sep=',',
                 names=['timestamp', 'value', 'status'])
df['timestamp'] = df['timestamp'].str.replace('+1', '+01:00', regex=False)
df['timestamp'] = df['timestamp'].str.replace('+2', '+02:00', regex=False)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df['value'] = pd.to_numeric(df['value'], errors='coerce')
df = df.dropna(subset=['timestamp', 'value'])
df_valid = df[df['value'] != invalid_value].copy()
df_valid['hour']  = df_valid['timestamp'].dt.hour
df_valid['month'] = df_valid['timestamp'].dt.month

print(f"Valid records : {len(df_valid)}")
print(f"Date range    : {df_valid['timestamp'].min()} → {df_valid['timestamp'].max()}")
print(f"Temp range    : {df_valid['value'].min():.1f} – {df_valid['value'].max():.1f} °C")

#%% Pivot Table
# rows = hour (0–23), columns = month (1–12)
heatmap_data = df_valid.pivot_table(
    values='value', index='hour', columns='month', aggfunc='mean'
)

# Marginal means
monthly_mean = heatmap_data.mean(axis=0)   # mean over all hours  → shape (12,)
hourly_mean  = heatmap_data.mean(axis=1)   # mean over all months → shape (24,)

month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

#%% Colormap shared by heatmap and marginals
cmap = 'OrRd'
vmin = heatmap_data.min().min()
vmax = heatmap_data.max().max()
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
cmap_obj = plt.get_cmap(cmap)

#%% Publication-quality font settings (scoped to this figure)
pub_rc = {
    'font.family': 'serif',
    'font.size': 15,
    'axes.labelsize': 17,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
}


def add_panel_label(ax, letter):
    ax.text(0.0, 1.12, f'({letter})',
            transform=ax.transAxes,
            fontsize=17, fontweight='bold',
            va='top', ha='left',
            bbox=dict(facecolor='white', edgecolor='black',
                      boxstyle='round,pad=0.3', linewidth=1.0),
            zorder=20)

with mpl.rc_context(pub_rc):
    fig = plt.figure(figsize=(13, 9))
    gs = GridSpec(2, 3, figure=fig,
                  width_ratios=[10, 2, 0.4],
                  height_ratios=[2, 10],
                  hspace=0.05, wspace=0.05)

    ax_top   = fig.add_subplot(gs[0, 0])   # monthly distribution (top)
    ax_heat  = fig.add_subplot(gs[1, 0])   # main heatmap
    ax_right = fig.add_subplot(gs[1, 1])   # hourly distribution (right)
    ax_cbar  = fig.add_subplot(gs[1, 2])   # colorbar

    # ── Main heatmap ──────────────────────────────────────────────────────────
    # Seaborn internally calls invert_yaxis() → hour 0 ends up at the top.
    # We call it again to flip back: hour 0 at the bottom, 23:00 at the top.
    sns.heatmap(heatmap_data, ax=ax_heat, cmap=cmap,
                vmin=vmin, vmax=vmax, cbar=False,
                linewidths=0.3, linecolor='white')
    ax_heat.invert_yaxis()   # net result: 00:00 at bottom, 23:00 at top

    ax_heat.set_xlabel('Month', labelpad=6)
    ax_heat.set_ylabel('Hour of day', labelpad=6)
    ax_heat.set_xticks([i + 0.5 for i in range(12)])
    ax_heat.set_xticklabels(month_labels, rotation=0)
    ax_heat.set_yticks([i + 0.5 for i in range(24)])
    ax_heat.set_yticklabels([f'{h:02d}:00' for h in range(24)], rotation=0, fontsize=12)

    # ── Top marginal: monthly mean temperature ────────────────────────────────
    # x positions 0.5, 1.5, ..., 11.5 align with heatmap columns
    bar_x      = [i + 0.5 for i in range(12)]
    colors_top = [cmap_obj(norm(v)) for v in monthly_mean.values]
    ax_top.bar(bar_x, monthly_mean.values, color=colors_top,
               width=0.8, edgecolor='white', linewidth=0.5)
    ax_top.set_xlim(0, 12)   # matches heatmap x range
    ax_top.set_xticks([])
    ax_top.set_ylabel('Mean T (°C)')
    ax_top.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)

    # ── Right marginal: hourly mean temperature ───────────────────────────────
    # y positions 0.5, 1.5, ..., 23.5 align with heatmap rows.
    # After the double invert_yaxis on the heatmap, its y-axis increases
    # upward (0 at bottom → 24 at top).  We match that here with set_ylim(0,24)
    # and NO invert_yaxis, so hour 0 stays at the bottom and 23 at the top.
    bar_y        = [i + 0.5 for i in range(24)]
    colors_right = [cmap_obj(norm(v)) for v in hourly_mean.values]
    ax_right.barh(bar_y, hourly_mean.values, color=colors_right,
                  height=0.8, edgecolor='white', linewidth=0.5)
    ax_right.set_ylim(0, 24)   # matches heatmap y range
    # no invert_yaxis → y increases upward, same as the heatmap after its flip
    ax_right.set_yticks([])
    ax_right.set_xlabel('Mean T (°C)')
    ax_right.grid(axis='x', alpha=0.3, linewidth=0.5)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cbar)
    cb.set_label('Temperature (°C)', fontsize=15)
    cb.ax.tick_params(labelsize=13)

    add_panel_label(ax_top, 'a')

    plt.savefig(f'{output_dir}/07_temperature_heatmap_hour_month.pdf',
                dpi=300, bbox_inches='tight')
    plt.show()

print("Done — saved to weather_plots/07_temperature_heatmap_hour_month.pdf")
