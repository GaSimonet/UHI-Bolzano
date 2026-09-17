# ── Consolidated into figure_scripts/ on 2026-08-03 ─────────────────────────
# Original location: multi_source_UHI_Bolzano_package/plot_for_comparing_resolution_of_data/spatio_tempo_compa_plot_v20260330.py
# Paper figure:       spatiotemporal_sketch.png
# Note: UNCONFIRMED best-candidate match, not a verified source. No script anywhere on disk saves the exact filename 'spatiotemporal_sketch.png'; this script's own output (spatiotemporal_improved_legend.png / spatiotemporal_sketch_raw.png) is the closest match by content and appears to have been manually finished/renamed into the cited figure. Confirm with whoever assembled UHI_bolzano/figures/ before relying on this.
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
import os

# Create output folder
output_folder = 'spatiotemporal_buildup'
os.makedirs(output_folder, exist_ok=True)

# Set publication-quality parameters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# Define data sources with their characteristics
data_sources = {
    'AWS': {
        'spatial_res': 0.01,
        'temporal_res': 0.20,
        'spatial_extent': 50,
        'temporal_extent': 20,
        'full_name': 'Automatic Weather Station'
    },
    'NetAtmo': {
        'spatial_res': 0.01,
        'temporal_res': 1/12,
        'spatial_extent': 20,
        'temporal_extent': 5,
        'full_name': 'Crowd Source Stations'
    },
    'Meteotrackers': {
        'spatial_res': 0.001,
        'temporal_res': 1/60,
        'spatial_extent': 10,
        'temporal_extent': 0.5,
        'full_name': 'Mobile Stations'
    },
    'WRF': {
        'spatial_res': 1.0,
        'temporal_res': 1,
        'spatial_extent': 500,
        'temporal_extent': 5,
        'full_name': 'Regional Atmospheric Model'
    },
    'UrbClim': {
        'spatial_res': 0.1,
        'temporal_res': 1,
        'spatial_extent': 30,
        'temporal_extent': 20,
        'full_name': 'Very High Resolution Urban Climate Model'
    },
    'Satellite': {
        'spatial_res': 0.25,
        'temporal_res': 24,
        'spatial_extent': 2000,
        'temporal_extent': 15,
        'full_name': 'Satellite Observation'
    }
}

# Axis limits
# x-axis (temporal scale, hours): minutes (<1 h), hours (1–24 h), days (24–168 h), weeks (>168 h)
X_LIM = (500, 0.005)
# y-axis (spatial scale, km): Micro (<1 km), Local (1–10 km), Meso (10–200 km), Macro (>200 km)
#   following Oke et al. urban climate classification
Y_LIM = (2000, 0.0001)

# Scale ranges used in the overlapping arrows (Oke et al. urban climate classification).
# Ranges overlap intentionally — see add_scale_lines() for details.


def add_scale_lines(ax):
    """Overlay meteorological scale arrows (Oke classification) with overlapping ranges.

    Arrows are staggered so that the fuzzy, overlapping nature of the boundaries
    is immediately visible (following Oke et al. Fig. 2.11 style).
    """

    trans_top   = blended_transform_factory(ax.transData, ax.transAxes)
    trans_right = blended_transform_factory(ax.transAxes, ax.transData)

    arrow_kw = dict(lw=1.5, mutation_scale=8)

    # --- Temporal scale arrows above x-axis (staggered rows, labels alternate above/below) ---
    # Rows are tighter (step 0.05); labels alternate sides to avoid vertical pile-up.
    # (x_start_h, x_end_h, y_axes_row, label, label_va, y_label_offset)
    temporal_arrows = [
        (0.005,  2.0,  1.04, 'Minutes', 'bottom', +0.012),
        (0.3,    50,   1.09, 'Hours',   'top',    -0.012),
        (12,     300,  1.14, 'Days',    'bottom', +0.012),
        (100,    500,  1.19, 'Weeks',   'top',    -0.012),
    ]
    for x0, x1, y_ax, label, va, dy in temporal_arrows:
        ax.annotate('',
            xy=(x1, y_ax), xytext=(x0, y_ax),
            xycoords=trans_top, textcoords=trans_top,
            annotation_clip=False,
            arrowprops=dict(arrowstyle='<->', color='grey', **arrow_kw))
        x_mid = np.sqrt(x0 * x1)   # geometric centre on log axis
        ax.text(x_mid, y_ax + dy, label,
                transform=trans_top, clip_on=False,
                fontsize=12, color='grey', ha='center', va=va,
                fontweight='bold')

    # --- Spatial scale arrows right of y-axis (staggered columns, labels alternate left/right) ---
    # Columns are tighter (step 0.05); labels alternate sides to avoid horizontal pile-up.
    # (x_axes_col, y_start_km, y_end_km, label, label_ha, x_label_offset)
    spatial_arrows = [
        (1.04, 0.0001,  2.0,  'Micro', 'left',  +0.012),
        (1.09, 0.3,     20,   'Local', 'right', -0.005),
        (1.14, 5,       500,  'Meso',  'left',  +0.012),
        (1.19, 100,     2000, 'Macro', 'right', -0.005),
    ]
    for x_ax, y0, y1, label, ha, dx in spatial_arrows:
        ax.annotate('',
            xy=(x_ax, y1), xytext=(x_ax, y0),
            xycoords=trans_right, textcoords=trans_right,
            annotation_clip=False,
            arrowprops=dict(arrowstyle='<->', color='grey', **arrow_kw))
        y_mid = np.sqrt(y0 * y1)   # geometric centre on log axis
        ax.text(x_ax + dx, y_mid, label,
                transform=trans_right, clip_on=False,
                fontsize=12, color='grey', ha=ha, va='center',
                fontweight='bold')


# ============= IMPROVED LEGEND PLOT =============
fig_legend, ax_legend = plt.subplots(figsize=(12, 9))

# Normalize temporal extent for color mapping
temp_extents = [props['temporal_extent'] for props in data_sources.values()]
norm = Normalize(vmin=0, vmax=max(temp_extents))
cmap = plt.cm.YlOrRd

# Plot data sources
for name, props in data_sources.items():
    x = props['temporal_res']   # x = temporal scale
    y = props['spatial_res']    # y = spatial scale
    bubble_size = props['spatial_extent'] * 25
    color_intensity = cmap(norm(props['temporal_extent']))

    ax_legend.scatter(x, y, s=bubble_size, alpha=0.8,
               color=color_intensity, marker='o',
               edgecolors='black', linewidth=2, zorder=3)

    # Large circles (WRF, Satellite): label centred inside; small circles: label above
    if bubble_size >= 5000:
        ax_legend.text(x, y, name, fontsize=10, ha='center', va='center',
                 fontweight='bold', zorder=4,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='black', alpha=0.85, linewidth=1))
    else:
        ax_legend.annotate(name, xy=(x, y),
                 xytext=(0, np.sqrt(bubble_size / np.pi) + 3),
                 textcoords='offset points',
                 fontsize=10, ha='center', va='bottom',
                 fontweight='bold', zorder=4,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='black', alpha=0.85, linewidth=1))

# Set log scales
ax_legend.set_xscale('log')
ax_legend.set_yscale('log')
# ax_legend.set_xlabel('Temporal Scale (hours)', fontsize=13, fontweight='bold')
# ax_legend.set_ylabel('Spatial Scale (km)', fontsize=13, fontweight='bold')
ax_legend.set_xlabel('Temporal scale & resolution (hours)', fontsize=13, fontweight='bold')
ax_legend.set_ylabel('Spatial scale &  resolution (km)', fontsize=13, fontweight='bold')

# Invert axes
ax_legend.invert_yaxis()
ax_legend.invert_xaxis()
ax_legend.set_xlim(*X_LIM)
ax_legend.set_ylim(*Y_LIM)

# Add scale reference lines and labels
add_scale_lines(ax_legend)

# Add grid
ax_legend.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, zorder=1)

# Add colorbar
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax_legend, orientation='horizontal', pad=0.12, fraction=0.046, aspect=40)
cbar.set_label('Temporal Extent (years)', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

# IMPROVED SIZE LEGEND - larger, more spaced out, with matching sizes
size_legend_elements = [
    plt.scatter([], [], s=10*25, c='gray', alpha=0.6, edgecolors='black', linewidth=1.5),
    plt.scatter([], [], s=50*25, c='gray', alpha=0.6, edgecolors='black', linewidth=1.5),
    plt.scatter([], [], s=200*25, c='gray', alpha=0.6, edgecolors='black', linewidth=1.5),
    plt.scatter([], [], s=600*25, c='gray', alpha=0.6, edgecolors='black', linewidth=1.5),
]
size_labels = ['10 km', '50 km', '200 km', '1000 km']

# Create legend with better spacing and scatterpoints to control size
legend1 = ax_legend.legend(size_legend_elements, size_labels,
                     title='Spatial Extent', loc='lower right',
                     frameon=True, framealpha=0.95, fontsize=11,
                     title_fontsize=12, labelspacing=3.0,
                     handletextpad=3.0, borderpad=2.5,
                     scatterpoints=1, markerscale=1,
                     borderaxespad=1)
legend1.get_title().set_fontweight('bold')
legend1.get_frame().set_linewidth(1.5)

plt.tight_layout()
plt.savefig(f'{output_folder}/spatiotemporal_improved_legend.png', dpi=300, bbox_inches='tight')
print(f"Figure saved: {output_folder}/spatiotemporal_improved_legend.png")

# ============= BUILD-UP SEQUENCE (NO RESEARCH ZONES) =============
print("\nCreating build-up sequence...")

# Get ordered list of data sources
source_names = list(data_sources.keys())

# Create a figure for each step
for i in range(1, len(source_names) + 1):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot only the first i data sources
    current_sources = source_names[:i]

    for name in current_sources:
        props = data_sources[name]
        x = props['temporal_res']   # x = temporal scale
        y = props['spatial_res']    # y = spatial scale
        bubble_size = props['spatial_extent'] * 25
        color_intensity = cmap(norm(props['temporal_extent']))

        ax.scatter(x, y, s=bubble_size, alpha=0.8,
                   color=color_intensity, marker='o',
                   edgecolors='black', linewidth=2, zorder=3)

        if bubble_size >= 5000:
            ax.text(x, y, name, fontsize=10, ha='center', va='center',
                    fontweight='bold', zorder=4,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='black', alpha=0.85, linewidth=1))
        else:
            ax.annotate(name, xy=(x, y),
                    xytext=(0, np.sqrt(bubble_size / np.pi) + 3),
                    textcoords='offset points',
                    fontsize=10, ha='center', va='bottom',
                    fontweight='bold', zorder=4,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='black', alpha=0.85, linewidth=1))

    # Set log scales
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Temporal Scale (hours)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Spatial Scale (km)', fontsize=13, fontweight='bold')

    # Invert axes
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)

    # Add scale reference lines and labels
    add_scale_lines(ax)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, zorder=1)

    # Add colorbar
    sm_step = ScalarMappable(cmap=cmap, norm=norm)
    sm_step.set_array([])
    cbar_step = plt.colorbar(sm_step, ax=ax, orientation='horizontal',
                             pad=0.12, fraction=0.046, aspect=40)
    cbar_step.set_label('Temporal Extent (years)', fontsize=12, fontweight='bold')
    cbar_step.ax.tick_params(labelsize=10)

    # Add size legend with improved spacing
    legend_step = ax.legend(size_legend_elements, size_labels,
                           title='Spatial Extent', loc='lower right',
                           frameon=True, framealpha=0.95, fontsize=11,
                           title_fontsize=12, labelspacing=3.0,
                           handletextpad=3.0, borderpad=2.5,
                           scatterpoints=1, markerscale=1,
                           borderaxespad=1)
    legend_step.get_title().set_fontweight('bold')
    legend_step.get_frame().set_linewidth(1.5)

    # Add title with full name of the data source
    title_text = data_sources[source_names[i-1]]['full_name']
    ax.set_title(title_text, fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()

    # Save each step
    filename_base = f'{output_folder}/spatiotemporal_step_{i:02d}.png'
    plt.savefig(filename_base, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename_base} (sources: {', '.join(current_sources)})")

    plt.close()

print("\n=== All figures generated successfully ===")
print(f"\nGenerated {len(source_names)} step-by-step figures in folder: {output_folder}/")
print(f"- {output_folder}/spatiotemporal_improved_legend.png (main figure with improved legend)")
print(f"- {output_folder}/spatiotemporal_step_01.png through {output_folder}/spatiotemporal_step_06.png")
print("\nOrder of data sources added:")
for i, name in enumerate(source_names, 1):
    print(f"  Step {i}: {name}")
