# ── Consolidated into figure_scripts/ on 2026-08-03 ─────────────────────────
# Original location: WRF_alto_adige_local/script/analysis/UHI/seasonal_UHI_WRF_KEEPME_20260223.py
# Paper figure:       seasonal_uhi_maps_corrected.png
# Note: Legacy/current source of this figure as currently embedded in the paper -- kept verbatim (its own docstring says 'kept untouched, not overwritten'). Uses a FIXED 0.0065 K/m lapse rate, unlike wrf_uhi/seasonal_uhi_maps_dynamiclapse.py in this same folder, which replaces it with the dynamic per-(season,day/night) valley rate. Regenerate the paper figure from the dynamiclapse version to pick up that methodology.
# ─────────────────────────────────────────────────────────────────────────────

# Import needed libraries
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import salem
import pandas as pd
import geopandas as gpd
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

print('need to be run under ROI_Python_38 env')

print('Starting UHI Seasonal Analysis...')

#%% Define paths and parameters
WRF_FILE = "/mnt/CEPH_PROJECTS/RETURN/DS_CLIMATE/BOLZANO_URBAN_CASE/WRF_ALTO_ADIGE/WRF_RAW_2020_subset/WRF_2020_RAW_subset.nc"
WRF_FILE = "/mnt/CEPH_PROJECTS/RETURN/DS_CLIMATE/BOLZANO_URBAN_CASE/WRF_ALTO_ADIGE/WRF_RAW_2021_subset/WRF_RAW_Alto_Adige_2021.nc"
WRF_FILE = "/mnt/CEPH_PROJECTS/RETURN/DS_CLIMATE/BOLZANO_URBAN_CASE/WRF_ALTO_ADIGE/WRF_RAW_2022_subset/WRF_RAW_Alto_Adige_2022.nc"

WRF_FILE = "/mnt/CEPH_PROJECTS/RETURN/DS_CLIMATE/BOLZANO_URBAN_CASE/WRF_ALTO_ADIGE/WRF_RAW_2023_subset/WRF_RAW_Alto_Adige_2023.nc"
WRF_FILE = "/mnt/CEPH_PROJECTS/RETURN/DS_CLIMATE/BOLZANO_URBAN_CASE/WRF_ALTO_ADIGE/WRF_RAW_2024_subset/WRF_RAW_Alto_Adige_2024.nc"


URBAN_SHAPE = "/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_urban_area.shp"
RURAL_SHAPE = "/home/gsimonet/Desktop/WRF_alto_adige_local/script/analysis/LCZ_shapefile_analysis/shapefiles/Bolzano_rural_area.shp"


LAPSE_RATE = 0.0065  # °C/m or K/m (standard environmental lapse rate)

#%% Load and display WRF domain with optimized approach
print("Loading WRF dataset with optimization...")

# 1. Use dask for parallel processing
import dask
dask.config.set({"array.slicing.split_large_chunks": False})

# 2. Load only necessary variables to reduce memory usage
parameters = ['W', 'THETA', 'Z', 'HGT', 'T','T2',
              'P','U','V','lat','lon']
print("Loading only necessary variables...")
ds = salem.open_wrf_dataset(WRF_FILE)[parameters]

# 3. Optional: Load subset of time if dataset is very large
# Use this if your full year analysis is still too slow
# ds = salem.open_wrf_dataset(WRF_FILE, vars=needed_vars, time_slice=slice('2021-01-01', '2021-12-31', 2))  # Every other day

# 4. Spatial pre-filtering: Get domain bounds from shapefiles to pre-filter spatially
print("Determining geographical subset bounds from shapefiles...")
try:
    # Read shapefile bounds without loading full dataset
    urban_shp = gpd.read_file(URBAN_SHAPE)
    rural_shp = gpd.read_file(RURAL_SHAPE)
    
    # Get bounding box with buffer
    urban_bounds = urban_shp.total_bounds
    rural_bounds = rural_shp.total_bounds
    
    # Combine bounds with a buffer (extend by 10%)
    min_lon = min(urban_bounds[0], rural_bounds[0])
    min_lat = min(urban_bounds[1], rural_bounds[1])
    max_lon = max(urban_bounds[2], rural_bounds[2])
    max_lat = max(urban_bounds[3], rural_bounds[3])
    
    # Add buffer
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    min_lon = min_lon - 0.1 * lon_range
    min_lat = min_lat - 0.1 * lat_range
    max_lon = max_lon + 0.1 * lon_range
    max_lat = max_lat + 0.1 * lat_range
    
    # Pre-filter dataset spatially
    print("Pre-filtering dataset by geographical bounds...")
    mask = ((ds.XLAT >= min_lat) & (ds.XLAT <= max_lat) & 
            (ds.XLONG >= min_lon) & (ds.XLONG <= max_lon))
    
    # Use isel with boolean mask for first time step (faster than where)
    first_time_mask = mask.isel(Time=0).values
    y_indices, x_indices = np.where(first_time_mask)
    
    if len(y_indices) > 0 and len(x_indices) > 0:
        # Determine slice bounds with small buffer
        y_min, y_max = max(0, y_indices.min() - 2), min(ds.sizes['south_north'] - 1, y_indices.max() + 2)
        x_min, x_max = max(0, x_indices.min() - 2), min(ds.sizes['west_east'] - 1, x_indices.max() + 2)
        
        # Apply spatial subset using isel (much faster than where)
        print("Applying coarse geographical subset...")
        ds = ds.isel(south_north=slice(y_min, y_max + 1), 
                     west_east=slice(x_min, x_max + 1))
except Exception as pre_subset_error:
    print(f"Pre-subsetting failed, will continue with full domain: {pre_subset_error}")

# Plot the domain with terrain for verification
plt.figure(figsize=(12, 10))
ds.HGT[0].salem.quick_map()
plt.grid(True)
plt.title('WRF Domain Map - Full Year')
plt.savefig('wrf_domain_map.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Create urban and rural subsets
print("Creating urban and rural subsets...")

# # 5. Use chunking to optimize memory usage during subsetting
# chunk_size = {'time': 'auto', 'south_north': 'auto', 'west_east': 'auto'}
# ds = ds.chunk(chunk_size)

# Use more specific chunking strategy
chunk_size = {
    'time': min(100, ds.dims['time']),        # 100 timesteps per chunk
    'south_north': min(200, ds.dims['south_north']),  # 200 grid cells per chunk 
    'west_east': min(200, ds.dims['west_east'])       # 200 grid cells per chunk
}
ds = ds.chunk(chunk_size)

# 6. Set compute=False to delay computation until necessary
try:
    print("Subsetting rural areas...")
    ds_rural = ds.salem.subset(shape=RURAL_SHAPE)#, compute=True)
    print("Subsetting urban areas...")
    ds_urban = ds.salem.subset(shape=URBAN_SHAPE)#, compute=False)
    
    # # 7. Force computation with optimized parallel settings
    # with dask.config.set(scheduler='threads', num_workers=4):  # Adjust workers based on your CPU
    #     print("Computing rural subset...")
    #     ds_rural = ds_rural.compute()
    #     print("Computing urban subset...")
    #     ds_urban = ds_urban.compute()
    
    print("Creating ROI...")
    ds_urban_roi = ds_rural.salem.roi(shape=URBAN_SHAPE)
    
except Exception as e:
    print(f"Error during subsetting: {e}")
    print("Attempting to import shapefiles directly and continue...")
    try:
        # Alternative approach using geopandas directly
        if 'urban_shp' not in locals():
            urban_shp = gpd.read_file(URBAN_SHAPE)
        if 'rural_shp' not in locals():
            rural_shp = gpd.read_file(RURAL_SHAPE)
        # Continue with available data...
        
        # 8. Fallback to simplified subsetting if full salem subsetting fails
        print("Falling back to simplified spatial subsetting...")
        ds_rural = ds.copy()
        ds_urban = ds.copy()
        ds_urban_roi = ds.copy()
    except Exception as e2:
        print(f"Error importing shapefiles: {e2}")
        print("Please check shapefile paths and try again.")
        # Proceed with the full dataset as a fallback
        ds_rural = ds
        ds_urban = ds
        ds_urban_roi = ds

#%% Define seasons and create seasonal masks
# Convert time to pandas datetime
times = pd.to_datetime(ds_rural.time.values)
months = np.array([t.month for t in times])
hours = np.array([t.hour for t in times])

# Create seasonal masks
winter_mask = (months == 12) | (months == 1) | (months == 2)
spring_mask = (months == 3) | (months == 4) | (months == 5)
summer_mask = (months == 6) | (months == 7) | (months == 8)
autumn_mask = (months == 9) | (months == 10) | (months == 11)

seasonal_masks = {
    'Winter': winter_mask,
    'Spring': spring_mask,
    'Summer': summer_mask,
    'Autumn': autumn_mask
}

# Create day/night masks
day_mask = (hours >= 6) & (hours < 18)
night_mask = ~day_mask

# Function to apply lapse rate correction
def apply_lapse_rate_correction(temperature, elevation):
    """Correct temperatures to reference elevation"""
    reference_elevation = float(ds_rural.HGT[0].min().values)
    elevation_diff = elevation - reference_elevation
    correction = LAPSE_RATE * elevation_diff
    return temperature + correction

#%% Create elevation-based rural mask
def create_rural_elevation_mask(ds_urban, ds_rural, max_diff=200):
    """
    Create a mask for rural areas based on elevation similarity to urban areas.
    
    Parameters:
    -----------
    ds_urban : xarray.Dataset
        Urban area dataset
    ds_rural : xarray.Dataset
        Rural area dataset
    max_diff : float, optional
        Maximum elevation difference in meters (default: 200)
        
    Returns:
    --------
    xarray.DataArray
        Boolean mask where True indicates rural points within elevation threshold
    """
    # Calculate mean urban elevation
    urban_mean_elevation = float(ds_urban.HGT[0].mean().values)
    print(f"Average urban elevation: {urban_mean_elevation:.2f} m ASL")
    
    # Get rural elevation
    rural_elevation = ds_rural.HGT[0]
    
    # Calculate absolute difference
    elevation_diff = abs(rural_elevation - urban_mean_elevation)
    
    # Create mask where difference is less than max_diff
    mask = elevation_diff <= max_diff
    
    # Print summary statistics
    total_points = float(mask.size)
    valid_points = float(mask.sum().values)
    percentage = (valid_points / total_points) * 100
    
    print(f"Rural points within {max_diff}m of urban elevation: {int(valid_points)} of {int(total_points)} ({percentage:.1f}%)")
    
    return mask

def calculate_uhi_with_mask(ds_urban_roi, ds_rural, rural_mask=None, apply_correction=True, LAPSE_RATE=0.0065):
    """
    Calculate Urban Heat Island effect using optional elevation-based masking.
    
    Parameters:
    -----------
    ds_urban_roi : xarray.Dataset
        Urban area dataset with ROI
    ds_rural : xarray.Dataset
        Rural area dataset
    rural_mask : xarray.DataArray, optional
        Boolean mask for rural areas based on elevation similarity
    apply_correction : bool, optional
        Whether to apply lapse rate correction
    LAPSE_RATE : float, optional
        Lapse rate in °C/m (default: 0.0065)
        
    Returns:
    --------
    tuple
        (temp_diff, temp_diff_corrected) - Temperature differences
    """
    # If mask is provided, apply it to rural dataset for calculations
    if rural_mask is not None:
        # Calculate mean temperature only from masked points
        # First create a masked version of T2
        rural_t2_masked = ds_rural['T2'].where(rural_mask)
        # Then calculate the mean of non-NaN values
        rural_mean = rural_t2_masked.mean(dim=['south_north', 'west_east'], skipna=True)
    else:
        # Original behavior - use all rural points
        rural_mean = ds_rural['T2'].mean(dim=['south_north', 'west_east'])
    
    # Calculate uncorrected temperature differences
    temp_diff = ds_urban_roi['T2'].copy()
    temp_diff.values = (ds_urban_roi['T2'].values - 
                       rural_mean.values[:, np.newaxis, np.newaxis])
    
    # Apply lapse rate correction if requested
    if apply_correction:
        if rural_mask is not None:
            # Apply mask to rural HGT and T2 data
            rural_hgt_masked = ds_rural.HGT[0].where(rural_mask)
            rural_t2_for_correction = ds_rural['T2'].where(rural_mask)
            
            # Apply lapse rate correction on masked data
            reference_elevation = float(rural_hgt_masked.min(skipna=True).values)
            elevation_diff = rural_hgt_masked - reference_elevation
            correction = LAPSE_RATE * elevation_diff
            rural_temp_corrected = rural_t2_for_correction + correction
            
            # Calculate mean of corrected values (skipna=True for masked points)
            rural_mean_corrected = rural_temp_corrected.mean(
                dim=['south_north', 'west_east'], skipna=True)
        else:
            # Original lapse rate correction
            reference_elevation = float(ds_rural.HGT[0].min().values)
            elevation_diff = ds_rural.HGT[0] - reference_elevation
            correction = LAPSE_RATE * elevation_diff
            rural_temp_corrected = ds_rural['T2'] + correction
            rural_mean_corrected = rural_temp_corrected.mean(
                dim=['south_north', 'west_east'])
        
        temp_diff_corrected = ds_urban_roi['T2'].copy()
        temp_diff_corrected.values = (ds_urban_roi['T2'].values - 
                                    rural_mean_corrected.values[:, np.newaxis, np.newaxis])
    else:
        temp_diff_corrected = temp_diff
    
    # Convert to Celsius if needed (if temperature is in Kelvin)
    for field in [temp_diff, temp_diff_corrected]:
        if field.mean() > 100:  # Simple check for Kelvin values
            field.values = field.values - 273.15
    
    return temp_diff, temp_diff_corrected
#%% Calculate UHI for each season (both uncorrected and corrected)
print("Calculating seasonal UHI metrics...")

# Dictionary to store results
seasonal_results = {}

for season_name, season_mask in seasonal_masks.items():
    print(f"Processing {season_name}...")
    
    # Calculate UHI for whole day
    # Get rural mean temperatures
    rural_mean = ds_rural['T2'].isel(time=season_mask).mean(dim=['south_north', 'west_east'])
    
    # Apply lapse rate correction to rural temperatures
    rural_temp_corrected = apply_lapse_rate_correction(ds_rural['T2'].isel(time=season_mask), 
                                                       ds_rural.HGT[0])
    rural_mean_corrected = rural_temp_corrected.mean(dim=['south_north', 'west_east'])
    
    # Calculate temperature differences
    temp_diff = ds_urban_roi['T2'].isel(time=season_mask).copy()
    temp_diff.values = (ds_urban_roi['T2'].isel(time=season_mask).values - 
                       rural_mean.values[:, np.newaxis, np.newaxis])
    
    temp_diff_corrected = ds_urban_roi['T2'].isel(time=season_mask).copy()
    temp_diff_corrected.values = (ds_urban_roi['T2'].isel(time=season_mask).values - 
                                 rural_mean_corrected.values[:, np.newaxis, np.newaxis])
    
    # Calculate day and night separately
    day_season_mask = season_mask & day_mask
    night_season_mask = season_mask & night_mask
    
    # Day rural means
    rural_mean_day = ds_rural['T2'].isel(time=day_season_mask).mean(dim=['south_north', 'west_east'])
    rural_temp_day_corrected = apply_lapse_rate_correction(ds_rural['T2'].isel(time=day_season_mask), 
                                                           ds_rural.HGT[0])
    rural_mean_day_corrected = rural_temp_day_corrected.mean(dim=['south_north', 'west_east'])
    
    # Night rural means
    rural_mean_night = ds_rural['T2'].isel(time=night_season_mask).mean(dim=['south_north', 'west_east'])
    rural_temp_night_corrected = apply_lapse_rate_correction(ds_rural['T2'].isel(time=night_season_mask), 
                                                             ds_rural.HGT[0])
    rural_mean_night_corrected = rural_temp_night_corrected.mean(dim=['south_north', 'west_east'])
    
    # Day temperature differences
    temp_diff_day = ds_urban_roi['T2'].isel(time=day_season_mask).copy()
    temp_diff_day.values = (ds_urban_roi['T2'].isel(time=day_season_mask).values - 
                            rural_mean_day.values[:, np.newaxis, np.newaxis])
    
    temp_diff_day_corrected = ds_urban_roi['T2'].isel(time=day_season_mask).copy()
    temp_diff_day_corrected.values = (ds_urban_roi['T2'].isel(time=day_season_mask).values - 
                                      rural_mean_day_corrected.values[:, np.newaxis, np.newaxis])
    
    # Night temperature differences
    temp_diff_night = ds_urban_roi['T2'].isel(time=night_season_mask).copy()
    temp_diff_night.values = (ds_urban_roi['T2'].isel(time=night_season_mask).values - 
                              rural_mean_night.values[:, np.newaxis, np.newaxis])
    
    temp_diff_night_corrected = ds_urban_roi['T2'].isel(time=night_season_mask).copy()
    temp_diff_night_corrected.values = (ds_urban_roi['T2'].isel(time=night_season_mask).values - 
                                        rural_mean_night_corrected.values[:, np.newaxis, np.newaxis])
    
    # Calculate temporal averages
    avg_full = temp_diff.mean(dim='time')
    avg_day = temp_diff_day.mean(dim='time')
    avg_night = temp_diff_night.mean(dim='time')
    
    avg_full_corrected = temp_diff_corrected.mean(dim='time')
    avg_day_corrected = temp_diff_day_corrected.mean(dim='time')
    avg_night_corrected = temp_diff_night_corrected.mean(dim='time')
    
    # Convert to Celsius if needed
    for field in [avg_full, avg_day, avg_night, avg_full_corrected, avg_day_corrected, avg_night_corrected]:
        if field.mean() > 100:
            field.values = field.values - 273.15
    
    # Store results
    seasonal_results[season_name] = {
        'uncorrected': {
            'full': avg_full,
            'day': avg_day,
            'night': avg_night,
            'daily_data': temp_diff,
            'day_data': temp_diff_day,
            'night_data': temp_diff_night
        },
        'corrected': {
            'full': avg_full_corrected,
            'day': avg_day_corrected,
            'night': avg_night_corrected,
            'daily_data': temp_diff_corrected,
            'day_data': temp_diff_day_corrected,
            'night_data': temp_diff_night_corrected
        }
    }
#%% PLOTS
def plot_seasonal_uhi_maps(data_type='corrected'):
    """Plot seasonal UHI maps for full day, day, and night periods"""
    
    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(4, 3, figure=fig, hspace=0.001, wspace=0.02)  # Minimal spacing between plots
    
    # Plot settings
    if data_type == 'corrected':
        plot_settings = {
            'cmap': 'RdBu_r',
            'vmin': -1.5,
            'vmax': 1.5,  # Adjust based on your data range
            'cbar_title': 'Urban-Rural Temperature Difference (°C)'
        }
        title_prefix = "Lapse Rate Corrected"
    else:
        plot_settings = {
            'cmap': 'RdBu_r',
            'vmin': -2.5,
            'vmax': 2.5,  # Adjust based on your data range
            'cbar_title': 'Urban-Rural Temperature Difference (°C)'
        }
        title_prefix = "Uncorrected"
    
    # Create axes for all subplots
    axes = {}
    for i, season in enumerate(seasonal_results.keys()):
        for j, period in enumerate(['full', 'day', 'night']):
            axes[(season, period)] = fig.add_subplot(gs[i, j])
    
    # Function to plot a single UHI map
    def plot_single_uhi(ax, data, title):
        # Get the salem map
        smap = ds_urban_roi.salem.get_map()
        
        # Set the temperature difference data
        smap.set_data(data)
        smap.set_cmap(plot_settings['cmap'])
        smap.set_vmin(plot_settings['vmin'])
        smap.set_vmax(plot_settings['vmax'])
        
        # Plot temperature differences
        smap.visualize(ax=ax, title=title, addcbar=False)
        
        # Add terrain contours
        terrain = ds_rural.HGT[0].values
        xx, yy = smap.grid.transform(
            ds_rural.west_east.values,
            ds_rural.south_north.values,
            crs=ds_rural.salem.grid.proj
        )
        xx, yy = np.meshgrid(xx, yy)
        ax.contour(xx, yy, terrain, levels=10, colors='black', linewidths=0.5, alpha=0.7)
        
        # Add urban boundary
        try:
            urban_shp = gpd.read_file(URBAN_SHAPE)
            urban_shp.boundary.plot(ax=ax, color='red', linewidth=2)
        except:
            pass  # Skip if shapefile cannot be loaded
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.2)
        
        return smap
    
    # Plot all seasons and periods
    for season in seasonal_results.keys():
        for period in ['full', 'day', 'night']:
            period_label = {'full': '24h', 'day': 'Day (6-18h)', 'night': 'Night (18-6h)'}[period]
            data = seasonal_results[season][data_type][period]
            title = f"{season} - {period_label}\n{title_prefix}"
            plot_single_uhi(axes[(season, period)], data, title)
    
    # Add a single colorbar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    norm = plt.Normalize(vmin=plot_settings['vmin'], vmax=plot_settings['vmax'])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plot_settings['cmap'])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(plot_settings['cbar_title'], fontsize=14)
    
    # Add title for the entire figure
    fig.suptitle(f"Seasonal Urban Heat Island Effect ({title_prefix})", fontsize=20, y=0.98)
    
    # Don't use tight_layout as we've already set spacing in GridSpec
    # Instead use subplots_adjust for fine-tuning if needed
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9)
    # plt.tight_layout(rect=[0, 0.07, 1, 0.96])

    # Save figure
    plt.savefig(f'seasonal_uhi_maps_{data_type}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate both types of maps
# plot_seasonal_uhi_maps(data_type='uncorrected')
plot_seasonal_uhi_maps(data_type='corrected')
#%%  seasonal UHI maps 
print("Generating seasonal UHI maps...")

def plot_seasonal_uhi_maps(data_type='corrected'):
    """Plot seasonal UHI maps for full day, day, and night periods"""
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig)
    
    # Plot settings
    if data_type == 'corrected':
        plot_settings = {
            'cmap': 'RdBu_r',
            'vmin': -1.5,
            'vmax': 1.5,  # Adjust based on your data range
            'cbar_title': 'Urban-Rural Temperature Difference (°C)'
        }
        title_prefix = "Lapse Rate Corrected"
    else:
        plot_settings = {
            'cmap': 'RdBu_r',
            'vmin': -2.5,
            'vmax': 2.5,  # Adjust based on your data range
            'cbar_title': 'Urban-Rural Temperature Difference (°C)'
        }
        title_prefix = "Uncorrected"
    
    # Create axes for all subplots
    axes = {}
    for i, season in enumerate(seasonal_results.keys()):
        for j, period in enumerate(['full', 'day', 'night']):
            axes[(season, period)] = fig.add_subplot(gs[i, j])
    
    # Function to plot a single UHI map
    def plot_single_uhi(ax, data, title):
        # Get the salem map
        smap = ds_urban_roi.salem.get_map()
        
        # Set the temperature difference data
        smap.set_data(data)
        smap.set_cmap(plot_settings['cmap'])
        smap.set_vmin(plot_settings['vmin'])
        smap.set_vmax(plot_settings['vmax'])
        
        # Plot temperature differences
        smap.visualize(ax=ax, title=title, addcbar=False)
        
        # Add terrain contours
        terrain = ds_rural.HGT[0].values
        xx, yy = smap.grid.transform(
            ds_rural.west_east.values,
            ds_rural.south_north.values,
            crs=ds_rural.salem.grid.proj
        )
        xx, yy = np.meshgrid(xx, yy)
        ax.contour(xx, yy, terrain, levels=10, colors='black', linewidths=0.5, alpha=0.7)
        
        # Add urban boundary
        try:
            urban_shp = gpd.read_file(URBAN_SHAPE)
            urban_shp.boundary.plot(ax=ax, color='red', linewidth=2)
        except:
            pass  # Skip if shapefile cannot be loaded
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.2)
        
        return smap
    
    # Plot all seasons and periods
    for season in seasonal_results.keys():
        for period in ['full', 'day', 'night']:
            period_label = {'full': '24h', 'day': 'Day (6-18h)', 'night': 'Night (18-6h)'}[period]
            data = seasonal_results[season][data_type][period]
            title = f"{season} - {period_label}"
            plot_single_uhi(axes[(season, period)], data, title)
    
    # Add a single colorbar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    norm = plt.Normalize(vmin=plot_settings['vmin'], vmax=plot_settings['vmax'])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plot_settings['cmap'])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(plot_settings['cbar_title'], fontsize=14)
    
    # Add title for the entire figure
    fig.suptitle(f"Seasonal Urban Heat Island Effect ({title_prefix})", fontsize=20, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    
    # Save figure
    plt.savefig(f'seasonal_uhi_maps_{data_type}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate both types of maps
# plot_seasonal_uhi_maps(data_type='uncorrected')
plot_seasonal_uhi_maps(data_type='corrected')

#%% Seasonal UHI maps 
print("Generating seasonal UHI maps...")

def plot_seasonal_uhi_maps(data_type='corrected'):
    """Plot seasonal UHI maps for full day, day, and night periods"""
    
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import numpy as np
    import geopandas as gpd
    
    # -------------------------------
    # Global font settings (publication)
    # -------------------------------
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 16,
        'axes.labelsize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14
    })
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig)
    gs.update(wspace=0.005, hspace=0.15)
    # -------------------------------
    # Plot settings
    # -------------------------------
    if data_type == 'corrected':
        plot_settings = {
            'cmap': 'RdBu_r',
            'vmin': -1.5,
            'vmax': 1.5,
            'cbar_title': 'UHI intensity [°C]'
        }
    else:
        plot_settings = {
            'cmap': 'RdBu_r',
            'vmin': -2.5,
            'vmax': 2.5,
            'cbar_title': 'Urban–Rural Temperature Difference (°C)'
        }
    
    # -------------------------------
    # Create axes
    # -------------------------------
    axes = {}
    seasons = list(seasonal_results.keys())
    
    for i, season in enumerate(seasons):
        for j, period in enumerate(['full', 'day', 'night']):
            axes[(season, period)] = fig.add_subplot(gs[i, j])
    
    # -------------------------------
    # Plot function
    # -------------------------------
    def plot_single_uhi(ax, data):
        smap = ds_urban_roi.salem.get_map()
        
        smap.set_data(data)
        smap.set_cmap(plot_settings['cmap'])
        smap.set_vmin(plot_settings['vmin'])
        smap.set_vmax(plot_settings['vmax'])
        
        smap.visualize(ax=ax, title="", addcbar=False)
        
        # Terrain contours
        terrain = ds_rural.HGT[0].values
        xx, yy = smap.grid.transform(
            ds_rural.west_east.values,
            ds_rural.south_north.values,
            crs=ds_rural.salem.grid.proj
        )
        xx, yy = np.meshgrid(xx, yy)
        contours = ax.contour(
            xx, yy, terrain,
            levels=10,
            colors='black',
            linewidths=0.5,
            alpha=0.7
        )
        
        # Add altitude labels
        ax.clabel(
            contours,
            inline=True,
            fontsize=10,
            fmt='%d m'   # adds unit (meters)
        )        
                # Urban boundary
        try:
            urban_shp = gpd.read_file(URBAN_SHAPE)
            urban_shp.boundary.plot(ax=ax, color='red', linewidth=2)
        except:
            pass
        
        # Light grid
        ax.grid(True, linestyle='--', alpha=0.2)
    
    # -------------------------------
    # Plot all panels
    # -------------------------------
    for season in seasons:
        for period in ['full', 'day', 'night']:
            data = seasonal_results[season][data_type][period]
            plot_single_uhi(axes[(season, period)], data)
    
    # -------------------------------
    # Add season labels (left side only)
    # -------------------------------
    for i, season in enumerate(seasons):
        axes[(season, 'full')].set_ylabel(
            season, fontsize=18, weight='bold'
        )
    
    # -------------------------------
    # Add time labels (bottom only)
    # -------------------------------
    period_labels = {
        'full': '24h',
        'day': 'Day (6–18h)',
        'night': 'Night (18–6h)'
    }
    
    for j, period in enumerate(['full', 'day', 'night']):
        axes[(seasons[-1], period)].set_xlabel(
            period_labels[period], fontsize=18, weight='bold'
        )
    
    # -------------------------------
    # Shared colorbar
    # -------------------------------
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    norm = plt.Normalize(vmin=plot_settings['vmin'], vmax=plot_settings['vmax'])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plot_settings['cmap'])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    
    cbar.set_label(plot_settings['cbar_title'], fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    
    # -------------------------------
    # Main title
    # -------------------------------
    # fig.suptitle(
    #     "Seasonal Urban Heat Island Effect",
    #     fontsize=22,
    #     y=0.98
    # )
    
    # -------------------------------
    # Layout & save
    # -------------------------------
    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    
    plt.savefig(
        f'seasonal_uhi_maps_{data_type}.png',
        dpi=300,
        bbox_inches='tight'
    )
    
    plt.show()


# Run
plot_seasonal_uhi_maps(data_type='corrected')

#%% seasonal UHI maps with elevation mask
print("Generating seasonal UHI maps with elevation masking...")

# Create elevation masks for each seasonal dataset
seasonal_elevation_masks = {}
for season_name in seasonal_results.keys():
    print(f"\nCreating elevation mask for {season_name} season...")
    
    # Use full-day data for mask creation
    if 'uncorrected' in seasonal_results[season_name] and 'full' in seasonal_results[season_name]['uncorrected']:
        # We need access to the actual datasets to create the mask
        # This assumes the original ds_urban and ds_rural are still available
        elevation_mask = create_rural_elevation_mask(ds_urban, ds_rural, max_diff=200)
        
        # Store the mask for this season
        seasonal_elevation_masks[season_name] = elevation_mask
        
        print(f"Created elevation mask for {season_name}")
    else:
        print(f"Cannot create mask for {season_name} - missing data")

# Dictionary to store elevation-masked results
elevation_masked_results = {}

# Calculate elevation-masked UHI for each season
for season_name, season_mask in seasonal_masks.items():
    print(f"\nCalculating elevation-masked UHI for {season_name}...")
    
    # Skip if no mask was created for this season
    if season_name not in seasonal_elevation_masks:
        print(f"Skipping {season_name} - no elevation mask available")
        continue
    
    # Get the elevation mask for this season
    elevation_mask = seasonal_elevation_masks[season_name]
    
    # Initialize results dictionary for this season
    elevation_masked_results[season_name] = {'uncorrected': {}, 'corrected': {}}
    
    # Calculate UHI for whole day
    for period in ['full', 'day', 'night']:
        # Determine the appropriate time mask
        if period == 'day':
            time_mask = season_mask & day_mask
        elif period == 'night':
            time_mask = season_mask & night_mask
        else:  # full
            time_mask = season_mask
        
        # Calculate UHI with elevation mask
        temp_diff, temp_diff_corrected = calculate_uhi_with_mask(
            ds_urban_roi,
            ds_rural,
            rural_mask=elevation_mask,
            apply_correction=True,
            LAPSE_RATE=LAPSE_RATE
        )
        
        # Store results
        elevation_masked_results[season_name]['uncorrected'][period] = temp_diff
        elevation_masked_results[season_name]['corrected'][period] = temp_diff_corrected
        
        # Print some statistics
        print(f"  {season_name} - {period}:")
        print(f"    Mean uncorrected UHI: {float(temp_diff.mean().values):.2f}°C")
        print(f"    Mean corrected UHI: {float(temp_diff_corrected.mean().values):.2f}°C")

def plot_seasonal_uhi_maps_with_mask(data_type='corrected'):
    """Plot seasonal UHI maps for full day, day, and night periods with elevation masking"""
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig)
    
    # Plot settings
    if data_type == 'corrected':
        plot_settings = {
            'cmap': 'RdBu_r',
            'vmin': -1.5,
            'vmax': 1.5,  # Adjust based on your data range
            'cbar_title': 'Urban-Rural Temperature Difference (°C)'
        }
        title_prefix = "Lapse Rate Corrected with Elevation Masking"
    else:
        plot_settings = {
            'cmap': 'RdBu_r',
            'vmin': -1,
            'vmax': 4.5,  # Adjust based on your data range
            'cbar_title': 'Urban-Rural Temperature Difference (°C)'
        }
        title_prefix = "Uncorrected with Elevation Masking"
    
    # Create axes for all subplots
    axes = {}
    for i, season in enumerate(elevation_masked_results.keys()):
        for j, period in enumerate(['full', 'day', 'night']):
            axes[(season, period)] = fig.add_subplot(gs[i, j])
    
    # Function to plot a single UHI map
    def plot_single_uhi(ax, data, title):
        # Get the salem map
        smap = ds_urban_roi.salem.get_map()
        
        # Average over time dimension to get 2D spatial data
        data_2d = data.mean(dim='time')
        
        # Set the temperature difference data
        smap.set_data(data_2d)
        smap.set_cmap(plot_settings['cmap'])
        smap.set_vmin(plot_settings['vmin'])
        smap.set_vmax(plot_settings['vmax'])
        
        # Plot temperature differences
        smap.visualize(ax=ax, title=title, addcbar=False)
        
        # Add terrain contours
        terrain = ds_rural.HGT[0].values
        xx, yy = smap.grid.transform(
            ds_rural.west_east.values,
            ds_rural.south_north.values,
            crs=ds_rural.salem.grid.proj
        )
        xx, yy = np.meshgrid(xx, yy)
        ax.contour(xx, yy, terrain, levels=10, colors='black', linewidths=0.5, alpha=0.7)
        
        # Add urban boundary
        try:
            urban_shp = gpd.read_file(URBAN_SHAPE)
            urban_shp.boundary.plot(ax=ax, color='red', linewidth=2)
        except:
            pass  # Skip if shapefile cannot be loaded
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        return smap
    
    # Plot all seasons and periods
    for season in elevation_masked_results.keys():
        for period in ['full', 'day', 'night']:
            period_label = {'full': '24h', 'day': 'Day (6-18h)', 'night': 'Night (18-6h)'}[period]
            data = elevation_masked_results[season][data_type][period]
            title = f"{season} - {period_label}\n{title_prefix}"
            plot_single_uhi(axes[(season, period)], data, title)
    
    # Add a single colorbar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    norm = plt.Normalize(vmin=plot_settings['vmin'], vmax=plot_settings['vmax'])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plot_settings['cmap'])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(plot_settings['cbar_title'], fontsize=14)
    
    # Add title for the entire figure
    fig.suptitle(f"Seasonal Urban Heat Island Effect ({title_prefix})", fontsize=20, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    
    # Save figure
    plt.savefig(f'seasonal_uhi_maps_elevation_masked_{data_type}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate both types of maps with elevation masking
plot_seasonal_uhi_maps_with_mask(data_type='uncorrected')
plot_seasonal_uhi_maps_with_mask(data_type='corrected')

print("\nAnalysis complete!")

#%%end cell