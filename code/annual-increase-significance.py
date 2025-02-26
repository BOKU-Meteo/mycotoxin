import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import ttest_ind
from matplotlib.colors import ListedColormap, BoundaryNorm

# File path
path_to_tas = './tas/ERA5-Land_tas_daily_{year}.nc'

# User input for time periods
print("Enter the first time period:")
start_year_1 = int(input("Start year (e.g., 1980): "))
end_year_1 = int(input("End year (e.g., 1990): "))

print("Enter the second time period:")
start_year_2 = int(input("Start year (e.g., 2000): "))
end_year_2 = int(input("End year (e.g., 2010): "))

# Bounding box for Europe
europe_lon_min, europe_lon_max = -25, 45
europe_lat_min, europe_lat_max = 35, 70

# Function to calculate High risk days (ARI>0.50) for all years in the given period
def calculate_ARI_days(start_year, end_year):
    ARI_days_list = []  # Store ARI days for each year

    for year in range(start_year, end_year + 1):
        print(f"Processing year {year}...")

        # Load temperature data
        tas_ds = xr.open_dataset(path_to_tas.format(year=year))
        tas = tas_ds['tas'] - 273.15  # Convert from Kelvin to Celsius

        # Mask non-European regions
        tas = tas.where((tas['longitude'] >= europe_lon_min) & (tas['longitude'] <= europe_lon_max) &
                        (tas['latitude'] >= europe_lat_min) & (tas['latitude'] <= europe_lat_max), drop=True)

        # Compute ARI days per year
        ARI_days_year = xr.zeros_like(tas.isel(time=0)) #Initializing a 2D array (latitude x longitude) filled with zeros
                                                         #to store the count of days where ARI > 0.50 for each grid cell

        for day in range(len(tas['time'])):
            tas_day = tas.isel(time=day)

            # Compute SPORULATION
            Teq_spor = (tas_day - 5) / (45 - 5)
            SPOR = xr.where((Teq_spor > 0) & (Teq_spor < 1),
                            5.28200632 * Teq_spor**2.05221609 * (1 - Teq_spor)**0.97677772, 0)

            # Compute GROWTH
            Teq_grow = (tas_day - 5) / (48 - 5)
            GROW_T = xr.where((Teq_grow > 0) & (Teq_grow < 1),
                              (5.98426163 * Teq_grow**1.70408086 * (1 - Teq_grow))**1.42921131, 0)

            # Compute AFLATOXIN PRODUCTION
            Teq_afla = (tas_day - 10) / (47 - 10)
            AFLA = xr.where((Teq_afla > 0) & (Teq_afla < 1),
                            (4.84 * Teq_afla**1.32 * (1 - Teq_afla))**5.59, 0)

            # Compute daily ARI
            ARI_day = SPOR * GROW_T * AFLA

            # Count days where ARI > 0.50
            ARI_days_year += (ARI_day > 0.50)

        ARI_days_list.append(ARI_days_year)

    return xr.concat(ARI_days_list, dim='time')

# Compute ARI days for both periods
ARI_days_1 = calculate_ARI_days(start_year_1, end_year_1)
ARI_days_2 = calculate_ARI_days(start_year_2, end_year_2)

# Convert to numpy arrays
data1 = ARI_days_1.values  # Shape: (years, lat, lon)
data2 = ARI_days_2.values  # Shape: (years, lat, lon)

# Perform pixel-wise t-test
p_values = np.full(data1.shape[1:], np.nan)  # Shape (lat, lon)

for i in range(data1.shape[1]):  # Latitude index
    for j in range(data1.shape[2]):  # Longitude index
        d1 = data1[:, i, j].flatten()  # Extract time-series
        d2 = data2[:, i, j].flatten()

        # Skip locations with missing data
        if np.isnan(d1).all() or np.isnan(d2).all():
            continue

        # Perform t-test
        try:
            _, p_values[i, j] = ttest_ind(d1, d2, equal_var=False, nan_policy='omit')
        except Exception as e:
            print(f"T-test failed at ({i},{j}): {e}")

# Mask non-significant areas
significance_mask = np.where((p_values < 0.05) & (~np.isnan(p_values)), 1, np.nan)

# **Plot the significance heatmap**
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

# Define colormap for significance (highlight in red)
colors = ['white', 'red']
cmap = ListedColormap(colors)
norm = BoundaryNorm([0, 1], ncolors=cmap.N, clip=True)

mesh = ax.pcolormesh(ARI_days_1['longitude'], ARI_days_1['latitude'], significance_mask, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

# Add features (borders, coastlines)
ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='black')

# Set extent for Europe
ax.set_extent([-25, 45, 35, 70], crs=ccrs.PlateCarree())

# Add colorbar
cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.05)
cbar.set_label('Significant Increase in High-Risk Days (p < 0.05)')

plt.title(f'Statistically Significant Increase in High Risk Days ({start_year_1}-{end_year_1} vs {start_year_2}-{end_year_2})', fontsize=14)

plt.tight_layout()
plt.show()
