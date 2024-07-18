import os
import conda
import pandas as pd
from toolbox import EasyMap, pc
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
from datetime import datetime, timedelta
import geopandas as gpd
import matplotlib.colors as mcolors
import shutil
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import time
from herbie import Herbie
mpl.use('Agg')

###May not be necesary
def setup_proj_lib():
    conda_file_dir = conda.__file__
    conda_dir = conda_file_dir.split('lib')[0]
    proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
    os.environ["PROJ_LIB"] = proj_lib

###Read in list of airports
def read_icao_data(filepath):
    return pd.read_csv(filepath)
  
###Get current UTC time
def get_adjusted_utc_time(hours):
    return datetime.utcnow() - timedelta(hours=hours)
  
###Format the time
def format_time(dt, fmt='%Y-%m-%d-%H'):
    return dt.strftime(fmt)

###NWS radar colormap
def radar_colormap():
    nws_reflectivity_colors = [
        "#646464", "#04e9e7", "#019ff4", "#0300f4", "#02fd02", "#01c501",
        "#008e00", "#fdf802", "#e5bc00", "#fd9500", "#fd0000", "#d40000",
        "#bc0000", "#f800fd", "#9854c6", "#fdfdfd"
    ]
    return mpl.colors.ListedColormap(nws_reflectivity_colors)

###Plot the airport points and label them
def plot_airports(ax, icao_data, extent):
    for i, row in icao_data.iterrows():
        lon = float(row['Longitude'])
        lat = float(row['Latitude'])
        if extent[0] <= lon <= extent[1] and extent[2] <= lat <= extent[3]:
            ax.plot(lon, lat, 'ro', markersize=5, transform=ccrs.PlateCarree())
            ax.text(lon, lat, row['ICAO'], transform=ccrs.PlateCarree(), fontsize=8)

###Load the ARTCC data from the shapefile
def load_shapefile(filepath):
    data = gpd.read_file(filepath)
    return data[data.TYPE_CODE == 'ARTCC']

###Grab latest HRRR data using Herbie
def grabdata(cycle, fxx):
    return Herbie(cycle, searchString=":REFD:1000 m:|:RETOP:cloud top:|:HGT:cloud ceiling:|:UGRD:10m_above_ground:|:VGRD:10m_above_ground:",
                  save_dir='/root/hrrr_grib', model="hrrr", product="sfc", freq="1H", member=1, fxx=fxx, verbose=True, overwrite=False)

###Fetch and load the data into xarray
def fetch_and_prepare_data(adjusted_formatted_time, fxx, max_retries=50, retry_delay=60):
    attempt = 0
    while attempt < max_retries:
        try:
            H = grabdata(adjusted_formatted_time, fxx)
            ref = H.xarray("REFD:1000 m", verbose=True, remove_grib=False)
            et = H.xarray("RETOP:cloud top", verbose=True, remove_grib=False)
            ceil = H.xarray("HGT:cloud ceiling", verbose=True, remove_grib=False)
            #u10 = H.xarray("UGRD:10m_above_ground", verbose=True, remove_grib=False)

            # If data is successfully fetched, return it
            return {
                "refdata": ref.refd,   
                "etdata": et.unknown,
                "ceildata": ceil.gh,
                #"u10data": u10.ugrd,
                "ref": ref,
                "et": et,
                "ceil": ceil,
             #   "u10": u10
            }
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error fetching data for forecast hour {fxx}: {e}")
            attempt += 1
            if attempt < max_retries:
                print(f"Waiting for {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)  # Wait for retry_delay seconds before next attempt
            else:
                print(f"All attempts failed for forecast hour {fxx}. Moving on to next hour.")
                return None

###Function to plot for each region and forecast hour
def plot_for_extent_and_hour(fxx, extent_name, extent, forecast_data, artcc, icao_data, adjusted_formatted_time):
    print(f"Plotting for forecast hour {fxx} and extent {extent_name}")

    extent_dir_map = {
        "1km_ref": "1km_ref",
        "echo_tops": "echo_tops",
        "ceiling": "ceiling",
        "uv10m": "uv10m"
    }

    def save_and_move_plot(filename, extent_dir, adjusted_formatted_time, fxx):
        output_path = f'{extent_dir}/new/{filename}_{adjusted_formatted_time}_{str(fxx).zfill(2)}.png'
        final_path = f'{extent_dir}/{filename}_{adjusted_formatted_time}_{str(fxx).zfill(2)}.png'
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        shutil.move(output_path, final_path)
        print(f"Plot saved and moved to: {final_path}")

    # Plot 1km_ref
    print(f"Creating plot for 1km_ref at forecast hour {fxx} for extent {extent_name}")
    ax1 = EasyMap("50m", crs=forecast_data["ref"].herbie.crs, figsize=[16, 12], dpi=200).STATES().ax
    refdata = forecast_data["refdata"]
    cmapref = radar_colormap()
    cmapref.set_under('white')
    p1 = ax1.pcolormesh(refdata.longitude, refdata.latitude, refdata, transform=ccrs.PlateCarree(), cmap=cmapref, norm=mpl.colors.Normalize(vmin=0, vmax=80))
    ax1.add_geometries(artcc.geometry, crs=ccrs.PlateCarree(), linewidth=0.50, facecolor='none', edgecolor='gray')
    plot_airports(ax1, icao_data, extent)
    ax1.set_extent(extent)
    extent_dir1 = f"{extent_dir_map['1km_ref']}/{extent_name}"
    plt.colorbar(p1, ax=ax1, orientation="horizontal", pad=0.01, shrink=0.6)
    ax1.set_title(f"HRRR\nInit: {refdata.time.dt.strftime('%H:%M UTC %d %b %Y').item()}  Forecast hour: {str(fxx)}   Valid: {refdata.valid_time.dt.strftime('%H:%M UTC %d %b %Y').item()}", loc="left")
    ax1.set_title("1km Reflectivity (dBZ)", loc="right")
    save_and_move_plot("1km_ref", extent_dir1, adjusted_formatted_time, fxx)
    plt.close(ax1.figure)
    print(f"Plot for 1km_ref at forecast hour {fxx} for extent {extent_name} created successfully")

    # Plot echo_tops
    print(f"Creating plot for echo_tops at forecast hour {fxx} for extent {extent_name}")
    ax2 = EasyMap("50m", crs=forecast_data["et"].herbie.crs, figsize=[16, 12], dpi=200).STATES().ax
    etdata = forecast_data["etdata"]
    cmapet = plt.get_cmap('gist_ncar')
    cmapet.set_under('white')
    p2 = ax2.pcolormesh(etdata.longitude, etdata.latitude, (etdata*3.28084)/1000, transform=ccrs.PlateCarree(), cmap=cmapet, vmin=20, vmax=60)
    ax2.add_geometries(artcc.geometry, crs=ccrs.PlateCarree(), linewidth=0.50, facecolor='none', edgecolor='gray')
    plot_airports(ax2, icao_data, extent)
    ax2.set_extent(extent)
    extent_dir2 = f"{extent_dir_map['echo_tops']}/{extent_name}"
    plt.colorbar(p2, ax=ax2, orientation="horizontal", pad=0.01, shrink=0.6)
    ax2.set_title(f"HRRR\nInit: {etdata.time.dt.strftime('%H:%M UTC %d %b %Y').item()}  Forecast hour: {str(fxx)}   Valid: {etdata.valid_time.dt.strftime('%H:%M UTC %d %b %Y').item()}", loc="left")
    ax2.set_title("Echo Tops (kft)", loc="right")
    save_and_move_plot("echo_tops", extent_dir2, adjusted_formatted_time, fxx)
    plt.close(ax2.figure)
    print(f"Plot for echo_tops at forecast hour {fxx} for extent {extent_name} created successfully")

    # Plot ceiling
    print(f"Creating plot for ceiling at forecast hour {fxx} for extent {extent_name}")
    ax3 = EasyMap("50m", crs=forecast_data["ceil"].herbie.crs, figsize=[16, 12], dpi=200).STATES().ax
    ceildata = forecast_data["ceildata"]
    thresholds = [100,200, 400, 600, 800, 1000, 1200, 1500, 2000, 2500, 3000, 4000,5000, 6000, 7000, 8000, 10000, 15000, 20000, 35000, 50000]
    norm = mcolors.BoundaryNorm(boundaries=thresholds, ncolors=cmapet.N)
    p3 = ax3.pcolormesh(ceildata.longitude, ceildata.latitude, ceildata*3.28084, transform=ccrs.PlateCarree(), cmap=plt.get_cmap('gist_ncar'), norm=norm)
    ax3.add_geometries(artcc.geometry, crs=ccrs.PlateCarree(), linewidth=0.50, facecolor='none', edgecolor='gray')
    plot_airports(ax3, icao_data, extent)
    ax3.set_extent(extent)
    extent_dir3 = f"{extent_dir_map['ceiling']}/{extent_name}"
    ax3.set_title(f"HRRR\nInit: {ceildata.time.dt.strftime('%H:%M UTC %d %b %Y').item()}  Forecast hour: {str(fxx)}   Valid: {ceildata.valid_time.dt.strftime('%H:%M UTC %d %b %Y').item()}", loc="left")
    ax3.set_title("Ceiling (ft)", loc="right")
    cb=plt.colorbar(p3,ax=ax3,orientation="horizontal",pad=0.01,shrink=0.6,)
    cb.ax.set_xticks(thresholds)
    cb.ax.set_xticklabels([f"{int(x)}" for x in thresholds])
    cb.ax.tick_params(labelsize=6, which="both")
    save_and_move_plot("ceiling", extent_dir3, adjusted_formatted_time, fxx)
    plt.close(ax3.figure)
    print(f"Plot for ceiling at forecast hour {fxx} for extent {extent_name} created successfully")

def determine_forecast_hours(adjusted_utc_time):
    hour = int(adjusted_utc_time.strftime('%H'))
    if hour in [0, 6, 12, 18]:
        return 49
    else:
        return 19

def main():
    setup_proj_lib()
    icao_data = read_icao_data('airports.csv')
    current_utc_time = get_adjusted_utc_time(1)
    adjusted_formatted_time = format_time(current_utc_time)
    artcc = load_shapefile('Airspace_Boundary.shp')
    
    fhours = determine_forecast_hours(current_utc_time)
    forecast_hours = range(0, fhours, 1)
    ####Southeast####
    se_extent = [-100, -72, 23, 40]
    ####Northeast####
    ne_extent = [-95, -70, 35, 48]
    ####Southwest####
    sw_extent = [-126.5, -90, 25, 40]
    ####Northwest####
    nw_extent = [-129, -95, 35, 51]
    ####N90####
    n90_extent = [-77, -72, 39, 42]
    extents = [se_extent, ne_extent, sw_extent, nw_extent, n90_extent]

    extent_names = ["Southeast", "Northeast", "Southwest", "Northwest", "N90"]

    for fxx in forecast_hours:
        forecast_data = fetch_and_prepare_data(adjusted_formatted_time, fxx)
        if not forecast_data:
            print(f"No data available for forecast hour {fxx} even after retries.")
            continue

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    plot_for_extent_and_hour,
                    fxx,
                    extent_name,
                    extent,
                    forecast_data,
                    artcc,
                    icao_data,
                    adjusted_formatted_time
                )
                for extent_name, extent in zip(extent_names, extents)
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                    print("Plotting task completed successfully.")
                except Exception as e:
                    print(f"An error occurred during plotting: {e}")

if __name__ == "__main__":
    main()

