#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BelSAR Validation Module
========================

This module contains functionality for validating biophysical parameter retrieval models
(e.g., Transformer-VAE) using the BelSAR (Belgian Synthetic Aperture Radar) 
agricultural database.

The BelSAR dataset contains field measurements of biophysical parameters (LAI, biomass, etc.)
for wheat and maize fields in Belgium, along with corresponding Sentinel-2 satellite imagery.
"""

import os
import pandas as pd
import geopandas as gpd
import rasterio as rio
import numpy as np
import socket
import argparse
import zipfile
import shutil
from tqdm import tqdm
from rasterio.mask import mask
from validation_utils import read_data_from_theia
from utils.image_utils import tensor_to_raster, get_encoded_image_from_batch
from datetime import datetime
import torch

from sensorsio import utils
import matplotlib.pyplot as plt
from prosailvae.ProsailSimus import BANDS

# List of Sentinel-2 filenames for validation
BELSAR_FILENAMES = [
    # "2A_20180518_both_BelSAR_agriculture_database", 
    "2A_20180528_both_BelSAR_agriculture_database",
    "2A_20180620_both_BelSAR_agriculture_database", 
    # "2B_20180715_both_BelSAR_agriculture_database",
    # "2B_20180804_both_BelSAR_agriculture_database"
    "2A_20180727_both_BelSAR_agriculture_database"
]

closest_filename_dict =  {
    # "2018-05-17" : "2A_20180508_both_BelSAR_agriculture_database",
    # '2018-05-18' : "2A_20180518_both_BelSAR_agriculture_database",
    '2018-05-31' : "2A_20180528_both_BelSAR_agriculture_database",
    '2018-06-01' : "2A_20180528_both_BelSAR_agriculture_database",
    '2018-06-05' : "2A_20180528_both_BelSAR_agriculture_database",
    '2018-06-21' : "2A_20180620_both_BelSAR_agriculture_database",
    '2018-06-22' : "2A_20180620_both_BelSAR_agriculture_database",
    # '2018-07-19' : "2B_20180715_both_BelSAR_agriculture_database",
    '2018-08-02' : "2A_20180727_both_BelSAR_agriculture_database",
    }

def plot_belsar_site(data_dir, filename):
    """
    Plot the BelSAR site with field boundaries overlaid on the Sentinel-2 image.
    
    This function creates a visualization of the agricultural fields in the BelSAR dataset,
    showing field boundaries for both wheat and maize fields overlaid on a Sentinel-2 RGB
    composite image. Each field is labeled with its ID.
    
    Parameters:
    -----------
    data_dir : str
        Path to the directory containing the preprocessed BelSAR data.
    filename : str
        Base filename of the preprocessed BelSAR data (without extension).
        
    Returns:
    --------
    None
        The plot is saved as a PNG file in the data directory.
    """
    # Load the preprocessed data
    df, s2_r, s2_a, mask, xcoords, ycoords, crs = load_belsar_validation_data(data_dir, filename)

    # Get field geometries for wheat and maize fields
    maize_sites = get_sites_geometry(data_dir, crs, crop="maize")
    wheat_sites = get_sites_geometry(data_dir, crs, crop="wheat")
    
    # Prepare mask for visualization
    mask[mask==0.] = np.nan
    
    # Create the figure and plot the RGB composite
    fig, ax = plt.subplots(dpi=200)
    visu, _, _ = utils.rgb_render(s2_r)
    ax.imshow(visu, extent = [xcoords[0], xcoords[-1], ycoords[-1], ycoords[0]])
    ax.imshow(mask.squeeze(), extent = [xcoords[0], xcoords[-1], ycoords[-1], ycoords[0]], cmap='YlOrRd')
    
    # Plot maize field boundaries in blue
    for i in range(len(maize_sites)):
        contour = maize_sites["geometry"].iloc[i].exterior.xy
        ax.plot(contour[0], contour[1], "blue", linewidth=0.5)
    
    # Plot wheat field boundaries in red
    for i in range(len(wheat_sites)):
        contour = wheat_sites["geometry"].iloc[i].exterior.xy
        ax.plot(contour[0], contour[1], "red", linewidth=0.5)
    
    # Add field labels for wheat fields
    for xi, yi, text in zip(wheat_sites.centroid.x, wheat_sites.centroid.y, wheat_sites["Name"]):
        ax.annotate(text,
                xy=(xi, yi), xycoords='data',
                xytext=(1.5, 1.5), textcoords='offset points',
                color='red')
    
    # Add field labels for maize fields
    for xi, yi, text in zip(maize_sites.centroid.x, maize_sites.centroid.y, maize_sites["Name"]):
        ax.annotate(text,
                xy=(xi, yi), xycoords='data',
                xytext=(1.5, 1.5), textcoords='offset points',
                color='blue')
    
    # Save the figure
    fig.savefig(os.path.join(data_dir, filename + "_mask.png"))



def get_all_belsar_predictions(belsar_data_dir, belsar_pred_dir, file_suffix, NO_DATA=-10000, 
                               bands_idx=torch.arange(10)):
    """
    Retrieve and compile predictions for all BelSAR fields across all available dates.
    
    This function processes all dates in the closest_filename_dict, loads the corresponding
    validation data and predictions, and compiles metrics into a single DataFrame.
    
    Parameters:
    -----------
    belsar_data_dir : str
        Path to the directory containing the preprocessed BelSAR data.
    belsar_pred_dir : str
        Path to the directory containing the prediction files.
    file_suffix : str
        Suffix to add to the prediction filenames.
    NO_DATA : int, default=-10000
        Value used to represent no data in the prediction rasters.
    bands_idx : torch.Tensor, default=torch.arange(10)
        Indices of the Sentinel-2 bands to use.
        
    Returns:
    --------
    metrics : pandas.DataFrame
        DataFrame containing compiled metrics for all fields and dates.
    """
    metrics = pd.DataFrame()
    
    # Process each date and its corresponding Sentinel-2 image
    for date, filename in closest_filename_dict.items():
        # Load validation data for this date/image
        validation_df, _, _, _, _, _, crs = load_belsar_validation_data(belsar_data_dir, filename)
        
        # Get unique field IDs in the validation dataset
        ids = pd.unique(validation_df["Field ID"]).tolist()
        
        # Get field geometries and filter to only include fields in the validation dataset
        sites_geometry = get_sites_geometry(belsar_data_dir, crs)
        sites_geometry = sites_geometry[sites_geometry['Name'].apply(lambda x: x in ids)]
        sites_geometry.reset_index(inplace=True, drop=True)
        
        # Set delta_t to 0 (no temporal interpolation in this function)
        delta_t = 0
        
        # Get metrics for this image and add to the compiled metrics
        image_metrics = get_belsar_image_metrics(sites_geometry, validation_df, belsar_pred_dir, 
                                                 filename, file_suffix, date, delta_t,
                                                 NO_DATA=NO_DATA, get_error=False, bands_idx=bands_idx)
        metrics = pd.concat((metrics, image_metrics))
    
    return metrics.reset_index(drop=True)

def get_data_point_bb(gdf, dataset, margin=100, res=10):
    """
    Get the bounding box of data points in a GeoDataFrame.
    
    This function calculates a bounding box that contains all points in the GeoDataFrame,
    with an additional margin around the points.
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing point geometries.
    dataset : rasterio.DatasetReader
        Rasterio dataset containing the extent information.
    margin : int, default=100
        Margin in meters to add around the bounding box.
    res : int, default=10
        Resolution in meters for rounding coordinates.
        
    Returns:
    --------
    bbox : rasterio.coords.BoundingBox
        Bounding box containing all points with the specified margin.
        
    Raises:
    -------
    AssertionError
        If any point is outside the dataset bounds.
    """
    # Get dataset bounds
    left, right, bottom, top = (dataset.bounds.left, dataset.bounds.right, dataset.bounds.bottom, dataset.bounds.top)
    
    # Round coordinates to the specified resolution
    x_data_point = np.round(gdf["geometry"].x.values / res) * res
    y_data_point = np.round(gdf["geometry"].y.values / res) * res
    
    # Check that all points are within dataset bounds
    assert all(x_data_point > left) and all(x_data_point < right)
    assert all(y_data_point > bottom) and all(x_data_point < top)
    
    # Calculate bounding box with margin
    left = float(int(min(x_data_point) - margin))
    bottom = float(int(min(y_data_point) - margin))
    right = float(int(max(x_data_point) + margin))
    top = float(int(max(y_data_point) + margin))
    
    return rio.coords.BoundingBox(left, bottom, right, top)

def get_data_idx_in_image(gdf, xmin_image_bb, ymax_image_bb, col_offset, row_offset, res=10):
    """
    Calculate the image indices (pixel coordinates) for points in a GeoDataFrame.
    
    This function maps geographic coordinates to pixel coordinates in the image,
    taking into account the image bounds and offsets.
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing point geometries.
    xmin_image_bb : float
        Minimum x-coordinate of the image bounding box.
    ymax_image_bb : float
        Maximum y-coordinate of the image bounding box.
    col_offset : int
        Column offset to apply to the calculated indices.
    row_offset : int
        Row offset to apply to the calculated indices.
    res : int, default=10
        Resolution in meters for the image.
        
    Returns:
    --------
    None
        The function adds 'x_idx' and 'y_idx' columns to the GeoDataFrame in-place.
    """
    # Calculate pixel coordinates based on geographic coordinates
    x_data_point = (np.round(gdf["geometry"].x.values / 10) * 10 - xmin_image_bb) / res
    y_data_point = (ymax_image_bb - np.round(gdf["geometry"].y.values / 10) * 10) / res
    
    # Add pixel coordinates to the GeoDataFrame
    gdf["x_idx"] = x_data_point - col_offset
    print(gdf["x_idx"])
    gdf["y_idx"] = y_data_point - row_offset
    print(gdf["y_idx"])

def get_prosailvae_train_parser():
    """
    Create a command-line argument parser for BelSAR data processing.
    
    This function defines and returns an argument parser with options for
    specifying the data filename, data directory, and Theia product name.
    
    Returns:
    --------
    parser : argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for BelSAR data processing')

    parser.add_argument("-f", dest="data_filename",
                        help="name of data files (without extension)",
                        type=str, default="BELSAR_agriculture_database")

    parser.add_argument("-d", dest="data_dir",
                        help="Path to data directory",
                        type=str, default="data/belsar_validation/")
    
    parser.add_argument("-p", dest="product_name",
                        help="Theia product name",
                        type=str, default="")
    return parser

def get_variable_column_names(variable="lai"):
    """
    Get the column names for different biophysical variables in the BelSAR dataset.
    
    This function returns the appropriate column names for the specified variable
    and its uncertainty in the BelSAR dataset.
    
    Parameters:
    -----------
    variable : str, default="lai"
        The biophysical variable to get column names for.
        Supported variables: "lai", "lai_eff", "ccc", "ccc_eff"
        
    Returns:
    --------
    tuple
        A tuple containing (variable_column_name, uncertainty_column_name)
        
    Raises:
    -------
    NotImplementedError
        If an unsupported variable is requested.
    """
    if variable == "lai":
        return "LAI", "Uncertainty.1"
    if variable == "lai_eff":
        return "LAIeff", "Uncertainty"
    if variable == "ccc":
        return "CCC (g m-2)", "Uncertainty (g m-2).2"
    if variable == "ccc_eff":
        return "CCCeff (g m-2)", "Uncertainty (g m-2).1"
    else:
        raise NotImplementedError

def preprocess_belsar_validation_data(data_dir, filename, s2_product_name, crop="both", margin=100):
    """
    Preprocess BelSAR validation data with corresponding Sentinel-2 imagery.
    
    This function performs the core preprocessing steps for BelSAR validation data:
    1. Reads the BelSAR agriculture database Excel file for the specified crop type
    2. Cleans and formats the field measurement data
    3. Extracts the corresponding Sentinel-2 reflectance data, angles, and validity mask
    4. Saves all preprocessed data as CSV and NumPy arrays for later use in validation
    
    The function handles both wheat and maize fields separately or together, depending
    on the 'crop' parameter. Each crop type has different measured parameters in the dataset:
    - Wheat: Plant Area Index (PAI) is used as LAI
    - Maize: Green Area Index (GAI) is used as LAI
    
    Parameters:
    -----------
    data_dir : str
        Path to the data directory containing BelSAR Excel file and Sentinel-2 data.
    filename : str
        Name of the Excel file (without extension) containing BelSAR measurements.
    s2_product_name : str
        Name of the Sentinel-2 product directory.
    crop : str, default='both'
        Which crop to process ('wheat', 'maize', or 'both').
    margin : int, default=100
        Margin in meters to add around the field bounding box.
    
    Returns:
    --------
    output_file_name : str
        Base name of the output files that were created.
    
    Raises:
    -------
    ValueError
        If an invalid crop type is specified.
    """
    # Construct output filename based on Sentinel-2 acquisition date and crop type
    output_file_name = s2_product_name[8:19] + f"_{crop}_" + os.path.basename(filename)
    data_file = filename + ".xlsx"
    
    # Process the data based on crop type
    if crop != "both":
        # Read single crop type data
        data_df = pd.read_excel(os.path.join(data_dir, data_file), sheet_name=f"BelSAR_{crop}", skiprows=[0])
        # Rename columns for consistency across crop types
        if crop == "wheat":
            # For wheat, Plant Area Index (PAI) is used as LAI
            data_df.rename(columns={"Date": "date", "PAI": "lai", "Sample dry weight (g)": "cm"}, inplace=True)
        else:
            # For maize, Green Area Index (GAI) is used as LAI
            data_df.rename(columns={"Date": "date", "GAI": "lai", "Sample dry weight (g)": "cm"}, inplace=True)
    else:
        # Read and combine both wheat and maize data
        wheat_df = pd.read_excel(os.path.join(data_dir, data_file), sheet_name=f"BelSAR_wheat", skiprows=[0])
        wheat_df.rename(columns={"Date": "date", "PAI": "lai", "Sample dry weight (g)": "cm"}, inplace=True)
        maize_df = pd.read_excel(os.path.join(data_dir, data_file), sheet_name=f"BelSAR_maize", skiprows=[0])
        maize_df.rename(columns={"Date": "date", "GAI": "lai", "Sample dry weight (g)": "cm"}, inplace=True)
        data_df = pd.concat((wheat_df, maize_df))
    
    # Clean the data by removing unnecessary columns and NaN values
    data_df = data_df.drop(columns=['Flight N°', 'BBCH', 'Line 1-1',
                                    'Line 1-2', 'Line 1-3', 'Line 2-1', 'Line 2-2', 'Line 2-3', 'Line 3-1',
                                    'Line 3-2', 'Line 3-3', 'Mean','FCOVER',
                                    'Total Fresh weight (g)', 'Sample fresh wieght (g)', 'Dry matter content (%)',
                                    'Interline distance mean (cm)', 'Interplant distance (cm)',
                                    'Note/comment']).dropna()
    
    # Save the processed data as CSV
    os.makedirs(os.path.dirname(os.path.join(data_dir, output_file_name + ".csv")), exist_ok=True)
    data_df.to_csv(os.path.join(data_dir, output_file_name + ".csv"))

    # Define bounding box based on crop type to include relevant fields
    if crop == "maize":
        # Bounding box for maize fields
        left, bottom, right, top = (526023, 6552201, 535123, 6558589)
    elif crop == "wheat":
        # Bounding box for wheat fields
        left, bottom, right, top = (522857, 6544318, 531075, 6553949)
    elif crop == "both":
        # Bounding box covering both wheat and maize fields
        left, bottom, right, top = (522857, 6544318, 535123, 6558589)
    else:
        raise ValueError("Invalid crop type. Must be 'wheat', 'maize', or 'both'.")
    
    # Set coordinate reference system and path to Sentinel-2 data
    src_epsg = "epsg:3857"
    path_to_theia_product = os.path.join(data_dir, s2_product_name)
    
    # Extract Sentinel-2 data for the specified bounding box
    (s2_r, s2_a, validity_mask, xcoords, ycoords, 
     crs) = read_data_from_theia(left, bottom, right, top, src_epsg, path_to_theia_product, margin=margin)
    
    # Save the extracted data as NumPy arrays for later use
    np.save(os.path.join(data_dir, output_file_name + "_angles.npy"), s2_a)
    np.save(os.path.join(data_dir, output_file_name + "_refl.npy"), s2_r)
    np.save(os.path.join(data_dir, output_file_name + "_xcoords.npy"), xcoords)
    np.save(os.path.join(data_dir, output_file_name + "_ycoords.npy"), ycoords)
    np.save(os.path.join(data_dir, output_file_name + "_mask.npy"), validity_mask)
    
    return output_file_name

def load_belsar_validation_data(data_dir, filename):
    """
    Load preprocessed BelSAR validation data.
    
    Parameters:
    -----------
    data_dir : str
        Path to the data directory
    filename : str
        Name of the preprocessed file (without extension)
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame containing the validation data
    s2_r : numpy.ndarray
        Sentinel-2 reflectance data
    s2_a : numpy.ndarray
        Sentinel-2 angle data
    mask : numpy.ndarray
        Validity mask
    xcoords : numpy.ndarray
        X coordinates
    ycoords : numpy.ndarray
        Y coordinates
    crs : rasterio.CRS
        Coordinate reference system
    """
    df = pd.read_csv(os.path.join(data_dir, filename + ".csv"))
    s2_r = np.load(os.path.join(data_dir, filename + "_refl.npy"))
    s2_a = np.load(os.path.join(data_dir, filename + "_angles.npy"))
    mask = np.load(os.path.join(data_dir, filename + "_mask.npy"))
    xcoords = np.load(os.path.join(data_dir, filename + "_xcoords.npy"))
    ycoords = np.load(os.path.join(data_dir, filename + "_ycoords.npy"))
    return df, s2_r, s2_a, mask, xcoords, ycoords, rio.CRS.from_epsg(32631)

def get_sites_geometry(data_dir, crs, crop:str|None=None):
    """
    Get the geometries of the field sites from shapefiles.
    
    Parameters:
    -----------
    data_dir : str
        Path to the data directory
    crs : rasterio.CRS
        Coordinate reference system
    crop : str or None, default=None
        Which crop to filter for ('wheat', 'maize', or None for all)
        
    Returns:
    --------
    sites_geometry : geopandas.GeoDataFrame
        GeoDataFrame containing the field geometries
    """
    sites_geometry = gpd.GeoDataFrame()
    
    # Read wheat fields shapefile
    wheat_file = os.path.join(data_dir, "BELSAR_wheat_fields", "BELSAR_wheat_fields.shp")
    if os.path.exists(wheat_file):
        wheat_gdf = gpd.read_file(wheat_file)
        wheat_gdf = wheat_gdf.rename(columns={'id': 'Name'})
        wheat_gdf.to_crs(crs, inplace=True)
        sites_geometry = pd.concat([sites_geometry, wheat_gdf], ignore_index=True)
    
    # Read maize fields shapefile
    maize_file = os.path.join(data_dir, "BELSAR_maize_fields", "BELSAR_maize_fields.shp")
    if os.path.exists(maize_file):
        maize_gdf = gpd.read_file(maize_file)
        maize_gdf = maize_gdf.rename(columns={'id': 'Name'})
        maize_gdf.to_crs(crs, inplace=True)
        sites_geometry = pd.concat([sites_geometry, maize_gdf], ignore_index=True)
    
    sites_geometry = sites_geometry[sites_geometry['geometry'].apply(lambda x: x.geom_type in ('Polygon', 'MultiPolygon'))]
    sites_geometry.reset_index(inplace=True, drop=True)
    
    if crop is None:
        return sites_geometry
    if crop == "maize":
        return sites_geometry[sites_geometry["Name"].apply(lambda x: x[0]=="M")]    
    if crop == "wheat":
        return sites_geometry[sites_geometry["Name"].apply(lambda x: x[0]=="W")]
    else:
        raise ValueError("Invalid crop type. Must be 'wheat', 'maize', or None.")

def get_delta_dict(filename_dict):
    """
    Calculate time deltas between field measurement dates and Sentinel-2 acquisition dates.
    
    This function computes the time difference in days between each field measurement date
    and the corresponding Sentinel-2 image acquisition date.
    
    Parameters:
    -----------
    filename_dict : dict
        Dictionary mapping measurement dates (str in format 'YYYY-MM-DD') to 
        Sentinel-2 filenames (str with date in format 'YYYYMMDD' at positions 3-11).
        
    Returns:
    --------
    delta_dict : dict
        Dictionary mapping measurement dates to time deltas in days.
        Positive delta means measurement was taken after the image acquisition.
        Negative delta means measurement was taken before the image acquisition.
    """
    delta_dict = {}
    for date, filename in filename_dict.items():
        # Extract the date from the filename (positions 3-11)
        filename_date_str = filename[3:11]
        # Calculate the difference in days
        delta = (datetime.strptime(date, "%Y-%m-%d")
                 - datetime.strptime(filename_date_str, "%Y%m%d")).days
        delta_dict[date] = delta
    return delta_dict

def get_belsar_image_metrics(sites_geometry, validation_df, belsar_pred_dir, belsar_pred_filename, 
                             belsar_pred_file_suffix, date, delta_t, NO_DATA=-10000, get_error=True, 
                             bands_idx=torch.arange(10)):
    """
    Calculate metrics for a single Sentinel-2 image prediction for all field sites.
    
    This function processes prediction results for a single Sentinel-2 image, calculating
    statistics for each field site by comparing predicted values with field measurements.
    
    Parameters:
    -----------
    sites_geometry : geopandas.GeoDataFrame
        GeoDataFrame containing the field geometries (polygons).
    validation_df : pandas.DataFrame
        DataFrame containing the validation data (field measurements).
    belsar_pred_dir : str
        Path to the directory containing the predictions.
    belsar_pred_filename : str
        Filename of the prediction GeoTIFF.
    belsar_pred_file_suffix : str
        Suffix to add to the prediction filename.
    date : str
        Date of the field measurements (format: 'YYYY-MM-DD').
    delta_t : int
        Time delta in days between field measurements and Sentinel-2 acquisition.
    NO_DATA : int, default=-10000
        Value used to represent no data in the prediction rasters.
    get_error : bool, default=True
        Whether to calculate reconstruction errors.
    bands_idx : torch.Tensor, default=torch.arange(10)
        Indices of the Sentinel-2 bands to use.
        
    Returns:
    --------
    metrics : pandas.DataFrame
        DataFrame containing the calculated metrics for each field site.
        Includes field information, reference values, predicted values,
        and prediction uncertainties.
    """
    # Dictionary mapping array indices to prediction variables
    pred_array_idx = {"lai":{"mean":0, "sigma":3}, "cm":{"mean":1, "sigma":4}, "hspot":{"mean":2, "sigma":5}}
    metrics = pd.DataFrame()
    
    # Process each field site
    for i in range(len(sites_geometry)):
        line = sites_geometry.iloc[i]
        site_name = line['Name']
        polygon = line['geometry']
        
        # Extract predictions for this field
        with rio.open(os.path.join(belsar_pred_dir, f"{belsar_pred_filename}{belsar_pred_file_suffix}.tif"), 
                      mode = 'r') as src:
            masked_array, _ = mask(src, [polygon], invert=False)
            masked_array[masked_array==NO_DATA] = np.nan
            # Extract reconstruction errors if requested
            if get_error:
                masked_err = masked_array[6,...] 
                masked_bands_err = masked_array[7:,...] 

        # Get validation data for this site
        site_samples = validation_df[validation_df["Field ID"]==site_name]

        # Initialize data dictionary with site information
        d = {"name": site_name,
             "land_cover": "Wheat" if site_name[0]=="W" else "Maize",
             "date": date,
             "delta": delta_t}
        
        # Process each biophysical variable (LAI and canopy mass)
        for variable in ["lai", "cm"]:
            # Get reference values
            site_ref = site_samples[variable]   
            d[f"ref_{variable}"] = np.mean(site_ref)
            d[f"ref_{variable}_std"] = np.std(site_ref)
            d[f"{variable}_mean"] = np.nan
            d[f"{variable}_std"] = np.nan
            d[f"{variable}_sigma_mean"] = np.nan
            d[f"{variable}_sigma_std"] = np.nan

            # Calculate statistics for predicted values
            if not np.isnan(masked_array[pred_array_idx[variable]['mean'],...]).all():
                d[f"{variable}_mean"] = np.nanmean(masked_array[pred_array_idx[variable]['mean'],...])
                d[f"{variable}_std"] = np.nanstd(masked_array[pred_array_idx[variable]['mean'],...])
            else:
                continue
                
            # Calculate statistics for prediction uncertainties
            if not np.isnan(masked_array[pred_array_idx[variable]['sigma'],...]).all():
                sigma_pred = masked_array[pred_array_idx[variable]['sigma'],...]
                d[f"{variable}_sigma_mean"] = np.nanmean(sigma_pred)
                d[f"{variable}_sigma_std"] = np.nanstd(sigma_pred)
        
        # Process reconstruction errors if requested
        if get_error:
            if not np.isnan(masked_err).all():
                d[f"rec_err_mean"] = np.nanmean(masked_err)
                d[f"rec_err_std"] = np.nanstd(masked_err)
            if not np.isnan(masked_bands_err).all():
                for i, band in enumerate(np.array(BANDS)[bands_idx.cpu()].tolist()):
                    d[f"{band}_rec_err_mean"] = np.nanmean(masked_bands_err[i,...])
                    d[f"{band}_rec_err_std"] = np.nanstd(masked_bands_err[i,...])
        
        # Process hotspot parameter
        if not np.isnan(masked_array[pred_array_idx[variable]['mean'],...]).all():
            d[f"hspot_mean"] = np.nanmean(masked_array[pred_array_idx["hspot"]['mean'],...])
            d[f"hspot_std"] = np.nanstd(masked_array[pred_array_idx["hspot"]['mean'],...])
            d[f"hspot_sigma_mean"] = np.nanmean(masked_array[pred_array_idx["hspot"]['sigma'],...])
            d[f"hspot_sigma_std"] = np.nanstd(masked_array[pred_array_idx["hspot"]['sigma'],...])
        
        # Add results for this field to the metrics DataFrame
        metrics = pd.concat((metrics, pd.DataFrame(d, index=[0])))
    
    return metrics.reset_index(drop=True)

def get_belsar_campaign_metrics_df(belsar_data_dir, filename_dict, belsar_pred_dir, file_suffix, NO_DATA=-10000, 
                                   get_error=True, bands_idx=torch.arange(10)):
    """
    Calculate metrics for all field sites at all measurement dates.
    
    This function processes predictions for all dates in the provided filename_dict,
    calculating metrics for each field site by comparing predictions with field measurements.
    It handles the loading of validation data for each date, filtering of field geometries,
    and time delta calculation.
    
    Parameters:
    -----------
    belsar_data_dir : str
        Path to the directory containing the preprocessed BelSAR data.
    filename_dict : dict
        Dictionary mapping measurement dates (str in format 'YYYY-MM-DD') to 
        Sentinel-2 filenames.
    belsar_pred_dir : str
        Path to the directory containing the predictions.
    file_suffix : str
        Suffix to add to the prediction filenames.
    NO_DATA : int, default=-10000
        Value used to represent no data in the prediction rasters.
    get_error : bool, default=True
        Whether to calculate reconstruction errors.
    bands_idx : torch.Tensor, default=torch.arange(10)
        Indices of the Sentinel-2 bands to use.
        
    Returns:
    --------
    metrics : pandas.DataFrame
        DataFrame containing the calculated metrics for all field sites and dates.
        Includes field information, reference values, predicted values,
        and prediction uncertainties.
    """
    metrics = pd.DataFrame()
    
    # Calculate time deltas between field measurements and Sentinel-2 acquisitions
    delta_dict = get_delta_dict(filename_dict)
    
    # Process each date and its corresponding Sentinel-2 image
    for date, filename in filename_dict.items():
        # Load validation data for this date/image
        validation_df, _, _, _, _, _, crs = load_belsar_validation_data(belsar_data_dir, filename)
        
        # Filter validation data to only include measurements from this date
        validation_df = validation_df[validation_df['date']==date]
        
        # Get unique field IDs in the validation dataset
        ids = pd.unique(validation_df["Field ID"]).tolist()
        
        # Get field geometries and filter to only include fields in the validation dataset
        sites_geometry = get_sites_geometry(belsar_data_dir, crs)
        sites_geometry = sites_geometry[sites_geometry['Name'].apply(lambda x: x in ids)]
        sites_geometry.reset_index(inplace=True, drop=True)
        
        # Get time delta for this date
        delta_t = delta_dict[date]
        
        # Get metrics for this image and add to the compiled metrics
        image_metrics = get_belsar_image_metrics(sites_geometry, validation_df, belsar_pred_dir, 
                                                 filename, file_suffix, date, delta_t,
                                                 NO_DATA=NO_DATA, get_error=get_error, bands_idx=bands_idx)
        metrics = pd.concat((metrics, image_metrics))
    
    return metrics.reset_index(drop=True)

def interpolate_belsar_metrics(belsar_data_dir, belsar_pred_dir, method="closest", file_suffix="", 
                               get_error=True, bands_idx=torch.arange(10)):
    """
    Interpolate BelSAR metrics using different methods.
    
    This function only implements the "closest" approach as requested.
    
    Parameters:
    -----------
    belsar_data_dir : str
        Path to the BelSAR data directory
    belsar_pred_dir : str
        Path to the directory containing the predictions
    method : str, default="closest"
        Method to use for interpolation (only "closest" is supported)
    file_suffix : str, default=""
    Suffix to add to the prediction filenames
    get_error : bool, default=True
        Whether to calculate reconstruction errors
    bands_idx : torch.Tensor, default=torch.arange(10)
        Indices of the Sentinel-2 bands to use
        
    Returns:
    --------
    metrics : pandas.DataFrame
        DataFrame containing the interpolated metrics
    """
    if method != "closest":
        raise ValueError("Only the 'closest' method is supported in this version.")
    
    metrics = get_belsar_campaign_metrics_df(belsar_data_dir, closest_filename_dict, belsar_pred_dir, 
                                             file_suffix, get_error=get_error, bands_idx=bands_idx)
    return metrics

def save_belsar_predictions(belsar_dir, model, res_dir, list_filenames, model_name="pvae", mode="lat_mode",
                            save_reconstruction=False):
    """
    Apply the model to BelSAR data and save the predictions.
    
    Parameters:
    -----------
    belsar_dir : str
        Path to the BelSAR data directory
    model : torch.nn.Module
        Model to use for predictions
    res_dir : str
        Path to the directory to save the results
    list_filenames : list
        List of filenames to process
    model_name : str, default="pvae"
        Name of the model
    mode : str, default="lat_mode"
        Mode to use for predictions
    save_reconstruction : bool, default=False
        Whether to save the reconstructed images
    """
    NO_DATA = -10000
    os.makedirs(res_dir, exist_ok=True)
    
    for filename in tqdm(list_filenames):
        df, s2_r, s2_a, mask, xcoords, ycoords, crs = load_belsar_validation_data(belsar_dir, filename)
        s2_r = torch.from_numpy(s2_r).float()
        mask[mask==1.] = np.nan
        mask[mask==0.] = 1.
        if np.isnan(mask).all():
            print(f"No valid pixels in {filename}!")
        s2_r = (s2_r * torch.from_numpy(mask).float()).unsqueeze(0)
        s2_a = torch.from_numpy(s2_a).float().unsqueeze(0)
        
        with torch.no_grad():
            (rec, sim_image, s2_r, _, sigma_image) = get_encoded_image_from_batch((s2_r, s2_a), model,
                                                        patch_size=32, bands=model.encoder.bands,
                                                        mode=mode, padding=True, no_rec=not save_reconstruction)
            s2_r = s2_r[:,model.encoder.bands.to(s2_r.device),...]

        tensor = torch.cat((sim_image[6,...].unsqueeze(0),
                            sim_image[5,...].unsqueeze(0),
                            sim_image[8,...].unsqueeze(0),
                            sigma_image[6,...].unsqueeze(0), 
                            sigma_image[5,...].unsqueeze(0),
                            sigma_image[8,...].unsqueeze(0)), 0)
        if save_reconstruction:
            err_tensor = (rec - s2_r.squeeze(0)).abs().mean(0, keepdim=True)
            full_err_tensor = (rec - s2_r.squeeze(0)).abs()
            tensor = torch.cat((tensor, err_tensor, full_err_tensor), 0)
        tensor[tensor.isnan()] = NO_DATA
        tensor_to_raster(tensor, os.path.join(res_dir, f"{filename}_{model_name}_{mode}.tif"),
                         crs=crs, resolution=10, dtype=np.float32, bounds=None,
                         xcoords=xcoords, ycoords=ycoords, nodata=NO_DATA,
                         hw=0, half_res_coords=True)
    return

def main():
    """
    Preprocess BelSAR data for all available Sentinel-2 images.
    
    This script is the main entry point for preprocessing BelSAR agricultural database data
    with corresponding Sentinel-2 satellite imagery. The preprocessing involves:
    
    1. Reading the BelSAR agriculture database Excel file
    2. Processing each available Sentinel-2 image
    3. Extracting field geometries for wheat and maize fields
    4. Extracting Sentinel-2 reflectance data, angles, and coordinates for each field
    5. Saving the preprocessed data as NumPy arrays for later use in validation
    6. Optionally generating visualization plots of the fields
    
    The script processes multiple Sentinel-2 images to support validation across
    different dates throughout the growing season.
    
    Command-line Arguments:
    ----------------------
    --data_dir : str
        Path to the directory containing BelSAR data and Sentinel-2 images
    --excel_file : str
        Path to the Excel file (without extension) relative to data_dir
    --crop : str
        Which crop type to process ("wheat", "maize", or "both")
    --margin : int
        Margin in meters to add around the bounding box
    --plot : flag
        Whether to generate and save visualization plots
    """
    import argparse
    from tqdm import tqdm
    
    # Create argument parser with detailed help messages
    parser = argparse.ArgumentParser(
        description='Preprocess BelSAR data for validation of biophysical parameter retrieval',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--data_dir", 
                        type=str, 
                        default="data/belsar_validation",
                        help="Path to the BelSAR data directory")
    
    parser.add_argument("--excel_file", 
                        type=str, 
                        default="UCLouvain/BELSAR_agriculture_database",
                        help="Path to the Excel file (without extension) relative to data_dir")
    
    parser.add_argument("--crop", 
                        type=str, 
                        default="both",
                        choices=["wheat", "maize", "both"],
                        help="Which crop type to process")
    
    parser.add_argument("--margin", 
                        type=int, 
                        default=100,
                        help="Margin in meters to add around the bounding box")
    
    parser.add_argument("--plot", 
                        action="store_true",
                        help="Whether to plot and save field boundaries on Sentinel-2 images")
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Check if the Excel file exists
    excel_path = os.path.join(args.data_dir, args.excel_file + ".xlsx")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    # Find all Sentinel-2 directories in the data directory
    sentinel_products = [d for d in os.listdir(args.data_dir) if d.startswith('SENTINEL2')]
    if not sentinel_products:
        raise ValueError(f"No Sentinel-2 data found in {args.data_dir}")
    
    print(f"Found {len(sentinel_products)} Sentinel-2 products")
    
    # Process each Sentinel-2 product
    for s2_product in tqdm(sentinel_products, desc="Processing Sentinel-2 products"):
        try:
            print(f"\nProcessing {s2_product}...")
            
            # Preprocess the data for this Sentinel-2 product
            output_file_name = preprocess_belsar_validation_data(
                args.data_dir, 
                args.excel_file, 
                s2_product, 
                crop=args.crop,
                margin=args.margin
            )
            
            # Generate visualization if requested
            if args.plot:
                plot_belsar_site(args.data_dir, output_file_name)
                print(f"  Plot saved as {output_file_name}_mask.png")
            
            print(f"  Successfully processed: {output_file_name}")
            
        except Exception as e:
            print(f"  Error processing {s2_product}: {str(e)}")
    
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()