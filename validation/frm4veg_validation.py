#!/usr/bin/env python
# coding:utf-8

import os
import sys
import gc
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import socket
from datetime import datetime
import torch
from tqdm import tqdm
import rasterio as rio
from sensorsio.sentinel2 import Sentinel2

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import essential modules
from utils.image_utils import get_encoded_image_from_batch
from prosailvae.ProsailSimus import BANDS

# Import validation utilities
if __name__ == "__main__":
    from validation_utils import var_of_product, simple_interpolate
else:
    from validation.validation_utils import var_of_product, simple_interpolate

# Comment out unnecessary imports and code
# BARRAX_FILENAMES = ["2B_20180516_FRM_Veg_Barrax_20180605_V2", "2A_20180613_FRM_Veg_Barrax_20180605_V2"]
BARRAX_FILENAMES = ["2A_20180613_FRM_Veg_Barrax_20180605_V2"]
BARRAX_2021_FILENAME = "2B_20210722_FRM_Veg_Barrax_20210719_V2"
WYTHAM_FILENAMES = ["2A_20180629_FRM_Veg_Wytham_20180703_V2"]

def load_frm4veg_data(data_dir, filename, variable="lai"):
    """
    Load FRM4VEG data files for a specific site and variable.
    
    Args:
        data_dir: Base directory containing FRM4VEG data
        filename: Name of data file (without extension)
        variable: Biophysical variable to load (lai, lai_eff, ccc, ccc_eff)
        
    Returns:
        Tuple of (gdf, s2_r, s2_a, xcoords, ycoords)
    """
    # Determine site-specific subdirectory
    site_name = "Barrax" if "Barrax" in filename else "Wytham"
    site_year = "2018" if "2018" in filename else "2021"
    site_dir = os.path.join(data_dir, f"{site_name}_{site_year}")
    
    # Check if the site directory exists
    if not os.path.exists(site_dir):
        # Fall back to the base directory if site subdirectory doesn't exist
        site_dir = data_dir
        print(f"Warning: Site directory {site_dir} not found, using base directory {data_dir}")
    
    print(f"Loading {variable} data from {site_dir}/{filename}")
    
    gdf = gpd.read_file(os.path.join(site_dir, filename + f"_{variable}.geojson"),
                        driver="GeoJSON")
    s2_r = np.load(os.path.join(site_dir, filename + "_refl.npy"))
    s2_a = np.load(os.path.join(site_dir, filename + "_angles.npy"))
    xcoords = np.load(os.path.join(site_dir, filename + "_xcoords.npy"))
    ycoords = np.load(os.path.join(site_dir, filename + "_ycoords.npy"))
    return gdf, s2_r, s2_a, xcoords, ycoords

def get_frm4veg_material(frm4veg_data_dir, frm4veg_filename):
    """
    Load FRM4VEG material for a specific site, including reflectance, angles, and reference data.
    
    Args:
        frm4veg_data_dir: Base directory containing FRM4VEG data
        frm4veg_filename: Name of the data file (without extension)
        
    Returns:
        Tuple of (s2_r, s2_a, site_idx_dict, ref_dict)
    """
    site_idx_dict = {}
    ref_dict = {}
    
    # Extract date from filename (assuming format like "2A_20180613_FRM_Veg_Barrax_20180605_V2")
    s2_date_str = frm4veg_filename.split("_")[1]
    s2_date = datetime.strptime(s2_date_str, '%Y%m%d').date()
    print(f"Processing {frm4veg_filename} with S2 acquisition date: {s2_date}")
    
    for variable in ['lai', 'lai_eff', 'ccc', 'ccc_eff']:
        gdf, _, _, _, _ = load_frm4veg_data(frm4veg_data_dir, frm4veg_filename, variable=variable)
        
        # Filter out alfalfa measurements from 2018 Barrax data
        # Check if this is 2018 Barrax data (based on filename)
        if "2018" in frm4veg_filename and "Barrax" in frm4veg_filename:
            # Keep track of the count before filtering
            count_before = len(gdf)
            # Filter out alfalfa measurements
            gdf = gdf[gdf["land cover"].str.lower() != "alfalfa"].reset_index(drop=True)
            count_after = len(gdf)
            print(f"Filtered out {count_before - count_after} alfalfa measurements from 2018 Barrax data for {variable}")
        
        # Add date information if not present
        if "date" not in gdf.columns:
            # Extract ground data collection date from filename (format: "..._FRM_Veg_Barrax_20180605_V2")
            # The ground data date is often in the second-to-last part before "_V2"
            try:
                ground_date_parts = frm4veg_filename.split("_")
                ground_date_str = [part for part in ground_date_parts if len(part) == 8 and part.isdigit()][-1]
                ground_date = datetime.strptime(ground_date_str, '%Y%m%d').date()
                gdf["date"] = ground_date
                print(f"Added ground data collection date: {ground_date} for {variable}")
            except Exception as e:
                print(f"Warning: Could not extract ground data date from filename: {e}")
                # Use S2 date as fallback
                gdf["date"] = s2_date
        
        # gdf = gdf.iloc[:51]
        ref_dict[variable] = gdf[variable].values.reshape(-1)
        ref_dict[variable+"_std"] = gdf["uncertainty"].values.reshape(-1)
        site_idx_dict[variable] = {"x_idx" : torch.from_numpy(gdf["x_idx"].values).int(),
                                   "y_idx" : torch.from_numpy(gdf["y_idx"].values).int()}
        
        # Also store dates in ref_dict
        ref_dict[f"{variable}_date"] = gdf["date"].apply(lambda x: (x.date() - s2_date).days if hasattr(x, 'date') else (x - s2_date).days).values
        
    _, s2_r, s2_a, _, _ = load_frm4veg_data(frm4veg_data_dir, frm4veg_filename, variable="lai")
    s2_r = torch.from_numpy(s2_r).float().unsqueeze(0)
    s2_a = torch.from_numpy(s2_a).float().unsqueeze(0)
    return s2_r, s2_a, site_idx_dict, ref_dict

def get_model_frm4veg_results(model, s2_r, s2_a, site_idx_dict, ref_dict, mode="lat_mode", 
                              get_reconstruction=False):
    with torch.no_grad():
        (rec, sim_image, cropped_s2_r, cropped_s2_a,
            sigma_image) = get_encoded_image_from_batch((s2_r, s2_a), model,
                                            patch_size=32, bands=model.encoder.bands,
                                            mode=mode, padding=True, no_rec=not get_reconstruction)
        cropped_s2_r = cropped_s2_r[:,model.encoder.bands.to(cropped_s2_r.device),...]
        rec_err = (rec - cropped_s2_r.squeeze(0)).abs().mean(0, keepdim=True)
        band_rec_err = (rec - cropped_s2_r.squeeze(0)).abs()
    model_pred = {"s2_r":cropped_s2_r, "s2_a":cropped_s2_a}

    for lai_variable in ['lai', 'lai_eff']: # 'ccc', 'ccc_eff']:
        model_pred[lai_variable] = sim_image[6, site_idx_dict[lai_variable]['y_idx'], 
                                                site_idx_dict[lai_variable]['x_idx']].numpy()
        model_pred[f"{lai_variable}_std"] = sigma_image[6, site_idx_dict[lai_variable]['y_idx'], 
                                                           site_idx_dict[lai_variable]['x_idx']].numpy()
        model_pred[f"ref_{lai_variable}"] = ref_dict[lai_variable]
        model_pred[f"ref_{lai_variable}_std"] = ref_dict[f"{lai_variable}_std"]

        model_pred[f"{lai_variable}_rec_err"] = rec_err[..., site_idx_dict[lai_variable]['y_idx'], 
                                                             site_idx_dict[lai_variable]['x_idx']].numpy()
        for i, band in enumerate(np.array(BANDS)[model.encoder.bands.detach().cpu()].tolist()):
            model_pred[f"{lai_variable}_{band}_rec_err"] = band_rec_err[i, site_idx_dict[lai_variable]['y_idx'], 
                                                                             site_idx_dict[lai_variable]['x_idx']].numpy()
    for ccc_variable in ['ccc', 'ccc_eff']:
        model_pred[ccc_variable] = (sim_image[1, site_idx_dict[ccc_variable]['y_idx'], 
                                                 site_idx_dict[ccc_variable]['x_idx']] 
                                    * sim_image[6, site_idx_dict[ccc_variable]['y_idx'], 
                                                   site_idx_dict[ccc_variable]['x_idx']]).numpy()
        m_1 = sim_image[1, site_idx_dict[ccc_variable]['y_idx'], site_idx_dict[ccc_variable]['x_idx']]
        m_2 = sim_image[6, site_idx_dict[ccc_variable]['y_idx'], site_idx_dict[ccc_variable]['x_idx']]
        v_1 = sigma_image[1, site_idx_dict[ccc_variable]['y_idx'], site_idx_dict[ccc_variable]['x_idx']].pow(2)
        v_2 = sigma_image[6, site_idx_dict[ccc_variable]['y_idx'], site_idx_dict[ccc_variable]['x_idx']].pow(2)
        model_pred[f"{ccc_variable}_std"] = var_of_product(v_1, v_2, m_1, m_2).sqrt().numpy()
        model_pred[f"ref_{ccc_variable}"] = ref_dict[ccc_variable]
        model_pred[f"ref_{ccc_variable}_std"] = ref_dict[f"{ccc_variable}_std"]
        model_pred[f"{ccc_variable}_rec_err"] = rec_err[..., site_idx_dict[ccc_variable]['y_idx'], 
                                                             site_idx_dict[ccc_variable]['x_idx']].numpy()
        for i, band in enumerate(np.array(BANDS)[model.encoder.bands.detach().cpu()].tolist()):
            model_pred[f"{ccc_variable}_{band}_rec_err"] = band_rec_err[i, site_idx_dict[ccc_variable]['y_idx'], 
                                                                        site_idx_dict[ccc_variable]['x_idx']].numpy()
                                                             
    return model_pred

def get_data_point_bb(gdf, dataset, margin=100, res=10):
    left, right, bottom, top = (dataset.bounds.left, dataset.bounds.right, 
                                dataset.bounds.bottom, dataset.bounds.top)
    x_data_point = np.round(gdf["geometry"].x.values / res) * res
    y_data_point = np.round(gdf["geometry"].y.values / res) * res
    assert all(x_data_point > left) and all(x_data_point < right)
    assert all(y_data_point > bottom) and all(x_data_point < top)
    left = float(int(min(x_data_point) - margin))
    bottom = float(int(min(y_data_point) - margin))
    right = float(int(max(x_data_point) + margin))
    top = float(int(max(y_data_point) + margin))
    return rio.coords.BoundingBox(left, bottom, right, top)

def get_data_idx_in_image(gdf, xmin_image_bb, ymax_image_bb, col_offset, row_offset, res=10):
    x_data_point = (np.round(gdf["geometry"].x.values / 10) * 10 - xmin_image_bb) / res
    y_data_point = (ymax_image_bb - np.round(gdf["geometry"].y.values / 10) * 10) / res
    gdf["x_idx"] = x_data_point - col_offset
    print(gdf["x_idx"])
    gdf["y_idx"] = y_data_point - row_offset
    print(gdf["y_idx"])


def get_bb_array_index(bb, image_bb, res=10):
    xmin = (bb[0] - image_bb[0]) / res
    ymin = ( - (bb[3] - image_bb[3])) / res
    xmax = xmin + (bb[2] - bb[0]) / res
    ymax = ymin + (bb[3] - bb[1]) / res
    return int(xmin), int(ymin), int(xmax), int(ymax)

def get_prosailvae_train_parser():
    """
    Creates an argument parser for FRM4VEG data processing.
    """
    parser = argparse.ArgumentParser(description='Parser for FRM4VEG data processing')

    parser.add_argument("-f", "--data_filename", dest="data_filename",
                        help="name of data files (without extension)",
                        type=str, default="FRM_Veg_Barrax_20210719_V2")

    parser.add_argument("-d", "--data_dir", dest="data_dir",
                        help="Directory containing FRM4VEG data and Sentinel-2 tiles",
                        type=str, default="data/frm4veg_validation")
    
    parser.add_argument("-p", "--product_name", dest="product_name",
                        help="Sentinel-2 product name",
                        type=str, default="SENTINEL2B_20210722-111020-007_L2A_T30SWJ_C_V3-0")
    
    parser.add_argument("--date", dest="date",
                        help="Date of the Sentinel-2 acquisition (YYYY-MM-DD)",
                        type=str, default="2021-07-22")
    
    parser.add_argument("--method", dest="method",
                        help="Method used for ground data collection (e.g., DHP)",
                        type=str, default="DHP")
    
    parser.add_argument("--no_angle_data", dest="no_angle_data",
                        help="Set to true if angle data is not available",
                        action='store_true')
    
    parser.add_argument("--process_all", dest="process_all",
                        help="Process all validation sites at once",
                        action='store_true')
    
    return parser

def get_variable_column_names(variable="lai", wytham=False):
    if variable == "lai":
        if wytham:
            return "LAI", "Uncertainty.5"
        return "LAI", "Uncertainty.1"
    if variable == "lai_eff":
        if wytham:
            return "LAIeff", "Uncertainty.2"
        return "LAIeff", "Uncertainty"
    if variable == "ccc":
        return "CCC (g m-2)", "Uncertainty (g m-2).2"
    if variable == "ccc_eff":
        return "CCCeff (g m-2)", "Uncertainty (g m-2).1"
    else:
        raise NotImplementedError

def get_bb_equivalent_polygon(bb, in_crs, out_crs):
    from shapely.geometry import Polygon
    coords = ((bb.left, bb.bottom), (bb.left, bb.top), (bb.right, bb.top), (bb.right, bb.bottom), (bb.left, bb.bottom))
    polygon = Polygon(coords)
    return gpd.GeoDataFrame(data={"geometry":[polygon]}).set_crs(in_crs).to_crs(out_crs)

def compute_frm4veg_data(data_dir, filename, s2_product_name, no_angle_data=False, date="2021-07-22", method="DHP"):
    """
    Compute FRM4VEG data for a specific site and date.
    
    Args:
        data_dir: Directory containing FRM4VEG data
        filename: Name of data file (without extension)
        s2_product_name: Name of the Sentinel-2 product
        no_angle_data: Whether angle data is available
        date: Date of S2 acquisition
        method: Method used for ground data collection (e.g., "DHP")
        
    Returns:
        Output filename generated
    """
    output_file_name = s2_product_name[8:19] + "_" + filename
    data_file = filename + ".xlsx"
    
    # Create site-specific output directory
    site_name = "Barrax" if "Barrax" in filename else "Wytham"
    site_year = "2018" if "2018" in filename else "2021"
    output_subdir = os.path.join(data_dir, f"{site_name}_{site_year}")
    os.makedirs(output_subdir, exist_ok=True)
    
    print(f"Processing {filename} with {s2_product_name}")
    print(f"Input data file: {os.path.join(data_dir, data_file)}")
    print(f"Output directory: {output_subdir}")

    data_df = pd.read_excel(os.path.join(data_dir, data_file), sheet_name="GroundData", skiprows=[0])
    print(f"Loaded data with columns: {data_df.columns}")
    data_df = data_df.drop(columns=['Comments'])
    data_df = data_df[data_df["Method"]==method]
    
    # Filter out alfalfa measurements from 2018 Barrax data as proposed in Brown et al. (2021b)
    is_2018_barrax = "2018" in filename and "Barrax" in filename
    if is_2018_barrax:
        # Keep track of the count before filtering
        count_before = len(data_df)
        # Filter out alfalfa (case insensitive)
        data_df = data_df[~data_df["Land Cover"].str.lower().str.contains("alfalfa", na=False)]
        count_after = len(data_df)
        print(f"Filtered out {count_before - count_after} alfalfa measurements from 2018 Barrax data")
    
    data_gdf = gpd.GeoDataFrame(data_df, geometry=gpd.points_from_xy(data_df['Easting Coord. '], data_df['Northing Coord. ']))
    data_gdf = data_gdf.set_crs('epsg:4326')

    # Path to the Sentinel-2 tile
    path_to_theia_product = os.path.join(data_dir, s2_product_name)
    print(f"Reading Sentinel-2 data from: {path_to_theia_product}")
    
    # Create output directory if it doesn't exist
    dataset = Sentinel2(path_to_theia_product)
    data_gdf = data_gdf.to_crs(dataset.crs.to_epsg())
    margin = 100
    bb = get_data_point_bb(data_gdf, dataset, margin=margin)
    gdf = get_bb_equivalent_polygon(bb, dataset.crs.to_epsg(), 'epsg:4326')
    # Save ROIs to the output subdirectory
    gdf.to_file(os.path.join(output_subdir, "rois_to_download.geojson"), driver="GeoJSON")
    
    bands = [Sentinel2.B2,
             Sentinel2.B3,
             Sentinel2.B4,
             Sentinel2.B5,
             Sentinel2.B6,
             Sentinel2.B7,
             Sentinel2.B8,
             Sentinel2.B8A,
             Sentinel2.B11,
             Sentinel2.B12]

    xmin, ymin, xmax, ymax = get_bb_array_index(bb, dataset.bounds, res=10)
    print(f"Reading reflectance data from bounds: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
    s2_r, masks, atm, xcoords, ycoords, crs = dataset.read_as_numpy(bands, bounds=bb,
                                                                    crs=dataset.crs,
                                                                    band_type=dataset.SRE)
    if no_angle_data:
        # Remove iris data loading and provide a comment about removing IRIS dependency
        print("Warning: no_angle_data is True but IRIS data loading has been removed")
        # Using placeholder angle data instead
        s2_a = np.zeros((3, s2_r.shape[1], s2_r.shape[2]))
        print("Using placeholder zero angles instead")
    else:
        if socket.gethostname()=='CELL200973':
            even_zen, odd_zen = dataset.read_zenith_angles_as_numpy()
            even_zen = even_zen[ymin:ymax, xmin:xmax]
            odd_zen = odd_zen[ymin:ymax, xmin:xmax]
            joint_zen = np.array(even_zen)
            joint_zen[np.isnan(even_zen)] = odd_zen[np.isnan(even_zen)]
            del even_zen
            del odd_zen

            even_az, odd_az = dataset.read_azimuth_angles_as_numpy()
            even_az = even_az[ymin:ymax, xmin:xmax]
            odd_az = odd_az[ymin:ymax, xmin:xmax]
            joint_az = np.array(even_az)
            joint_az[np.isnan(even_az)] = odd_az[np.isnan(even_az)]
            del even_az
            del odd_az

        else:
            even_zen, odd_zen, even_az, odd_az = dataset.read_incidence_angles_as_numpy()
            joint_zen = np.array(even_zen)[ymin:ymax, xmin:xmax]
            joint_zen[np.isnan(even_zen[ymin:ymax, xmin:xmax])] = odd_zen[ymin:ymax, xmin:xmax][np.isnan(even_zen[ymin:ymax, xmin:xmax])]
            del even_zen
            del odd_zen
            joint_az = np.array(even_az)[ymin:ymax, xmin:xmax]
            joint_az[np.isnan(even_az[ymin:ymax, xmin:xmax])] = odd_az[ymin:ymax, xmin:xmax][np.isnan(even_az[ymin:ymax, xmin:xmax])]
            del even_az
            del odd_az
        sun_zen, sun_az = dataset.read_solar_angles_as_numpy()
        sun_zen = sun_zen[ymin:ymax, xmin:xmax]
        sun_az = sun_az[ymin:ymax, xmin:xmax]
        s2_a = np.stack((sun_zen, joint_zen, sun_az - joint_az), 0).data
    print(f"Angle data shape: {s2_a.shape}")
    
    # Save outputs to the site-specific subdirectory
    np.save(os.path.join(output_subdir, output_file_name + "_angles.npy"), s2_a)
    s2_r, masks, atm, xcoords, ycoords, crs = dataset.read_as_numpy(bands, bounds=bb,
                                                                    crs=dataset.crs,
                                                                    band_type=dataset.SRE)
    if masks.sum()>0:
        raise ValueError("Invalid mask data detected in reflectance.")
    s2_r = s2_r.data
    print(f"Reflectance data shape: {s2_r.shape}")
    np.save(os.path.join(output_subdir, output_file_name + "_refl.npy"), s2_r)
    np.save(os.path.join(output_subdir, output_file_name + "_xcoords.npy"), xcoords)
    np.save(os.path.join(output_subdir, output_file_name + "_ycoords.npy"), ycoords)
    get_data_idx_in_image(data_gdf, dataset.bounds[0], dataset.bounds[3], xmin, ymin, res=10)

    for variable in ["lai", "lai_eff", "ccc", "ccc_eff"]:
        variable_col, uncertainty_col = get_variable_column_names(variable=variable, wytham=no_angle_data)
        gdf = data_gdf[[variable_col, uncertainty_col, "Land Cover", "x_idx", "y_idx",
                        "geometry"]].dropna().reset_index(drop=True)
        gdf.rename(columns = {variable_col:variable,
                              uncertainty_col:"uncertainty",
                              "Land Cover":"land cover"}, inplace = True)
        if variable in ["ccc", "ccc_eff"]:
            gdf[variable] = gdf[variable] * 100
            gdf["uncertainty"] = gdf["uncertainty"] * 100
        gdf.to_file(os.path.join(output_subdir, output_file_name + f"_{variable}.geojson"), driver="GeoJSON")
    
    print(f"All output files saved to: {output_subdir}")
    return output_file_name

def get_frm4veg_results_at_date(model, frm4veg_data_dir, filename, mode="sim_tg_mean", 
                                get_reconstruction=True):
    """
    Get validation results for a specific date
    
    Args:
        model: The PROSAIL-VAE model
        frm4veg_data_dir: Directory with FRM4VEG validation data
        filename: Filename of the validation data
        mode: Mode for model prediction ('sim_tg_mean', 'lat_mode', etc.)
        
    Returns:
        Dictionary with validation results
    """
    sensor = filename.split("_")[0]
    (s2_r, s2_a, site_idx_dict, ref_dict) = get_frm4veg_material(frm4veg_data_dir, filename)
    validation_results = get_model_frm4veg_results(model, s2_r, s2_a, site_idx_dict, 
                                                  ref_dict, mode=mode, 
                                                  get_reconstruction=get_reconstruction)
    
    d = datetime.strptime(filename.split("_")[1], '%Y%m%d').date()
    for variable in ["lai", "lai_eff", "ccc", "ccc_eff"]:
        gdf, _, _ , _, _ = load_frm4veg_data(frm4veg_data_dir, filename, variable=variable)
        validation_results[f"{variable}_land_cover"] = gdf["land cover"].values
        validation_results[f"{variable}_date"] = gdf["date"].apply(lambda x: (x.date() - d).days).values
    return validation_results

def interpolate_frm4veg_pred(model, frm4veg_data_dir, filename_before, filename_after=None, 
                             method="simple_interpolate", mode="sim_tg_mean",
                             get_reconstruction=True, bands_idx=torch.arange(10)):
    """
    Interpolate validation results between two dates
    
    Args:
        model: The PROSAIL-VAE model
        frm4veg_data_dir: Directory with FRM4VEG validation data
        filename_before: Filename for first date
        filename_after: Filename for second date
        method: Interpolation method ('simple_interpolate', 'best', 'worst')
        mode: Mode for model prediction
        get_reconstruction: Whether to get reconstruction
        bands_idx: Indices of bands to use
        
    Returns:
        Dictionary with interpolated results
        
    Note:
        As proposed in Brown et al. (2021b), alfalfa measurements from 2018 Barrax data 
        are automatically filtered out because these crops had been thinned prior to 
        the Sentinel acquisitions, but after the in-situ measurements were made.
    """
    validation_results_before = get_frm4veg_results_at_date(model, frm4veg_data_dir, filename_before, 
                                                          mode=mode,
                                                          get_reconstruction=get_reconstruction)
    d_before = datetime.strptime(filename_before.split("_")[1], '%Y%m%d').date()
    validation_results_after = get_frm4veg_results_at_date(model, frm4veg_data_dir, filename_after, 
                                                          mode=mode,
                                                          get_reconstruction=get_reconstruction)
    d_after = datetime.strptime(filename_after.split("_")[1], '%Y%m%d').date()
    
    model_results = {}

    for variable in ["lai", "lai_eff", "ccc", "ccc_eff"]:
        model_results[f'ref_{variable}'] = validation_results_before[f'ref_{variable}']
        model_results[f'ref_{variable}_std'] = validation_results_before[f'ref_{variable}_std']
        gdf, _, _ , _, _ = load_frm4veg_data(frm4veg_data_dir, filename_before, variable=variable)
        model_results[f"{variable}_land_cover"] = gdf["land cover"].values
        dt_before = gdf["date"].apply(lambda x: (x.date() - d_before).days).values
        dt_after = gdf["date"].apply(lambda x: (x.date() - d_after).days).values
        
        if method=="simple_interpolate":
            model_results[variable] = simple_interpolate(validation_results_after[variable].squeeze(),
                                                       validation_results_before[variable].squeeze(),
                                                       dt_after, dt_before).squeeze()
            
            model_results[f"{variable}_rec_err"] = simple_interpolate(validation_results_after[f"{variable}_rec_err"].squeeze(),
                                                                    validation_results_before[f"{variable}_rec_err"].squeeze(),
                                                  dt_after, dt_before).squeeze()
            for band in np.array(BANDS)[bands_idx.cpu()].tolist():
                model_results[f"{variable}_{band}_rec_err"] = simple_interpolate(validation_results_after[f"{variable}_{band}_rec_err"].squeeze(),
                                                                               validation_results_before[f"{variable}_{band}_rec_err"].squeeze(),
                                                                               dt_after, dt_before).squeeze()
            model_results[f"{variable}_std"] = simple_interpolate(validation_results_after[f"{variable}_std"].squeeze(),
                                                                validation_results_before[f"{variable}_std"].squeeze(),
                                                                dt_after, dt_before, is_std=True).squeeze()
            model_results[f"{variable}_date"] = (abs(dt_before) + abs(dt_after)) / 2
        
        elif method == "best":
            ref = validation_results_before[f"ref_{variable}"]
            err_1 = np.abs(validation_results_before[f"{variable}"] - ref)
            err_2 = np.abs(validation_results_after[f"{variable}"] - ref)
            date = np.zeros_like(ref)
            results = np.zeros_like(ref)
            results_std = np.zeros_like(ref)
            results_rec_err = np.zeros_like(ref)
            err_1_le_err_2 = (err_1 <= err_2).reshape(-1)
            
            # Choose results from before or after based on error
            results[err_1_le_err_2] = validation_results_before[f"{variable}"].reshape(-1)[err_1_le_err_2]
            results[np.logical_not(err_1_le_err_2)] = validation_results_after[f"{variable}"].reshape(-1)[np.logical_not(err_1_le_err_2)]
            model_results[variable] = results

            date[err_1_le_err_2] = abs(dt_before[err_1_le_err_2])
            date[np.logical_not(err_1_le_err_2)] = abs(dt_after[np.logical_not(err_1_le_err_2)])
            model_results[f"{variable}_date"] = date

            results_std[err_1_le_err_2] = validation_results_before[f"{variable}_std"].reshape(-1)[err_1_le_err_2]
            results_std[np.logical_not(err_1_le_err_2)] = validation_results_after[f"{variable}_std"].reshape(-1)[np.logical_not(err_1_le_err_2)]
            model_results[f"{variable}_std"] = results_std

            results_rec_err[err_1_le_err_2] = validation_results_after[f"{variable}_rec_err"].reshape(-1)[err_1_le_err_2]
            results_rec_err[np.logical_not(err_1_le_err_2)] = validation_results_after[f"{variable}_rec_err"].reshape(-1)[np.logical_not(err_1_le_err_2)]
            model_results[f"{variable}_rec_err"] = results_rec_err
            
            for band in np.array(BANDS)[bands_idx.cpu()].tolist():
                results_band_rec_err = np.zeros_like(ref)
                results_band_rec_err[err_1_le_err_2] = validation_results_after[f"{variable}_{band}_rec_err"].reshape(-1)[err_1_le_err_2]
                results_band_rec_err[np.logical_not(err_1_le_err_2)] = validation_results_after[f"{variable}_{band}_rec_err"].reshape(-1)[np.logical_not(err_1_le_err_2)]
                model_results[f"{variable}_{band}_rec_err"] = results_band_rec_err
        
        elif method == "worst":
            ref = validation_results_before[f"ref_{variable}"]
            err_1 = np.abs(validation_results_before[f"{variable}"] - ref)
            err_2 = np.abs(validation_results_after[f"{variable}"] - ref)
            results = np.zeros_like(ref)
            results_std = np.zeros_like(ref)
            results_rec_err = np.zeros_like(ref)
            err_1_le_err_2 = (err_1 <= err_2).reshape(-1)
            date = np.zeros_like(ref)
            
            # Take worst results (opposite of "best" method)
            date[err_1_le_err_2] = abs(dt_after[err_1_le_err_2])
            date[np.logical_not(err_1_le_err_2)] = abs(dt_before[np.logical_not(err_1_le_err_2)])
            model_results[f"{variable}_date"] = date

            results[err_1_le_err_2] = validation_results_after[f"{variable}"].reshape(-1)[err_1_le_err_2]
            results[np.logical_not(err_1_le_err_2)] = validation_results_before[f"{variable}"].reshape(-1)[np.logical_not(err_1_le_err_2)]
            model_results[variable] = results

            results_std[err_1_le_err_2] = validation_results_after[f"{variable}_std"].reshape(-1)[err_1_le_err_2]
            results_std[np.logical_not(err_1_le_err_2)] = validation_results_before[f"{variable}_std"].reshape(-1)[np.logical_not(err_1_le_err_2)]
            model_results[f"{variable}_std"] = results_std

            results_rec_err[err_1_le_err_2] = validation_results_after[f"{variable}_rec_err"].reshape(-1)[err_1_le_err_2]
            results_rec_err[np.logical_not(err_1_le_err_2)] = validation_results_before[f"{variable}_rec_err"].reshape(-1)[np.logical_not(err_1_le_err_2)]
            model_results[f"{variable}_rec_err"] = results_rec_err
            
            for band in np.array(BANDS)[bands_idx.cpu()].tolist():
                results_band_rec_err = np.zeros_like(ref)
                results_band_rec_err[err_1_le_err_2] = validation_results_after[f"{variable}_{band}_rec_err"].reshape(-1)[err_1_le_err_2]
                results_band_rec_err[np.logical_not(err_1_le_err_2)] = validation_results_before[f"{variable}_{band}_rec_err"].reshape(-1)[np.logical_not(err_1_le_err_2)]
                model_results[f"{variable}_{band}_rec_err"] = results_band_rec_err
        
        elif method == "dist_interpolate":
            raise NotImplementedError
        else:
            raise ValueError(method)
    
    return model_results

def process_all_frm4veg_sites(base_dir="data/frm4veg_validation"):
    """
    Process FRM4VEG data for all validation sites.
    
    This function processes all the FRM4VEG validation sites:
    - Barrax 2018
    - Barrax 2021
    - Wytham 2018
    
    Args:
        base_dir: Base directory containing FRM4VEG validation data
    
    Returns:
        List of output filenames generated
    """
    sites_data = [
        # Barrax 2018
        {
            "data_file": "FRM_Veg_Barrax_20180605_V2.xlsx",
            "tile": "SENTINEL2A_20180613-110957-425_L2A_T30SWJ_D_V1-8",
            "date": "2018-06-13",
            "no_angle_data": False,
            "method": "DHP"
        },
        # Barrax 2021
        {
            "data_file": "FRM_Veg_Barrax_20210719_V2.xlsx",
            "tile": "SENTINEL2B_20210722-111020-007_L2A_T30SWJ_C_V3-0",
            "date": "2021-07-22",
            "no_angle_data": False,
            "method": "DHP"
        },
        # Wytham 2018
        {
            "data_file": "FRM_Veg_Wytham_20180703_V2.xlsx",
            "tile": "SENTINEL2A_20180629-112645-306_L2A_T30UXC_C_V4-0",
            "date": "2018-06-29",
            "no_angle_data": False,
            "method": "DHP"
        }
    ]
    
    output_filenames = []
    
    for site in sites_data:
        print(f"\n==== Processing {site['data_file']} with {site['tile']} ====\n")
        
        # Extract the filename without extension
        filename = site['data_file'].split(".")[0]
        
        try:
            output_filename = compute_frm4veg_data(
                data_dir=base_dir,
                filename=filename,
                s2_product_name=site['tile'],
                no_angle_data=site['no_angle_data'],
                date=site['date'],
                method=site['method']
            )
            
            output_filenames.append(output_filename)
            print(f"Successfully processed: {output_filename}")
            
        except Exception as e:
            print(f"Error processing {filename} with {site['tile']}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    return output_filenames

def main():
    """Process FRM4VEG data for validation sites."""
    try:
        # Parse command line arguments if provided
        parser = get_prosailvae_train_parser()
        args = parser.parse_args()
        
        # Default base directory - can be overridden by command line args
        base_dir = args.data_dir
        
        # Check if we should process all sites
        if args.process_all:
            print(f"Processing all FRM4VEG validation sites from {base_dir}")
            output_filenames = process_all_frm4veg_sites(base_dir=base_dir)
            
            print("\n==== Completed processing all sites ====")
            for filename in output_filenames:
                print(f" - {filename}")
            return
        
        # If on Prince's computer, default to processing all sites unless specific args are given
        if (socket.gethostname() == 'CELL200973' or os.path.exists("/Users/princemensah")) and not args.data_filename:
            # Try to find the base directory
            if not os.path.exists(base_dir):
                # Check common locations
                possible_paths = [
                    "data/frm4veg_validation",
                    "/Users/princemensah/Desktop/transformervae/prosailvae/data/frm4veg_validation",
                    "/Users/princemensah/Desktop/InstaDeep/prosailvae/data/frm4veg_validation",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/frm4veg_validation")
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        base_dir = path
                        print(f"Found data directory: {base_dir}")
                        break
                        
            # Process all validation sites 
            print(f"Processing all FRM4VEG validation sites from {base_dir}")
            output_filenames = process_all_frm4veg_sites(base_dir=base_dir)
            
            print("\n==== Completed processing all sites ====")
            for filename in output_filenames:
                print(f" - {filename}")
            return
        
        # Process a single site based on command line arguments
        if args.data_filename and args.product_name:
            print(f"Processing single site: {args.data_filename} with {args.product_name}")
            s2_product_name = args.product_name
            output_file_name = compute_frm4veg_data(
                args.data_dir, 
                args.data_filename, 
                s2_product_name,
                date=args.date, 
                no_angle_data=args.no_angle_data,
                method=args.method
            )
            print(f"Successfully processed: {output_file_name}")
        else:
            print("Error: Please provide data_filename and product_name arguments, or use --process_all")
            print("Example: python -m validation.frm4veg_validation -f FRM_Veg_Barrax_20210719_V2 -p SENTINEL2B_20210722-111020-007_L2A_T30SWJ_C_V3-0")
            print("Or: python -m validation.frm4veg_validation --process_all")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()