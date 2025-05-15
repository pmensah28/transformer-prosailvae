#!/usr/bin/env python
# coding:utf-8

"""
A script to validate Transformer-VAE models on multiple datasets
(BelSAR, FRM4VEG) and create individual or combined validation plots.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
import traceback
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import validation utilities
from validation.validation import (get_all_campaign_lai_results, 
                                  get_frm4veg_ccc_results,
                                  get_belsar_x_frm4veg_lai_results,
                                  get_validation_global_metrics)
from validation.belsar_validation import (save_belsar_predictions, 
                                         interpolate_belsar_metrics,
                                         BELSAR_FILENAMES)
from validation.frm4veg_validation import (get_frm4veg_material, 
                                          get_model_frm4veg_results,
                                          load_frm4veg_data,
                                          BARRAX_FILENAMES, 
                                          WYTHAM_FILENAMES, 
                                          BARRAX_2021_FILENAME)
from metrics.prosail_plots import regression_plot_2hues, patch_validation_reg_scatter_plot
from utils.utils import load_dict, StandardizeCoeff, IOStandardizeCoeffs
from prosailvae.prosail_vae import load_prosail_vae_with_hyperprior, get_prosail_vae_config
from prosailvae.ProsailSimus import get_bands_idx

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Validation")

# Set matplotlib parameters
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.7,
})

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Validation script for Transformer-VAE models on multiple datasets"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model file or directory",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base directory for validation data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="validation_results/",
        help="Directory to save validation results and plots",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference (cuda or cpu)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", "belsar", "frm4veg", "frm4veg_barrax2018", "frm4veg_barrax2021", "frm4veg_wytham2018"],
        help="Datasets to validate on",
    )
    parser.add_argument(
        "--combine_results",
        action="store_true",
        help="Generate combined validation plots across all datasets",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="closest",
        choices=["simple_interpolate", "closest", "best", "worst"],
        help="Method to use for interpolation (BelSAR data)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="sim_tg_mean",
        choices=["sim_tg_mean", "lat_mode", "sim_mode", "lat_mean", "sim_mean"],
        help="Mode to use for predictions",
    )
    parser.add_argument(
        "--reconstruction",
        action="store_true",
        help="Save reconstruction errors",
    )
    parser.add_argument(
        "--export_plots",
        action="store_true",
        help="Export plots as PDF for publication",
    )
    parser.add_argument(
        "--rsr_dir",
        type=str,
        default="rsr_data",
        help="Directory containing RSR (Relative Spectral Response) files",
    )
    parser.add_argument(
        "--crop_type",
        type=str,
        default="all",
        choices=["all", "wheat", "maize"],
        help="Crop type filter for BelSAR validation",
    )
    parser.add_argument(
        "--export_detailed",
        action="store_true",
        help="Export detailed sample-by-sample CSV files",
    )
    parser.add_argument(
        "--comparison_plot",
        action="store_true",
        help="Generate side-by-side comparison plots for multiple variables",
    )
    
    return parser.parse_args() 

def load_model(model_path, device="cpu", rsr_dir="rsr_data"):
    """Load the model from a checkpoint file or directory."""
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Check if model_path is a directory or a file
        if os.path.isdir(model_path):
            model_weights = os.path.join(model_path, "prosailvae_weights.tar")
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(model_weights):
                raise FileNotFoundError(f"Weights file {model_weights} does not exist")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file {config_path} does not exist")
                
            # Load model config
            params = load_dict(config_path)
            logger.info(f"Loaded config from {config_path}")
            
            # Load standardization coefficients
            bands_loc = torch.load(os.path.join(model_path, "norm_mean.pt"), map_location=device)
            bands_scale = torch.load(os.path.join(model_path, "norm_std.pt"), map_location=device)
            idx_loc = torch.load(os.path.join(model_path, "idx_loc.pt"), map_location=device)
            idx_scale = torch.load(os.path.join(model_path, "idx_scale.pt"), map_location=device)
            angles_loc = torch.load(os.path.join(model_path, "angles_loc.pt"), map_location=device)
            angles_scale = torch.load(os.path.join(model_path, "angles_scale.pt"), map_location=device)

            bands = StandardizeCoeff(loc=bands_loc, scale=bands_scale)
            idx = StandardizeCoeff(loc=idx_loc, scale=idx_scale)
            angles = StandardizeCoeff(loc=angles_loc, scale=angles_scale)
            io_coeffs = IOStandardizeCoeffs(bands=bands, idx=idx, angles=angles)

            # For loading the trained model
            params["load_model"] = True
            params["vae_load_file_path"] = model_weights
            
            # Get bands
            band_indices, prosail_bands = get_bands_idx(params.get("weiss_bands", False))

            # Build prosail config
            pv_config = get_prosail_vae_config(
                params=params, 
                bands=band_indices,
                prosail_bands=prosail_bands,
                io_coeffs=io_coeffs,
                inference_mode=True,
                rsr_dir=rsr_dir
            )
            
            # Load the VAE
            logger.info(f"Loading Transformer-VAE from {model_weights}")
            model = load_prosail_vae_with_hyperprior(
                pv_config=pv_config, 
                pv_config_hyper=None,
                logger_name="Validation"
            )
        else:
            # Handle .tar files that contain state_dict rather than the full model
            logger.info(f"Loading model from {model_path}")
            
            # Try to find config nearby to create proper model
            possible_config = os.path.join(os.path.dirname(model_path), "config.json")
            if os.path.exists(possible_config):
                params = load_dict(possible_config)
                logger.info(f"Found and loaded config from {possible_config}")
                
                try:
                    # Load standardization coefficients from same directory
                    model_dir = os.path.dirname(model_path)
                    bands_loc = torch.load(os.path.join(model_dir, "norm_mean.pt"), map_location=device)
                    bands_scale = torch.load(os.path.join(model_dir, "norm_std.pt"), map_location=device)
                    idx_loc = torch.load(os.path.join(model_dir, "idx_loc.pt"), map_location=device)
                    idx_scale = torch.load(os.path.join(model_dir, "idx_scale.pt"), map_location=device)
                    angles_loc = torch.load(os.path.join(model_dir, "angles_loc.pt"), map_location=device)
                    angles_scale = torch.load(os.path.join(model_dir, "angles_scale.pt"), map_location=device)
                    
                    bands = StandardizeCoeff(loc=bands_loc, scale=bands_scale)
                    idx = StandardizeCoeff(loc=idx_loc, scale=idx_scale)
                    angles = StandardizeCoeff(loc=angles_loc, scale=angles_scale)
                    io_coeffs = IOStandardizeCoeffs(bands=bands, idx=idx, angles=angles)
                except Exception as e:
                    logger.warning(f"Could not load standardization coefficients: {e}")
                    logger.warning("Continuing with default coefficients")
                    io_coeffs = None
                
                # Get bands
                band_indices, prosail_bands = get_bands_idx(params.get("weiss_bands", False))
                
                # Add load model params
                params["load_model"] = True
                params["vae_load_file_path"] = model_path
                
                # Build prosail config
                pv_config = get_prosail_vae_config(
                    params=params,
                    bands=band_indices,
                    prosail_bands=prosail_bands,
                    io_coeffs=io_coeffs,
                    inference_mode=True,
                    rsr_dir=rsr_dir
                )
                
                # Load the model structure first
                model = load_prosail_vae_with_hyperprior(
                    pv_config=pv_config,
                    pv_config_hyper=None,
                    logger_name="Validation"
                )
                
                # Then load the state dict
                checkpoint = torch.load(model_path, map_location=device)
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        # Assume it's the direct state dict
                        model.load_state_dict(checkpoint)
                else:
                    # This is a full model object
                    model = checkpoint
            else:
                # No config file found, try to load directly
                logger.warning("No config file found near model. Attempting direct load.")
                checkpoint = torch.load(model_path, map_location=device)
                if isinstance(checkpoint, dict) and ('model' in checkpoint):
                    model = checkpoint['model']
                else:
                    model = checkpoint
                    
                if not hasattr(model, 'forward'):
                    raise ValueError("Loaded object does not appear to be a PyTorch model")
        
        # Move model to device and set to evaluation mode
        model = model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully (on {device}).")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())
        raise 

def calculate_metrics(y_true, y_pred):
    """Calculate and return common regression metrics"""
    try:
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        r2 = np.corrcoef(y_true, y_pred)[0, 1]**2 if len(y_true) > 1 else 0
        logger.debug(f"Calculated metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        return mae, rmse, r2
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise

def plot_validation_scatter(
    y_true, 
    y_pred, 
    y_err=None,
    x_label="In-situ", 
    y_label="Predicted", 
    variable="", 
    site="", 
    figsize=(10, 8), 
    color_by_value=True,
    output_path=None
):
    """
    Create a scatter plot with a 1:1 line and error metrics displayed
    """
    logger.info(f"Creating validation scatter plot for {variable}")
    
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    if y_err is not None:
        y_err = np.array(y_err).flatten()
    
    mae, rmse, r2 = calculate_metrics(y_true, y_pred)
    m, b = np.polyfit(y_true, y_pred, 1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Optional error bars
    if y_err is not None:
        for i in range(len(y_true)):
            # Vertical error bars
            ax.plot(
                [y_true[i], y_true[i]],
                [y_pred[i] - y_err[i], y_pred[i] + y_err[i]],
                color='gray', alpha=0.5, linewidth=1
            )
            
            # Horizontal error bars
            ax.plot(
                [y_true[i] - y_err[i], y_true[i] + y_err[i]],
                [y_pred[i], y_pred[i]],
                color='gray', alpha=0.5, linewidth=1
            )
    
    # Scatter points with optional color
    if color_by_value:
        scatter = ax.scatter(
            y_true, y_pred, c=y_true, cmap='viridis', s=60, alpha=0.8,
            edgecolor='white', linewidth=0.5
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(f"{x_label} {variable}")
    else:
        ax.scatter(y_true, y_pred, color='#5b9bd5', s=60, alpha=0.8,
                   edgecolor='white', linewidth=0.5)
    
    # 1:1 line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    margin = (max_val - min_val) * 0.05
    lims = [min_val - margin, max_val + margin]
    ax.plot(lims, lims, 'k--', alpha=0.8, zorder=0, label="1:1 Line")
    
    # Regression line
    ax.plot(lims, [m*lims[0] + b, m*lims[1] + b], 'r-', linewidth=1.5,
            label=f'y = {m:.2f}x + {b:.2f}')
    
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # Metrics text
    metrics_text = f"N = {len(y_true)}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    ax.grid(True)
    ax.set_xlabel(f"{x_label} {variable}")
    ax.set_ylabel(f"{y_label} {variable}")
    ax.set_title(f"{site} - {variable} Validation")
    ax.legend(loc='lower right')
    fig.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved validation plot to {output_path}")
    
    logger.info(f"{variable} metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    return fig, ax, (mae, rmse, r2)

def export_detailed_validation_results(y_true, y_pred, y_err, site, var_key, output_dir):
    """
    Export detailed sample-by-sample validation results to CSV.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        y_err: Error/uncertainty values (can be None)
        site: Site name for the data
        var_key: Variable name (e.g., "lai", "ccc")
        output_dir: Directory to save the CSV file
    
    Returns:
        Path to the saved CSV file
    """
    detailed_data = []
    
    # Calculate global PICP and MPIW if uncertainty estimates are available
    global_picp = None
    global_mpiw = None
    if y_err is not None:
        n_sigma = 2  # Use 2-sigma for 95% confidence interval
        # Calculate global PICP
        in_interval = np.logical_and(
            y_true < y_pred + n_sigma * y_err,
            y_true > y_pred - n_sigma * y_err
        ).astype(int)
        global_picp = np.mean(in_interval)
        
        # Calculate global MPIW
        global_mpiw = np.mean(2 * n_sigma * y_err)
    
    for i in range(len(y_true)):
        row = {
            'Site': site,
            'Variable': var_key,
            'Ground_Truth_Value': y_true[i],
            'Predicted_Value': y_pred[i],
            'Absolute_Error': abs(y_true[i] - y_pred[i]),
            'Relative_Error': abs(y_true[i] - y_pred[i]) / y_true[i] if y_true[i] != 0 else np.nan
        }
        if y_err is not None:
            row['Uncertainty'] = y_err[i]
            row['Prediction_Interval_Width'] = 2 * n_sigma * y_err[i]
            row['Within_Interval'] = 1 if (y_true[i] <= y_pred[i] + n_sigma * y_err[i] and 
                                           y_true[i] >= y_pred[i] - n_sigma * y_err[i]) else 0
        detailed_data.append(row)
    
    # Save detailed data for this variable
    var_df = pd.DataFrame(detailed_data)
    var_csv_path = os.path.join(output_dir, f"{site}_{var_key}_detailed.csv")
    var_df.to_csv(var_csv_path, index=False)
    
    # Add a summary row with global metrics
    summary_file = os.path.join(output_dir, f"{site}_{var_key}_summary.csv")
    mae, rmse, r2 = calculate_metrics(y_true, y_pred)
    summary_data = {
        'Site': site,
        'Variable': var_key,
        'Samples': len(y_true),
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
    if global_picp is not None:
        summary_data['PICP'] = global_picp
    if global_mpiw is not None:
        summary_data['MPIW'] = global_mpiw
    
    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(summary_file, index=False)
    
    logger.info(f"Exported detailed results for {site}_{var_key} to {var_csv_path}")
    logger.info(f"Exported summary metrics for {site}_{var_key} to {summary_file}")
    
    return var_csv_path

def export_validation_results_to_csv(results_dict, validation_output_dir, export_detailed=False):
    """Export validation results to CSV for further analysis"""
    if not results_dict:
        logger.warning("No results available to export")
        return None
    
    # Prepare data for CSV
    csv_data = []
    for var_key, result in results_dict.items():
        y_true = result['y_true']
        y_pred = result['y_pred']
        y_err = result.get('y_err', None)
        site = result.get('site', 'Unknown')
        
        # Calculate metrics
        mae, rmse, r2 = calculate_metrics(y_true, y_pred)
        
        # Calculate PICP and MPIW if uncertainty estimates are available
        picp = None
        mpiw = None
        if y_err is not None:
            n_sigma = 2  # Use 2-sigma for 95% confidence interval
            # Calculate PICP
            in_interval = np.logical_and(
                y_true < y_pred + n_sigma * y_err,
                y_true > y_pred - n_sigma * y_err
            ).astype(int)
            picp = np.mean(in_interval)
            
            # Calculate MPIW
            mpiw = np.mean(2 * n_sigma * y_err)
        
        # Create row for summary table
        row_data = {
            'Site': site,
            'Variable': var_key,
            'Samples': len(y_true),
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Min_True': np.min(y_true),
            'Max_True': np.max(y_true),
            'Mean_True': np.mean(y_true),
            'StdDev_True': np.std(y_true),
            'Min_Pred': np.min(y_pred),
            'Max_Pred': np.max(y_pred),
            'Mean_Pred': np.mean(y_pred),
            'StdDev_Pred': np.std(y_pred)
        }
        
        # Add PICP and MPIW if available
        if picp is not None:
            row_data['PICP'] = picp
        if mpiw is not None:
            row_data['MPIW'] = mpiw
        
        csv_data.append(row_data)
        
        # Export detailed results if requested
        if export_detailed:
            export_detailed_validation_results(
                y_true=y_true,
                y_pred=y_pred,
                y_err=y_err,
                site=site,
                var_key=var_key,
                output_dir=validation_output_dir
            )
    
    # Save summary table
    summary_df = pd.DataFrame(csv_data)
    summary_csv_path = os.path.join(validation_output_dir, "validation_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Exported validation summary to {summary_csv_path}")
    
    return summary_csv_path

def get_frm4veg_validation_data_paths():
    """Get paths to validation data for different FRM4VEG sites and years"""
    
    validation_data = {
        "frm4veg_barrax2018": {
            "data_dir": "data/frm4veg_validation/Barrax/GroundData/Las_Tiesas_Barrax_2018",
            "filename": "2A_20180613_FRM_Veg_Barrax_20180605_V2",
            "display_name": "Barrax 2018"
        },
        "frm4veg_barrax2021": {
            "data_dir": "data/frm4veg_validation/Barrax/GroundData/Las_Tiesas_Barrax_2021",
            "filename": "2B_20210722_FRM_Veg_Barrax_20210719_V2",
            "display_name": "Barrax 2021"
        },
        "frm4veg_wytham2018": {
            "data_dir": "data/frm4veg_validation/Wytham/Wytham_Woods_2018",
            "filename": "2A_20180629_FRM_Veg_Wytham_20180703_V2",
            "display_name": "Wytham Woods 2018"
        }
    }
    
    return validation_data

def custom_get_frm4veg_results(model, frm4veg_data_dir, filename, mode, save_reconstruction):
    """
    Custom function to get FRM4VEG results with better handling of missing 'date' column.
    
    Args:
        model: The trained model
        frm4veg_data_dir: Directory containing FRM4VEG data
        filename: Name of the file to process
        mode: Prediction mode
        save_reconstruction: Whether to save reconstruction errors
        
    Returns:
        Dictionary of validation results
    """
    try:
        # Check if this is Barrax 2018 data
        is_barrax_2018 = "20180613_FRM_Veg_Barrax" in filename or "2018" in filename and "Barrax" in frm4veg_data_dir

        # Get materials, filtering alfalfa if this is Barrax 2018
        sensor = filename.split("_")[0]
        
        site_idx_dict = {}
        ref_dict = {}
        
        # Special handling for Barrax 2018 to filter out alfalfa measurements
        if is_barrax_2018:
            logger.info("Processing Barrax 2018 data - filtering out alfalfa measurements")
            
            # Process each variable separately to filter out alfalfa
            for variable in ['lai', 'lai_eff', 'ccc', 'ccc_eff']:
                gdf, _, _, _, _ = load_frm4veg_data(frm4veg_data_dir, filename, variable=variable)
                
                # Filter out alfalfa measurements for Barrax 2018
                if "land cover" in gdf.columns:
                    before_filter = len(gdf)
                    gdf = gdf[gdf["land cover"].str.lower() != "alfalfa"]
                    after_filter = len(gdf)
                    
                    # Log how many samples were filtered
                    if before_filter > after_filter:
                        logger.info(f"Filtered out {before_filter - after_filter} alfalfa measurements for {variable}")
                
                # Store the filtered data
                ref_dict[variable] = gdf[variable].values.reshape(-1)
                ref_dict[variable+"_std"] = gdf["uncertainty"].values.reshape(-1)
                site_idx_dict[variable] = {"x_idx": torch.from_numpy(gdf["x_idx"].values).int(),
                                           "y_idx": torch.from_numpy(gdf["y_idx"].values).int()}
            
            # Load reflectance and angle data - they don't need filtering
            gdf, s2_r, s2_a, _, _ = load_frm4veg_data(frm4veg_data_dir, filename, variable="lai")
            s2_r = torch.from_numpy(s2_r).float().unsqueeze(0)
            s2_a = torch.from_numpy(s2_a).float().unsqueeze(0)
        else:
            # Standard loading procedure for other datasets
            (s2_r, s2_a, site_idx_dict, ref_dict) = get_frm4veg_material(frm4veg_data_dir, filename)
        
        # Get model results
        validation_results = get_model_frm4veg_results(
            model, s2_r, s2_a, site_idx_dict, 
            ref_dict, mode=mode, 
            get_reconstruction=save_reconstruction
        )
        
        # Try to parse date from filename
        try:
            d = datetime.strptime(filename.split("_")[1], '%Y%m%d').date()
        except (ValueError, IndexError):
            # If we can't parse the date, use current date
            logger.warning(f"Could not parse date from filename {filename}, using current date")
            d = datetime.now().date()
        
        # Handle land cover and dates for each variable
        for variable in ["lai", "lai_eff", "ccc", "ccc_eff"]:
            try:
                gdf, _, _, _, _ = load_frm4veg_data(frm4veg_data_dir, filename, variable=variable)
                
                # Filter out alfalfa for Barrax 2018
                if is_barrax_2018 and "land cover" in gdf.columns:
                    gdf = gdf[gdf["land cover"].str.lower() != "alfalfa"]
                
                # Set land cover
                if "land cover" in gdf.columns:
                    validation_results[f"{variable}_land_cover"] = gdf["land cover"].values
                else:
                    # Use default if land cover is missing
                    n_samples = len(validation_results.get(f'ref_{variable}', []))
                    validation_results[f"{variable}_land_cover"] = np.array(['Unknown'] * n_samples)
                    logger.warning(f"Missing 'land cover' column in {filename} for {variable}. Using default value.")
                
                # Handle date column with better error messages
                if 'date' in gdf.columns:
                    try:
                        validation_results[f"{variable}_date"] = gdf["date"].apply(lambda x: (x.date() - d).days).values
                    except Exception as date_error:
                        logger.warning(f"Error processing 'date' column for {variable}: {date_error}")
                        n_samples = len(validation_results.get(f'ref_{variable}', []))
                        validation_results[f"{variable}_date"] = np.zeros(n_samples, dtype=int)
                else:
                    # Use zero as default (same date as satellite image)
                    logger.warning(f"Missing 'date' column in {filename} for {variable}. Using default value.")
                    n_samples = len(validation_results.get(f'ref_{variable}', []))
                    validation_results[f"{variable}_date"] = np.zeros(n_samples, dtype=int)
            except Exception as e:
                logger.warning(f"Error processing {variable} for {filename}: {e}")
                # Set some defaults if we couldn't load this variable
                if f'ref_{variable}' in validation_results:
                    n_samples = len(validation_results[f'ref_{variable}'])
                    validation_results[f"{variable}_land_cover"] = np.array(['Unknown'] * n_samples)
                    validation_results[f"{variable}_date"] = np.zeros(n_samples, dtype=int)
        
        return validation_results
    except Exception as e:
        logger.error(f"Error in custom_get_frm4veg_results for {filename}: {e}")
        traceback.print_exc()
        raise

def run_frm4veg_validation(model, data_dirs, output_dir, mode="sim_tg_mean", save_reconstruction=False, export_detailed=False, export_plots=False):
    """
    Run validation on FRM4VEG datasets.
    
    Args:
        model: Trained model to validate
        data_dirs: Dictionary of FRM4VEG data directories by site
        output_dir: Directory to save validation results
        mode: Prediction mode ("sim_tg_mean", "lat_mode", etc.)
        save_reconstruction: Whether to save reconstruction error
        export_detailed: Whether to export detailed CSV files
        export_plots: Whether to export PDF plots for publication
        
    Returns:
        Dictionary of validation results by site
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get validation data paths
        validation_data = get_frm4veg_validation_data_paths()
        site_results = {}
        
        # Process each site
        for site_key, site_data in validation_data.items():
            # Override data directory if provided
            site_data_dir = data_dirs.get(site_key, site_data["data_dir"])
            
            # Extract data_dir string if site_data_dir is a dictionary
            if isinstance(site_data_dir, dict) and "data_dir" in site_data_dir:
                site_data_dir = site_data_dir["data_dir"]
                
            filename = site_data["filename"]
            display_name = site_data["display_name"]
            
            # Create site-specific output directory
            site_output_dir = os.path.join(output_dir, site_key)
            os.makedirs(site_output_dir, exist_ok=True)
            
            logger.info(f"Running FRM4VEG validation for {display_name}")
            logger.info(f"Data directory: {site_data_dir}")
            logger.info(f"Filename: {filename}")
            logger.info(f"Mode: {mode}")
                
            try:
                # Get FRM4VEG validation results
                validation_results = custom_get_frm4veg_results(
                    model=model,
                    frm4veg_data_dir=site_data_dir,
                    filename=filename,
                    mode=mode,
                    save_reconstruction=save_reconstruction
                )
                
                # Store results for this site
                site_results[site_key] = validation_results
                
                # Prepare variables for validation
                variables = [
                    ("lai", "LAI"),
                    ("lai_eff", "LAI_EFF"),
                    ("ccc", "CCC"),
                    ("ccc_eff", "CCC_EFF")
                ]
                
                # Generate validation plots for each variable
                for var_key, var_display in variables:
                    if var_key not in validation_results or f"ref_{var_key}" not in validation_results:
                        logger.warning(f"Variable '{var_key}' missing in reference or predictions for {site_key}")
                        continue
                    
                    y_true = np.array(validation_results[f"ref_{var_key}"])
                    y_pred = np.array(validation_results[var_key])
                    y_err = np.array(validation_results[f"ref_{var_key}_std"]) if f"ref_{var_key}_std" in validation_results else None
                    
                    # Create and save plot
                    output_path = os.path.join(site_output_dir, f"{site_key}_{var_key}_validation.png")
                    pdf_path = os.path.join(site_output_dir, f"{site_key}_{var_key}_validation.pdf")
                    
                    fig, ax, metrics = plot_validation_scatter(
                        y_true=y_true,
                        y_pred=y_pred,
                        y_err=y_err,
                        variable=var_display,
                        site=display_name,
                        color_by_value=True,
                        output_path=output_path
                    )
                    
                    # Also save as PDF for publication quality
                    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
                    plt.close(fig)
                    
                    logger.info(f"Generated {var_key} validation plot for {site_key}")
                
                # Export results to CSV for this site
                combined_results = {}
                for var_key, var_display in variables:
                    if var_key not in validation_results or f"ref_{var_key}" not in validation_results:
                        continue
                        
                    combined_results[var_key] = {
                        'y_true': validation_results[f"ref_{var_key}"],
                        'y_pred': validation_results[var_key],
                        'y_err': validation_results.get(f"ref_{var_key}_std", None),
                        'site': site_key
                    }
                
                # Export results to CSV
                if export_detailed:
                    export_validation_results_to_csv(combined_results, site_output_dir, export_detailed=True)
                else:
                    export_validation_results_to_csv(combined_results, site_output_dir)
                
                logger.info(f"Completed validation for {display_name}")
                
            except Exception as e:
                logger.error(f"Error validating {display_name}: {e}")
                logger.error(traceback.format_exc())
        
        logger.info("FRM4VEG validation completed successfully")
        return site_results
        
    except Exception as e:
        logger.error(f"Error during FRM4VEG validation: {e}")
        logger.error(traceback.format_exc())
        return None

def run_belsar_validation(model, data_dir, output_dir, model_name="pvae", mode="lat_mode", method="closest", crop_type="all", save_reconstruction=False, export_detailed=False, export_plots=False):
    """
    Run validation on BelSAR dataset.
    
    Args:
        model: The trained model
        data_dir: Directory containing BelSAR data
        output_dir: Directory to save validation results
        model_name: Name to use for the model
        mode: Prediction mode
        method: Interpolation method
        crop_type: Crop type to filter by
        save_reconstruction: Whether to save reconstruction errors
        export_detailed: Whether to export detailed results
        export_plots: Whether to export publication-quality plots
        
    Returns:
        Tuple containing validation results dataframe and summary dictionary
    """
    try:
        # Create predictions directory
        pred_dir = os.path.join(output_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        logger.info(f"Saving predictions to {pred_dir}")
        
        # Generate predictions
        logger.info(f"Generating BelSAR predictions using mode={mode}...")
        save_belsar_predictions(
            belsar_dir=data_dir,
            model=model,
            res_dir=pred_dir,
            list_filenames=BELSAR_FILENAMES,
            model_name=model_name,
            mode=mode,
            save_reconstruction=save_reconstruction
        )
        
        # Calculate metrics using the specified approach
        logger.info(f"Calculating metrics with '{method}' approach...")
        file_suffix = f"_{model_name}_{mode}"
        metrics = interpolate_belsar_metrics(
            belsar_data_dir=data_dir,
            belsar_pred_dir=pred_dir,
            method=method,
            file_suffix=file_suffix,
            get_error=save_reconstruction,
            bands_idx=model.encoder.bands if hasattr(model, 'encoder') and hasattr(model.encoder, 'bands') else None
        )
        
        # Save metrics to CSV
        metrics_file = os.path.join(output_dir, f"belsar_metrics_{model_name}_{mode}.csv")
        metrics.to_csv(metrics_file)
        logger.info(f"Metrics saved to {metrics_file}")
        
        # Create result summaries
        summary = {}
        combined_results = {}
        
        # Process results for all crops and crop-specific filters
        for current_crop in ["all", "wheat", "maize"]:
            # Skip crops that don't match the filter if a specific crop type was requested
            if crop_type != "all" and current_crop != "all" and current_crop != crop_type:
                continue
            
            # Create crop-specific output directory
            crop_output_dir = os.path.join(output_dir, current_crop)
            os.makedirs(crop_output_dir, exist_ok=True)
            
            # Filter metrics for the current crop
            if current_crop == "all":
                crop_metrics = metrics
            else:
                crop_metrics = metrics[metrics["land_cover"].str.lower() == current_crop]
            
            # Skip if no samples for this crop
            if len(crop_metrics) == 0:
                logger.warning(f"No samples found for crop type: {current_crop}")
                continue
            
            # Calculate crop-specific metrics
            crop_rmse = ((crop_metrics["lai_mean"] - crop_metrics["ref_lai"])**2).mean()**0.5
            crop_mae = (crop_metrics["lai_mean"] - crop_metrics["ref_lai"]).abs().mean()
            # Calculate R² with protection against divide by zero
            if crop_metrics["ref_lai"].std() == 0:
                crop_r2 = 0  # R² is undefined when all reference values are identical
            else:
                crop_r2 = 1 - ((crop_metrics["lai_mean"] - crop_metrics["ref_lai"])**2).sum() / \
                          ((crop_metrics["ref_lai"] - crop_metrics["ref_lai"].mean())**2).sum()
            
            # Store crop-specific results
            summary[f"lai_{current_crop}"] = {
                "n_samples": len(crop_metrics),
                "RMSE": float(crop_rmse),
                "MAE": float(crop_mae),
                "R²": float(crop_r2)
            }
            
            # Store data for comprehensive CSV export
            combined_results[f"lai_{current_crop}"] = {
                'y_true': crop_metrics["ref_lai"].values,
                'y_pred': crop_metrics["lai_mean"].values,
                'y_err': crop_metrics["lai_sigma_mean"].values if "lai_sigma_mean" in crop_metrics.columns else None,
                'site': f'BelSAR_{current_crop.capitalize()}'
            }
            
            # Generate crop-specific validation plot
            logger.info(f"Creating validation scatter plot for LAI ({current_crop})")
            
            crop_plot_path = os.path.join(crop_output_dir, f"lai_regression_{current_crop}_{model_name}_{mode}.png")
            crop_pdf_path = os.path.join(crop_output_dir, f"lai_regression_{current_crop}_{model_name}_{mode}.pdf")
            
            # Create the validation scatter plot
            fig, ax, (mae, rmse, r2) = plot_validation_scatter(
                y_true=crop_metrics["ref_lai"],
                y_pred=crop_metrics["lai_mean"],
                y_err=crop_metrics["lai_sigma_mean"] if "lai_sigma_mean" in crop_metrics.columns else None,
                x_label="In-situ",
                y_label="Predicted",
                variable="LAI",
                site=f"BelSAR - {current_crop.capitalize()}",
                color_by_value=True,
                output_path=crop_plot_path
            )
            
            # Save as PDF for publication quality if requested
            if export_plots:
                fig.savefig(crop_pdf_path, format='pdf', bbox_inches='tight')
                logger.info(f"Saved PDF plot to {crop_pdf_path}")
            
            plt.close(fig)
            
            # Update the summary with file paths
            summary[f"lai_{current_crop}"]["output_file"] = crop_plot_path
            if export_plots:
                summary[f"lai_{current_crop}"]["pdf_file"] = crop_pdf_path
            
            # Save crop-specific metrics to CSV
            crop_metrics_file = os.path.join(crop_output_dir, f"belsar_metrics_{current_crop}_{model_name}_{mode}.csv")
            crop_metrics.to_csv(crop_metrics_file)
            logger.info(f"Saved {current_crop} metrics to {crop_metrics_file}")
            
            # Export detailed CSV files if requested
            if export_detailed:
                # Create detailed results dict
                detailed_results = {
                    f"lai_{current_crop}": {
                        'y_true': crop_metrics["ref_lai"].values,
                        'y_pred': crop_metrics["lai_mean"].values,
                        'y_err': crop_metrics["lai_sigma_mean"].values if "lai_sigma_mean" in crop_metrics.columns else None,
                        'site': f'BelSAR_{current_crop.capitalize()}'
                    }
                }
                
                # Export detailed CSV
                export_validation_results_to_csv(detailed_results, crop_output_dir, export_detailed=True)
                logger.info(f"Exported detailed validation results for {current_crop}")
        
        # Export all validation results to CSV
        logger.info("Exporting detailed validation results to CSV...")
        export_validation_results_to_csv(combined_results, output_dir, export_detailed=export_detailed)
        logger.info(f"Exported comprehensive validation results to {output_dir}")
        
        # Save summary text file
        summary_path = os.path.join(output_dir, f"belsar_validation_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"=== Transformer-VAE BelSAR Validation Results ===\n\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Mode: {mode}\n\n")
            
            # Write summary for each crop type
            for crop in ["all", "wheat", "maize"]:
                if f"lai_{crop}" in summary:
                    f.write(f"{crop.capitalize()} LAI Validation:\n")
                    f.write(f"  Samples: {summary[f'lai_{crop}']['n_samples']}\n")
                    f.write(f"  MAE: {summary[f'lai_{crop}']['MAE']:.4f}\n")
                    f.write(f"  RMSE: {summary[f'lai_{crop}']['RMSE']:.4f}\n")
                    f.write(f"  R²: {summary[f'lai_{crop}']['R²']:.4f}\n\n")
        
        logger.info(f"Validation summary saved to {summary_path}")
        logger.info("BelSAR validation completed successfully")
        
        return metrics, summary
        
    except Exception as e:
        logger.error(f"Error during BelSAR validation: {e}")
        logger.error(traceback.format_exc())
        return None, None

def create_comparison_plot(results_dict, variables, output_dir, timestamp=None):
    """
    Create a multi-variable comparison plot for different datasets.
    
    Args:
        results_dict: Dictionary where keys are variable names and values are DataFrames
                     containing the validation results.
        variables: List of variable names to plot
        output_dir: Directory to save the plot
        timestamp: Optional timestamp for the file name
    
    Returns:
        Path to the saved plot
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    n_vars = len(variables)
    if n_vars == 0:
        logger.warning("No variables provided for comparison plot")
        return None
    
    # Create a single figure with two subplots side by side
    fig, axes = plt.subplots(1, n_vars, figsize=(n_vars*8, 7), dpi=300)
    if n_vars == 1:
        axes = [axes]  # Make it iterable
    
    # Define units for variables
    variable_units = {
        "lai": "",  # LAI is unitless
        "ccc": r"$\mu\mathrm{g}\,\mathrm{cm}^{-2}$"
    }
    
    # Store all legend handles and labels
    all_handles = []
    all_labels = []
    land_cover_handles = []
    land_cover_labels = []
    site_handles = []
    site_labels = []
    
    # Process each variable
    for i, var in enumerate(variables):
        if var not in results_dict:
            logger.warning(f"Variable {var} not found in results")
            continue
        
        data = results_dict[var]
        var_upper = var.upper()
        var_display = f"In-situ {var_upper}"
        var_pred_display = f"Predicted {var_upper}"
        
        # Calculate metrics
        r2 = np.corrcoef(data[var_display], data[var_pred_display])[0, 1]**2
        rmse = np.sqrt(np.mean((data[var_display] - data[var_pred_display])**2))
        
        # Create color dictionaries for land cover types
        if "Land cover" in data.columns:
            hue_elements = pd.unique(data["Land cover"])
            hue_color_dict = {h_e: f"C{j}" for j, h_e in enumerate(hue_elements)}
        else:
            hue_color_dict = None
        
        # Create marker dictionaries for campaigns
        if "Campaign" in data.columns:
            hue2_elements = pd.unique(data["Campaign"])
            default_markers = ["o", "v", "D", "s", "+", "^", "1", "."]
            hue2_markers_dict = {h2_e: default_markers[j % len(default_markers)] 
                                for j, h2_e in enumerate(hue2_elements)}
        else:
            hue2_markers_dict = None
        
        # Get axis title with proper units
        axis_title = f"{var_upper} Validation"
        
        # Plot the data but capture handles and labels
        scatter_handles, scatter_labels = [], []
        
        # Calculate axes limits with some padding
        xmin, xmax = data[var_display].min(), data[var_display].max()
        ymin, ymax = data[var_pred_display].min(), data[var_pred_display].max()
        
        # Add some padding (10%)
        xrange = xmax - xmin
        yrange = ymax - ymin
        xmin = max(0, xmin - 0.1 * xrange)
        xmax = xmax + 0.1 * xrange
        ymin = max(0, ymin - 0.1 * yrange)
        ymax = ymax + 0.1 * yrange
        
        # Use the same limits for both axes
        plot_min = min(xmin, ymin)
        plot_max = max(xmax, ymax)
        
        # Plot 1:1 line
        axes[i].plot([plot_min, plot_max], [plot_min, plot_max], 'k-', alpha=0.8)
        
        # Plot regression line
        m, b = np.polyfit(data[var_display], data[var_pred_display], 1)
        axes[i].plot([plot_min, plot_max], [m*plot_min + b, m*plot_max + b], 'r-', alpha=0.8)
        
        # Plot points by Land cover and Campaign
        for land_cover in pd.unique(data["Land cover"]):
            land_cover_data = data[data["Land cover"] == land_cover]
            
            for campaign in pd.unique(land_cover_data["Campaign"]):
                campaign_data = land_cover_data[land_cover_data["Campaign"] == campaign]
                
                color = hue_color_dict.get(land_cover, "gray")
                marker = hue2_markers_dict.get(campaign, "o")
                
                # Plot points
                h = axes[i].scatter(
                    campaign_data[var_display], 
                    campaign_data[var_pred_display],
                    c=color, 
                    marker=marker,
                    s=50,
                    alpha=0.8,
                    label=f"{land_cover} - {campaign}"
                )
                
                # Plot error bars if available
                if f'{var} std' in campaign_data.columns and f'{var_pred_display} std' in campaign_data.columns:
                    for idx, row in campaign_data.iterrows():
                        # X error bars
                        if not np.isnan(row[f'{var} std']):
                            axes[i].plot(
                                [row[var_display] - row[f'{var} std'], row[var_display] + row[f'{var} std']],
                                [row[var_pred_display], row[var_pred_display]],
                                color='gray', alpha=0.3, linewidth=0.5
                            )
                        
                        # Y error bars
                        if not np.isnan(row[f'{var_pred_display} std']):
                            axes[i].plot(
                                [row[var_display], row[var_display]],
                                [row[var_pred_display] - row[f'{var_pred_display} std'], 
                                 row[var_pred_display] + row[f'{var_pred_display} std']],
                                color='gray', alpha=0.3, linewidth=0.5
                            )
                
                # Collect handles for land cover and site separately
                if land_cover not in land_cover_labels:
                    land_cover_handles.append(plt.Line2D([0], [0], marker='o', color=color, 
                                                         linestyle='None', markersize=8))
                    land_cover_labels.append(land_cover)
                
                if campaign not in site_labels:
                    site_handles.append(plt.Line2D([0], [0], marker=marker, color='black', 
                                                   linestyle='None', markersize=8))
                    site_labels.append(campaign)
        
        # Set labels with units
        if var.lower() == "lai":
            axes[i].set_xlabel(f"In-situ {var_upper}")
            axes[i].set_ylabel(f"Transformer-VAE {var_upper}")
        else:
            axes[i].set_xlabel(f"In-situ {var_upper} {variable_units.get(var, '')}")
            axes[i].set_ylabel(f"Transformer-VAE {var_upper} {variable_units.get(var, '')}")
        
        # Set title
        axes[i].set_title(axis_title)
        
        # Add metrics text in lower right
        metrics_text = f"R²: {r2:.2f} - RMSE: {rmse:.2f}"
        if var.lower() == "ccc":
            metrics_text += r" $\mu$g cm$^{-2}$"
        
        axes[i].text(0.98, 0.05, metrics_text, transform=axes[i].transAxes, fontsize=11,
                     verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Set equal aspect ratio and limits
        axes[i].set_xlim(plot_min, plot_max)
        axes[i].set_ylim(plot_min, plot_max)
        axes[i].grid(True, alpha=0.3)
    
    # Create a single legend for both land cover and site outside the plots
    plt.figlegend(
        handles=land_cover_handles + site_handles,
        labels=['Land cover'] + land_cover_labels + ['Site'] + site_labels,
        loc='center right',
        bbox_to_anchor=(1.15, 0.5),
        title_fontsize=12,
        fontsize=10,
        frameon=True,
        ncol=1
    )
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for legend
    
    # Save the plots
    output_path = os.path.join(output_dir, f"comparison_plot_{timestamp}.png")
    pdf_output_path = os.path.join(output_dir, f"comparison_plot_{timestamp}.pdf")
    plt.savefig(output_path)
    plt.savefig(pdf_output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved comparison plot to {output_path}")
    return output_path

def run_combined_validation(frm4veg_results, belsar_metrics, output_dir, timestamp=None, include_side_by_side=True):
    """
    Generate combined validation results across all datasets.
    
    Args:
        frm4veg_results: Dictionary of FRM4VEG validation results
        belsar_metrics: BelSAR validation metrics
        output_dir: Directory to save combined results
        timestamp: Optional timestamp for filenames
        include_side_by_side: Whether to include side-by-side plots
        
    Returns:
        Dictionary containing combined results
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info("Generating combined LAI/CCC validation plots and statistics...")
        
        # Process FRM4VEG results
        barrax_results = frm4veg_results.get("frm4veg_barrax2018", {})
        barrax_2021_results = frm4veg_results.get("frm4veg_barrax2021", {})
        wytham_results = frm4veg_results.get("frm4veg_wytham2018", {})
            
        # Step 1: Combine all LAI results
        logger.info("Combining LAI results from all datasets...")
        combined_lai_results = get_belsar_x_frm4veg_lai_results(
            belsar_results=belsar_metrics, 
            barrax_results=barrax_results, 
            barrax_2021_results=barrax_2021_results, 
            wytham_results=wytham_results,
            frm4veg_lai="lai",
            get_reconstruction_error=False,
            bands_idx=None  # Will be ignored if get_reconstruction_error is False
        )
        
        # Save the combined LAI dataset
        combined_lai_path = os.path.join(output_dir, f"combined_lai_validation_{timestamp}.csv")
        combined_lai_results.to_csv(combined_lai_path)
        logger.info(f"Saved combined LAI results to {combined_lai_path}")
        
        # Step 2: Generate CCC results (only FRM4VEG datasets)
        logger.info("Combining CCC results...")
        combined_ccc_results = get_frm4veg_ccc_results(
            barrax_results=barrax_results,
            barrax_2021_results=barrax_2021_results,
            wytham_results=wytham_results,
            frm4veg_ccc="ccc",
            get_reconstruction_error=False,
            bands_idx=None  # Will be ignored if get_reconstruction_error is False
        )
        
        # Save the combined CCC dataset
        combined_ccc_path = os.path.join(output_dir, f"combined_ccc_validation_{timestamp}.csv")
        combined_ccc_results.to_csv(combined_ccc_path)
        logger.info(f"Saved combined CCC results to {combined_ccc_path}")
        
        # Step 3: Calculate global metrics for LAI
        logger.info("Calculating LAI validation metrics...")
        lai_rmse_dict, lai_picp_dict, lai_mpiw_dict, lai_mestdr_dict = get_validation_global_metrics(
            combined_lai_results, 
            decompose_along_columns=["Campaign", "Land cover", "Site"],
            n_sigma=2,
            variable="lai"
        )
        
        # Step 4: Calculate global metrics for CCC
        logger.info("Calculating CCC validation metrics...")
        ccc_rmse_dict, ccc_picp_dict, ccc_mpiw_dict, ccc_mestdr_dict = get_validation_global_metrics(
            combined_ccc_results, 
            decompose_along_columns=["Campaign", "Land cover", "Site"],
            n_sigma=2,
            variable="ccc"
        )
        
        # Save metrics summaries
        with open(os.path.join(output_dir, f"validation_metrics_summary_{timestamp}.txt"), "w") as f:
            # LAI metrics
            f.write("=== LAI Validation Metrics Summary ===\n\n")
            f.write("Overall RMSE: {:.4f}\n".format(lai_rmse_dict["Site"].iloc[0]["lai_rmse_all"]))
            f.write("Overall PICP: {:.4f}\n".format(lai_picp_dict["Site"].iloc[0]["lai_picp_all"]))
            f.write("Overall MPIW: {:.4f}\n\n".format(lai_mpiw_dict["Site"].iloc[0]["lai_mpiw_all"]))
            
            f.write("--- By Campaign ---\n")
            # Extract keys for each campaign from the metrics dictionaries
            lai_rmse_keys = [col for col in lai_rmse_dict["Campaign"].iloc[0].index 
                             if col != "lai_rmse_all" and col.startswith("lai_rmse_")]
            
            # Process each campaign with available metrics
            for campaign in pd.unique(combined_lai_results["Campaign"]):
                # Try to find matching metric key
                matching_keys = [k for k in lai_rmse_keys if k.split("lai_rmse_")[1] in campaign.lower().replace("(", "").replace(")", "").replace(" ", "_")]
                if matching_keys:
                    metric_key = matching_keys[0]
                    campaign_name = metric_key.replace("lai_rmse_", "")
                    picp_key = f"lai_picp_{campaign_name}"
                    mpiw_key = f"lai_mpiw_{campaign_name}"
                    
                    f.write(f"{campaign}:\n")
                    f.write("  RMSE: {:.4f}\n".format(lai_rmse_dict["Campaign"].iloc[0][metric_key]))
                    f.write("  PICP: {:.4f}\n".format(lai_picp_dict["Campaign"].iloc[0][picp_key]))
                    f.write("  MPIW: {:.4f}\n".format(lai_mpiw_dict["Campaign"].iloc[0][mpiw_key]))
            f.write("\n")
            
            # CCC metrics
            f.write("=== CCC Validation Metrics Summary ===\n\n")
            f.write("Overall RMSE: {:.4f}\n".format(ccc_rmse_dict["Site"].iloc[0]["ccc_rmse_all"]))
            f.write("Overall PICP: {:.4f}\n".format(ccc_picp_dict["Site"].iloc[0]["ccc_picp_all"]))
            f.write("Overall MPIW: {:.4f}\n\n".format(ccc_mpiw_dict["Site"].iloc[0]["ccc_mpiw_all"]))
            
            f.write("--- By Campaign ---\n")
            # Extract keys for each campaign from the metrics dictionaries
            ccc_rmse_keys = [col for col in ccc_rmse_dict["Campaign"].iloc[0].index 
                             if col != "ccc_rmse_all" and col.startswith("ccc_rmse_")]
            
            # Process each campaign with available metrics
            for campaign in pd.unique(combined_ccc_results["Campaign"]):
                # Try to find matching metric key
                matching_keys = [k for k in ccc_rmse_keys if k.split("ccc_rmse_")[1] in campaign.lower().replace("(", "").replace(")", "").replace(" ", "_")]
                if matching_keys:
                    metric_key = matching_keys[0]
                    campaign_name = metric_key.replace("ccc_rmse_", "")
                    picp_key = f"ccc_picp_{campaign_name}"
                    mpiw_key = f"ccc_mpiw_{campaign_name}"
                    
                    f.write(f"{campaign}:\n")
                    f.write("  RMSE: {:.4f}\n".format(ccc_rmse_dict["Campaign"].iloc[0][metric_key]))
                    f.write("  PICP: {:.4f}\n".format(ccc_picp_dict["Campaign"].iloc[0][picp_key]))
                    f.write("  MPIW: {:.4f}\n".format(ccc_mpiw_dict["Campaign"].iloc[0][mpiw_key]))
            f.write("\n")
        
        # Define colors and markers for the plots
        hue_elements_lai = pd.unique(combined_lai_results["Land cover"])
        hue2_elements_lai = pd.unique(combined_lai_results["Campaign"])
        
        hue_elements_ccc = pd.unique(combined_ccc_results["Land cover"])
        hue2_elements_ccc = pd.unique(combined_ccc_results["Campaign"])
        
        # Create color dictionaries for land cover types
        hue_color_dict_lai = {}
        for j, h_e in enumerate(hue_elements_lai):
            hue_color_dict_lai[h_e] = f"C{j}"
        
        hue_color_dict_ccc = {}
        for j, h_e in enumerate(hue_elements_ccc):
            hue_color_dict_ccc[h_e] = f"C{j}"
        
        # Create marker dictionaries for campaigns
        default_markers = ["o", "v", "D", "s", "+", "^", "1", "."]
        
        hue2_markers_dict_lai = {}
        for j, h2_e in enumerate(hue2_elements_lai):
            hue2_markers_dict_lai[h2_e] = default_markers[j % len(default_markers)]
        
        hue2_markers_dict_ccc = {}
        for j, h2_e in enumerate(hue2_elements_ccc):
            hue2_markers_dict_ccc[h2_e] = default_markers[j % len(default_markers)]
        
        # Create LAI plot
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        regression_plot_2hues(
            combined_lai_results, 
            x="In-situ LAI", 
            y="Predicted LAI", 
            fig=fig, 
            ax=ax, 
            hue="Land cover", 
            hue2="Campaign",
            hue_color_dict=hue_color_dict_lai, 
            hue2_markers_dict=hue2_markers_dict_lai,
            title_hue="Land cover", 
            title_hue2="Site",
            legend_col=1,
            s=60,
            error_x="lai std",
            error_y="Predicted LAI std"
        )
        ax.set_title("LAI Validation")
        plt.tight_layout()
        lai_plot_path = os.path.join(output_dir, f"combined_lai_validation_{timestamp}.png")
        plt.savefig(lai_plot_path)
        plt.savefig(os.path.join(output_dir, f"combined_lai_validation_{timestamp}.pdf"), format='pdf', bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved LAI validation plot to {lai_plot_path}")
        
        # Create CCC plot
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        regression_plot_2hues(
            combined_ccc_results, 
            x="In-situ CCC", 
            y="Predicted CCC", 
            fig=fig, 
            ax=ax, 
            hue="Land cover", 
            hue2="Campaign",
            hue_color_dict=hue_color_dict_ccc, 
            hue2_markers_dict=hue2_markers_dict_ccc,
            title_hue="Land cover", 
            title_hue2="Site",
            legend_col=1,
            s=60,
            error_x="ccc std",
            error_y="Predicted CCC std"
        )
        # Add CCC units to axis labels
        ax.set_xlabel(r"In-situ CCC $\mu$g cm$^{-2}$")
        ax.set_ylabel(r"Predicted CCC $\mu$g cm$^{-2}$")
        
        ax.set_title("CCC Validation")
        plt.tight_layout()
        ccc_plot_path = os.path.join(output_dir, f"combined_ccc_validation_{timestamp}.png")
        plt.savefig(ccc_plot_path)
        plt.savefig(os.path.join(output_dir, f"combined_ccc_validation_{timestamp}.pdf"), format='pdf', bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved CCC validation plot to {ccc_plot_path}")
        
        # Create a side-by-side plot with LAI and CCC
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=300)
        
        # LAI plot on the left
        regression_plot_2hues(
            combined_lai_results, 
            x="In-situ LAI", 
            y="Predicted LAI", 
            fig=fig, 
            ax=ax1, 
            hue="Land cover", 
            hue2="Campaign",
            hue_color_dict=hue_color_dict_lai, 
            hue2_markers_dict=hue2_markers_dict_lai,
            title_hue="Land cover", 
            title_hue2="Site",
            legend_col=1,
            s=60,
            display_text=True,
            error_x="lai std",
            error_y="Predicted LAI std"
        )
        ax1.set_title("LAI Validation", fontsize=14)
        
        # CCC plot on the right
        regression_plot_2hues(
            combined_ccc_results, 
            x="In-situ CCC", 
            y="Predicted CCC", 
            fig=fig, 
            ax=ax2, 
            hue="Land cover", 
            hue2="Campaign",
            hue_color_dict=hue_color_dict_ccc, 
            hue2_markers_dict=hue2_markers_dict_ccc,
            title_hue="", 
            title_hue2="",
            legend_col=1,
            s=60,
            display_text=True,
            error_x="ccc std",
            error_y="Predicted CCC std"
        )
        # Add CCC units to axis labels for the right subplot
        ax2.set_xlabel(r"In-situ CCC $\mu$g cm$^{-2}$")
        ax2.set_ylabel(r"Predicted CCC $\mu$g cm$^{-2}$")
        
        ax.set_title("CCC Validation")
        plt.tight_layout()
        combined_plot_path = os.path.join(output_dir, f"combined_lai_ccc_validation_{timestamp}.png")
        plt.savefig(combined_plot_path)
        plt.savefig(os.path.join(output_dir, f"combined_lai_ccc_validation_{timestamp}.pdf"), format='pdf', bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved combined LAI/CCC validation plot to {combined_plot_path}")
        
        # Print statistics to console
        print("\n=== LAI Validation Statistics ===")
        print(f"Overall RMSE: {lai_rmse_dict['Site'].iloc[0]['lai_rmse_all']:.4f}")
        print(f"Overall PICP: {lai_picp_dict['Site'].iloc[0]['lai_picp_all']:.4f}")
        print(f"Overall MPIW: {lai_mpiw_dict['Site'].iloc[0]['lai_mpiw_all']:.4f}")
        print("\nBy Campaign:")
        
        # Process each campaign with available metrics for console output
        for campaign in pd.unique(combined_lai_results["Campaign"]):
            # Try to find matching metric key
            matching_keys = [k for k in lai_rmse_keys if k.split("lai_rmse_")[1] in campaign.lower().replace("(", "").replace(")", "").replace(" ", "_")]
            if matching_keys:
                metric_key = matching_keys[0]
                campaign_name = metric_key.replace("lai_rmse_", "")
                picp_key = f"lai_picp_{campaign_name}"
                mpiw_key = f"lai_mpiw_{campaign_name}"
                
                print(f"{campaign}:")
                print(f"  RMSE: {lai_rmse_dict['Campaign'].iloc[0][metric_key]:.4f}")
                print(f"  PICP: {lai_picp_dict['Campaign'].iloc[0][picp_key]:.4f}")
                print(f"  MPIW: {lai_mpiw_dict['Campaign'].iloc[0][mpiw_key]:.4f}")
        
        print("\n=== CCC Validation Statistics ===")
        print(f"Overall RMSE: {ccc_rmse_dict['Site'].iloc[0]['ccc_rmse_all']:.4f}")
        print(f"Overall PICP: {ccc_picp_dict['Site'].iloc[0]['ccc_picp_all']:.4f}")
        print(f"Overall MPIW: {ccc_mpiw_dict['Site'].iloc[0]['ccc_mpiw_all']:.4f}")
        print("\nBy Campaign:")
        
        # Process each campaign with available metrics for console output
        for campaign in pd.unique(combined_ccc_results["Campaign"]):
            # Try to find matching metric key
            matching_keys = [k for k in ccc_rmse_keys if k.split("ccc_rmse_")[1] in campaign.lower().replace("(", "").replace(")", "").replace(" ", "_")]
            if matching_keys:
                metric_key = matching_keys[0]
                campaign_name = metric_key.replace("ccc_rmse_", "")
                picp_key = f"ccc_picp_{campaign_name}"
                mpiw_key = f"ccc_mpiw_{campaign_name}"
                
                print(f"{campaign}:")
                print(f"  RMSE: {ccc_rmse_dict['Campaign'].iloc[0][metric_key]:.4f}")
                print(f"  PICP: {ccc_picp_dict['Campaign'].iloc[0][picp_key]:.4f}")
                print(f"  MPIW: {ccc_mpiw_dict['Campaign'].iloc[0][mpiw_key]:.4f}")
        
        logger.info("Combined LAI/CCC validation completed successfully")
        
        # Also create a multi-variable comparison plot if requested
        if include_side_by_side:
            # Create dictionary with all variables
            all_variables = {
                "lai": combined_lai_results,
                "ccc": combined_ccc_results
            }
            
            # Create comparison plot for LAI and CCC
            comparison_plot_path = create_comparison_plot(
                results_dict=all_variables,
                variables=["lai", "ccc"],
                output_dir=output_dir,
                timestamp=timestamp
            )
            
            logger.info(f"Saved multi-variable comparison plot to {comparison_plot_path}")
        
        return {
            "lai_results": combined_lai_path,
            "ccc_results": combined_ccc_path,
            "lai_plot": lai_plot_path,
            "ccc_plot": ccc_plot_path,
            "combined_plot": combined_plot_path if include_side_by_side else None,
            "lai_metrics": {
                "rmse": lai_rmse_dict["Site"].iloc[0]["lai_rmse_all"],
                "picp": lai_picp_dict["Site"].iloc[0]["lai_picp_all"],
                "mpiw": lai_mpiw_dict["Site"].iloc[0]["lai_mpiw_all"]
            },
            "ccc_metrics": {
                "rmse": ccc_rmse_dict["Site"].iloc[0]["ccc_rmse_all"],
                "picp": ccc_picp_dict["Site"].iloc[0]["ccc_picp_all"],
                "mpiw": ccc_mpiw_dict["Site"].iloc[0]["ccc_mpiw_all"]
            }
        }
    
    except Exception as e:
        logger.error(f"Error during combined validation: {e}")
        logger.error(traceback.format_exc())
        raise

def main():
    """Main execution function"""
    # Parse arguments
    args = parse_args()
    
    # Set up absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(args.model_path)
    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Load model
        model = load_model(model_path, args.device, args.rsr_dir)
        
        # Extract model name from path for filenames
        model_name = os.path.basename(model_path)
        if model_name.endswith('.pth') or model_name.endswith('.tar'):
            model_name = model_name.rsplit('.', 1)[0]
        if model_name == "":
            model_name = "pvae"
        
        # Determine which datasets to validate
        datasets_to_validate = args.datasets
        if "all" in datasets_to_validate:
            datasets_to_validate = ["belsar", "frm4veg"]
        
        # Dictionary to store results
        frm4veg_results = {}
        belsar_metrics = None
        belsar_summary = None
        
        # Run FRM4VEG validation if requested
        if "frm4veg" in datasets_to_validate or any("frm4veg" in d for d in datasets_to_validate):
            frm4veg_output_dir = os.path.join(output_dir, "frm4veg")
            os.makedirs(frm4veg_output_dir, exist_ok=True)
            
            # Get FRM4VEG data paths
            frm4veg_data = get_frm4veg_validation_data_paths()
            
            # Filter sites if specific FRM4VEG sites requested
            sites_to_validate = {}
            for site_key in frm4veg_data:
                if site_key in datasets_to_validate or "frm4veg" in datasets_to_validate:
                    sites_to_validate[site_key] = frm4veg_data[site_key]
            
            logger.info(f"Running FRM4VEG validation for {len(sites_to_validate)} site(s)...")
            frm4veg_results = run_frm4veg_validation(
                model=model,
                data_dirs=sites_to_validate,
                output_dir=frm4veg_output_dir,
                mode=args.mode,
                save_reconstruction=args.reconstruction,
                export_detailed=args.export_detailed,
                export_plots=args.export_plots
            )
            
            logger.info(f"FRM4VEG validation completed for {len(frm4veg_results)} site(s)")
        
        # Run BelSAR validation if requested
        if "belsar" in datasets_to_validate:
            belsar_output_dir = os.path.join(output_dir, "belsar")
            os.makedirs(belsar_output_dir, exist_ok=True)
            
            # Ensure BelSAR data is found in the belsar_validation subdirectory
            belsar_data_dir = os.path.join(data_dir, "belsar_validation")
            if not os.path.exists(belsar_data_dir):
                logger.warning(f"BelSAR data directory not found at {belsar_data_dir}")
                logger.warning("Will attempt to use the main data directory instead")
                belsar_data_dir = data_dir
            
            logger.info(f"Running BelSAR validation using data from: {belsar_data_dir}")
            belsar_metrics, belsar_summary = run_belsar_validation(
                model=model,
                data_dir=belsar_data_dir,
                output_dir=belsar_output_dir,
                model_name=model_name,
                mode=args.mode,
                method=args.method,
                crop_type=args.crop_type,
                save_reconstruction=args.reconstruction,
                export_detailed=args.export_detailed,
                export_plots=args.export_plots
            )
            
            if belsar_metrics is not None:
                logger.info("BelSAR validation completed successfully")
            else:
                logger.warning("BelSAR validation did not produce results")
        
        # Generate combined validation if requested and we have both datasets
        if args.combine_results and frm4veg_results and belsar_metrics is not None:
            combined_output_dir = os.path.join(output_dir, "combined")
            os.makedirs(combined_output_dir, exist_ok=True)
            
            logger.info("Generating combined validation results...")
            combined_results = run_combined_validation(
                frm4veg_results=frm4veg_results,
                belsar_metrics=belsar_metrics,
                output_dir=combined_output_dir,
                timestamp=timestamp,
                include_side_by_side=args.comparison_plot
            )
            
            logger.info("Combined validation completed successfully")
            
            # Print summary of all validation results
            print("\n=== Validation Results Summary ===")
            print(f"Model: {model_path}")
            print(f"Output directory: {output_dir}")
            print(f"Datasets validated: {', '.join(datasets_to_validate)}")
            
            if combined_results:
                print("\nCombined validation files:")
                print(f"- LAI results: {combined_results['lai_results']}")
                print(f"- CCC results: {combined_results['ccc_results']}")
                print(f"- LAI plot: {combined_results['lai_plot']}")
                print(f"- CCC plot: {combined_results['ccc_plot']}")
                print(f"- Combined plot: {combined_results['combined_plot']}")
        
        logger.info("validation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 