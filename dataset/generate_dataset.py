import argparse
import os
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from dataset.dataset_utils import min_max_to_loc_scale

import prosailvae
from prosailvae.prosail_var_dists import (
    VariableDistribution,
    get_prosail_var_dist,
)
from prosailvae.ProsailSimus import ProsailSimulator, SensorSimulator, EnMapSensorSimulator
from prosailvae.spectral_indices import get_spectral_idx


def correlate_with_lai(
    lai, V, param_dist, mode="v2", lai_conv_override=None, lai_max=15, lai_thresh=None
):
    """
    Correlate a variable with LAI using one of two methods.
    
    Args:
        lai: LAI values
        V: Variable values to correlate
        param_dist: Variable distribution parameters
        mode: Correlation mode ('v1' or 'v2')
        lai_conv_override: Override lai_conv parameter
        lai_max: Maximum LAI value
        lai_thresh: LAI threshold for capping
        
    Returns:
        Correlated variable values
    """
    if mode == "v2":
        if param_dist.C_lai_min is not None:
            # V2 correlation mode
            Vmin_0 = param_dist.low
            Vmax_0 = param_dist.high
            Vmin_lai_max = param_dist.C_lai_min
            Vmax_lai_max = param_dist.C_lai_max
            
            Vmin_lai = Vmin_0 + lai / lai_max * (Vmin_lai_max - Vmin_0)
            Vmax_lai = Vmax_0 + lai / lai_max * (Vmax_lai_max - Vmax_0)
            V_corr = (V - Vmin_0) * (Vmax_lai - Vmin_lai) / (Vmax_0 - Vmin_0) + Vmin_lai
            
            if lai_thresh is not None:
                V_corr[lai > lai_thresh] = Vmin_0 + lai_thresh / lai_max * (Vmin_lai_max - Vmin_0)
            
            return V_corr
    else:  # v1 mode
        if lai is not None and (param_dist.lai_conv is not None or lai_conv_override is not None):
            lai_conv = lai_conv_override if lai_conv_override is not None else param_dist.lai_conv
            V_mean = param_dist.loc if param_dist.loc is not None else (param_dist.high - param_dist.low) / 2
            V_corr = V_mean + (V - V_mean) * np.maximum((lai_conv - lai), 0) / lai_conv
            return V_corr
    
    # No correlation applied
    return V.copy()


def correlate_all_variables_with_lai(
    samples, var_dists, lai_conv_override=None, lai_corr_mode="v2", lai_thresh=None
):
    variable_idx_dict = {
        "N": 0,
        "cab": 1,
        "car": 2,
        "cbrown": 3,
        "cw": 4,
        "cm": 5,
        "lidfa": 7,
        "hspot": 8,
        "psoil": 9,
        "rsoil": 10,
    }
    correlated_samples = samples.copy()
    for variable, idx in variable_idx_dict.items():
        variable_dist = VariableDistribution(**var_dists.asdict()[variable])
        correlated_samples[:, idx] = correlate_with_lai(
            samples[:, 6],
            samples[:, idx],
            variable_dist,
            mode=lai_corr_mode,
            lai_conv_override=lai_conv_override,
            lai_max=var_dists.lai.high,
            lai_thresh=lai_thresh,
        )

    return correlated_samples


def np_sample_param(
    param_dist,
    lai=None,
    n_samples=1,
    uniform_mode=True,
    lai_conv_override=None,
    lai_corr_mode="v2",
    lai_max=15,
    lai_thresh=None,
):
    if param_dist.law == "uniform" or uniform_mode:
        sample = np.random.uniform(
            low=param_dist.low, high=param_dist.high, size=(n_samples)
        )
    elif param_dist.law == "gaussian":
        sample = stats.truncnorm(
            (param_dist.low - param_dist.loc) / param_dist.scale,
            (param_dist.high - param_dist.loc) / param_dist.scale,
            loc=param_dist.loc,
            scale=param_dist.scale,
        ).rvs(n_samples)
    elif param_dist.law == "lognormal":
        low = max(param_dist.low, 1e-8)
        X = stats.truncnorm(
            (np.log(low) - param_dist.loc) / param_dist.scale,
            (np.log(param_dist.high) - param_dist.loc) / param_dist.scale,
            loc=param_dist.loc,
            scale=param_dist.scale,
        ).rvs(n_samples)
        sample = np.exp(X)
    else:
        raise NotImplementedError(
            "Please choose sample distribution among gaussian, uniform and lognormal"
        )
    if lai is not None:
        correlate_with_lai(
            lai,
            sample,
            param_dist,
            mode=lai_corr_mode,
            lai_conv_override=lai_conv_override,
            lai_max=lai_max,
            lai_thresh=lai_thresh,
        )
    return sample


def sample_angles(n_samples=100):
    """
    Generate random viewing and illumination angles for PROSAIL simulation.
    
    Args:
        n_samples: Number of angle configurations to generate.
        
    Returns:
        tuple: (sun zenith, observer zenith, relative azimuth) angles in degrees.
    """
    # Generate random angles within typical ranges
    # Sun zenith angle (usually between 20 and 70 degrees)
    tts = np.random.uniform(20, 70, n_samples)
    
    # Observer/sensor zenith angle (usually between 0 and 10 degrees for nadir-looking satellites)
    tto = np.random.uniform(0, 10, n_samples)
    
    # Relative azimuth angle (between 0 and 360 degrees)
    psi = np.random.uniform(0, 360, n_samples)
    
    return tts, tto, psi


def partial_sample_prosail_vars(
    var_dists,
    lai=None,
    tts=None,
    tto=None,
    psi=None,
    n_samples=1,
    uniform_mode=True,
    lai_corr=False,
    lai_conv_override=None,
    lai_var_dist: VariableDistribution | None = None,
    lai_corr_mode="v2",
    lai_thresh=None,
):
    prosail_vars = np.zeros((n_samples, 14))
    if lai is None:
        if lai_var_dist is not None:
            lai = np_sample_param(
                lai_var_dist, lai=None, n_samples=n_samples, uniform_mode=uniform_mode
            )
        else:
            lai = np_sample_param(
                var_dists.lai, lai=None, n_samples=n_samples, uniform_mode=uniform_mode
            )
    prosail_vars[:, 6] = lai
    variable_idx_dict = {
        "N": 0,
        "cab": 1,
        "car": 2,
        "cbrown": 3,
        "cw": 4,
        "cm": 5,
        "lidfa": 7,
        "hspot": 8,
        "psoil": 9,
        "rsoil": 10,
    }
    for variable, idx in variable_idx_dict.items():
        variable_dist = VariableDistribution(**var_dists.asdict()[variable])
        prosail_vars[:, idx] = np_sample_param(
            variable_dist,
            lai=lai if lai_corr else None,
            n_samples=n_samples,
            uniform_mode=uniform_mode,
            lai_conv_override=lai_conv_override,
            lai_max=var_dists.lai.high,
            lai_corr_mode=lai_corr_mode,
            lai_thresh=lai_thresh,
        )
    tts, tto, psi = sample_angles(n_samples)
    prosail_vars[:, 11] = tts
    prosail_vars[:, 12] = tto
    prosail_vars[:, 13] = psi

    return prosail_vars


def sample_prosail_vars(
    nb_simus=2048,
    prosail_var_dist_type="legacy",
    uniform_mode=False,
    lai_corr=True,
    lai_var_dist: VariableDistribution | None = None,
    lai_corr_mode="v2",
    lai_thresh=None,
):
    prosail_var_dist = get_prosail_var_dist(prosail_var_dist_type)
    samples = partial_sample_prosail_vars(
        prosail_var_dist,
        n_samples=nb_simus,
        uniform_mode=uniform_mode,
        lai_corr=lai_corr,
        lai_var_dist=lai_var_dist,
        lai_corr_mode=lai_corr_mode,
        lai_thresh=lai_thresh,
    )
    return samples


def simulate_reflectances(
    prosail_vars, noise=0, psimulator=None, ssimulator=None, n_samples_per_batch=1024
):
    nb_simus = prosail_vars.shape[0]
    prosail_s2_sim = np.zeros((nb_simus, ssimulator.rsr.size(1)))
    n_full_batch = nb_simus // n_samples_per_batch
    last_batch = nb_simus - nb_simus // n_samples_per_batch * n_samples_per_batch

    for i in range(n_full_batch):
        prosail_r = psimulator(
            torch.from_numpy(
                prosail_vars[i * n_samples_per_batch : (i + 1) * n_samples_per_batch, :]
            )
            .view(n_samples_per_batch, -1)
            .float()
        )
        sim_s2_r = ssimulator(prosail_r).numpy()
        if noise > 0:
            sigma = (
                np.random.rand(n_samples_per_batch, 1) * noise * np.ones_like(sim_s2_r)
            )
            add_noise = np.random.normal(
                loc=np.zeros_like(sim_s2_r), scale=sigma, size=sim_s2_r.shape
            )
            sim_s2_r += add_noise
        prosail_s2_sim[i * n_samples_per_batch : (i + 1) * n_samples_per_batch, :] = (
            sim_s2_r
        )
    if last_batch > 0:
        sim_s2_r = ssimulator(
            psimulator(
                torch.from_numpy(prosail_vars[n_full_batch * n_samples_per_batch :, :])
                .view(last_batch, -1)
                .float()
            )
        ).numpy()
        if noise > 0:
            sigma = np.random.rand(last_batch, 1) * noise * np.ones_like(sim_s2_r)
            add_noise = np.random.normal(
                loc=np.zeros_like(sim_s2_r), scale=sigma, size=sim_s2_r.shape
            )
            sim_s2_r += add_noise
        prosail_s2_sim[n_full_batch * n_samples_per_batch :, :] = sim_s2_r
    return prosail_s2_sim


def np_simulate_prosail_dataset(
    nb_simus=2048,
    noise=0,
    psimulator=None,
    ssimulator=None,
    n_samples_per_batch=1024,
    uniform_mode=False,
    lai_corr=True,
    prosail_var_dist_type="legacy",
    lai_var_dist: VariableDistribution | None = None,
    lai_corr_mode="v2",
    lai_thresh=None,
):
    prosail_vars = sample_prosail_vars(
        nb_simus=nb_simus,
        prosail_var_dist_type=prosail_var_dist_type,
        uniform_mode=uniform_mode,
        lai_corr=lai_corr,
        lai_var_dist=lai_var_dist,
        lai_corr_mode=lai_corr_mode,
        lai_thresh=lai_thresh,
    )

    prosail_s2_sim = simulate_reflectances(
        prosail_vars,
        noise=noise,
        psimulator=psimulator,
        ssimulator=ssimulator,
        n_samples_per_batch=n_samples_per_batch,
    )

    return prosail_vars, prosail_s2_sim


def get_refl_normalization(prosail_refl):
    return prosail_refl.mean(0), prosail_refl.std(0)


def get_bands_norm_factors(s2_r_samples, mode="mean"):
    cos_angle_min = torch.tensor(
        [0.342108564072183, 0.979624800125421, -1.0000]
    )  # sun zenith, S2 senith, relative azimuth
    cos_angle_max = torch.tensor([0.9274847491748729, 1.0000, 1.0000])
    with torch.no_grad():
        spectral_idx = get_spectral_idx(s2_r_samples, bands_dim=1).reshape(4, -1)
        if mode == "mean":
            norm_mean = s2_r_samples.mean(1)
            norm_std = s2_r_samples.std(1)
            idx_norm_mean = spectral_idx.mean(1)
            idx_norm_std = spectral_idx.std(1)

        elif mode == "quantile":
            max_samples = int(1e7)
            norm_mean = torch.quantile(
                s2_r_samples[:, :max_samples], q=torch.tensor(0.5), dim=1
            )
            norm_std = torch.quantile(
                s2_r_samples[:, :max_samples], q=torch.tensor(0.95), dim=1
            ) - torch.quantile(
                s2_r_samples[:, :max_samples], q=torch.tensor(0.05), dim=1
            )
            idx_norm_mean = torch.quantile(
                spectral_idx[:, :max_samples], q=torch.tensor(0.5), dim=1
            )
            idx_norm_std = torch.quantile(
                spectral_idx[:, :max_samples], q=torch.tensor(0.95), dim=1
            ) - torch.quantile(
                spectral_idx[:, :max_samples], q=torch.tensor(0.05), dim=1
            )

        cos_angles_loc, cos_angles_scale = min_max_to_loc_scale(
            cos_angle_min, cos_angle_max
        )

    return (
        norm_mean,
        norm_std,
        cos_angles_loc,
        cos_angles_scale,
        idx_norm_mean,
        idx_norm_std,
    )


def save_dataset(
    data_dir,
    data_file_prefix,
    rsr_dir,
    nb_simus,
    noise=0,
    uniform_mode=False,
    lai_corr=True,
    prosail_var_dist_type="legacy",
    lai_var_dist: VariableDistribution | None = None,
    lai_corr_mode="v2",
    lai_thresh=None,
    prospect_version="5",
):
    psimulator = ProsailSimulator(prospect_version=prospect_version)
    bands = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        11,
        12,
    ]  # B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
    ssimulator = SensorSimulator(rsr_dir + "/sentinel2.rsr", bands=bands)
    prosail_vars, prosail_s2_sim = np_simulate_prosail_dataset(
        nb_simus=nb_simus,
        noise=noise,
        psimulator=psimulator,
        ssimulator=ssimulator,
        n_samples_per_batch=1024,
        uniform_mode=uniform_mode,
        lai_corr=lai_corr,
        prosail_var_dist_type=prosail_var_dist_type,
        lai_var_dist=lai_var_dist,
        lai_corr_mode=lai_corr_mode,
        lai_thresh=lai_thresh,
    )
    (
        norm_mean,
        norm_std,
        cos_angles_loc,
        cos_angles_scale,
        idx_loc,
        idx_scale,
    ) = get_bands_norm_factors(
        torch.from_numpy(prosail_s2_sim).float().transpose(1, 0), mode="quantile"
    )
    torch.save(
        torch.from_numpy(prosail_vars),
        os.path.join(data_dir, f"{data_file_prefix}prosail_sim_vars.pt"),
    )
    torch.save(
        torch.from_numpy(prosail_s2_sim),
        os.path.join(data_dir, f"{data_file_prefix}prosail_s2_sim_refl.pt"),
    )
    torch.save(norm_mean, os.path.join(data_dir, f"{data_file_prefix}norm_mean.pt"))
    torch.save(norm_std, os.path.join(data_dir, f"{data_file_prefix}norm_std.pt"))
    torch.save(
        cos_angles_loc, os.path.join(data_dir, f"{data_file_prefix}angles_loc.pt")
    )
    torch.save(
        cos_angles_scale, os.path.join(data_dir, f"{data_file_prefix}angles_scale.pt")
    )
    torch.save(idx_loc, os.path.join(data_dir, f"{data_file_prefix}idx_loc.pt"))
    torch.save(idx_scale, os.path.join(data_dir, f"{data_file_prefix}idx_scale.pt"))


def save_enmap_dataset(
    data_dir,
    data_file_prefix,
    rsr_dir,
    nb_simus,
    noise=0,
    uniform_mode=False,
    lai_corr=True,
    prosail_var_dist_type="legacy",
    lai_var_dist=None,
    lai_corr_mode="v2",
    lai_thresh=None,
    prospect_version="5",
    R_down=1
):
    """
    Generate and save a simulated EnMAP hyperspectral dataset.
    
    The EnMAP sensor has 224 spectral bands covering the spectral range from 420 to 2450 nm.
    
    Args:
        data_dir: Directory where to save the data
        data_file_prefix: Prefix for the saved files
        rsr_dir: Directory containing the EnMAP RSR file
        nb_simus: Number of simulations to generate
        noise: Noise level to add to the simulated data
        uniform_mode: Whether to use uniform sampling for PROSAIL variables
        lai_corr: Whether to correlate PROSAIL variables with LAI
        prosail_var_dist_type: Type of distribution for PROSAIL variables
        lai_var_dist: Optional specific distribution for LAI
        lai_corr_mode: Mode for LAI correlation (v1 or v2)
        lai_thresh: Optional threshold for LAI correlation
        prospect_version: PROSPECT model version (5, D, or PRO)
        R_down: Downsampling factor for spectral bands
    """
    # Initialize simulators
    psimulator = ProsailSimulator(prospect_version=prospect_version)
    ssimulator = EnMapSensorSimulator(
        rsr_file="enmap.rsr", 
        rsr_dir=rsr_dir,
        device='cpu',
        apply_norm=True,
        R_down=R_down
    )
    
    # Generate PROSAIL variables and simulated reflectance
    prosail_vars, prosail_s2_sim = np_simulate_prosail_dataset(
        nb_simus=nb_simus,
        noise=noise,
        psimulator=psimulator,
        ssimulator=ssimulator,
        n_samples_per_batch=1024,
        uniform_mode=uniform_mode,
        lai_corr=lai_corr,
        prosail_var_dist_type=prosail_var_dist_type,
        lai_var_dist=lai_var_dist,
        lai_corr_mode=lai_corr_mode,
        lai_thresh=lai_thresh,
    )
    
    # Calculate normalization factors
    (
        norm_mean,
        norm_std,
        cos_angles_loc,
        cos_angles_scale,
        idx_loc,
        idx_scale,
    ) = get_bands_norm_factors(
        torch.from_numpy(prosail_s2_sim).float().transpose(1, 0), mode="quantile"
    )
    
    # Save all the data and normalization factors
    os.makedirs(data_dir, exist_ok=True)
    torch.save(
        torch.from_numpy(prosail_vars),
        os.path.join(data_dir, f"{data_file_prefix}prosail_sim_vars.pt"),
    )
    torch.save(
        torch.from_numpy(prosail_s2_sim),
        os.path.join(data_dir, f"{data_file_prefix}prosail_s2_sim_refl.pt"),
    )
    torch.save(norm_mean, os.path.join(data_dir, f"{data_file_prefix}norm_mean.pt"))
    torch.save(norm_std, os.path.join(data_dir, f"{data_file_prefix}norm_std.pt"))
    torch.save(
        cos_angles_loc, os.path.join(data_dir, f"{data_file_prefix}angles_loc.pt")
    )
    torch.save(
        cos_angles_scale, os.path.join(data_dir, f"{data_file_prefix}angles_scale.pt")
    )
    torch.save(idx_loc, os.path.join(data_dir, f"{data_file_prefix}idx_loc.pt"))
    torch.save(idx_scale, os.path.join(data_dir, f"{data_file_prefix}idx_scale.pt"))
    
    # Save info about the dataset
    with open(os.path.join(data_dir, f"{data_file_prefix}info.txt"), "w") as f:
        f.write(f"EnMAP simulated dataset with {nb_simus} samples\n")
        f.write(f"PROSPECT version: {prospect_version}\n")
        f.write(f"Number of bands: {ssimulator.nb_bands}\n")
        f.write(f"Spectral range: {ssimulator.rsr_range[0]/1000}-{ssimulator.rsr_range[1]/1000} µm\n")
        f.write(f"Noise level: {noise}\n")
        f.write(f"LAI correlation: {lai_corr_mode if lai_corr else 'None'}\n")
        f.write(f"Downsampling factor: {R_down}\n")
    
    print(f"EnMAP dataset with {nb_simus} samples saved to {data_dir}")
    
    return prosail_vars, prosail_s2_sim


def get_data_generation_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(os.getcwd(), 'data'),
                        help="Directory where data is saved")
    parser.add_argument('--file_prefix', type=str, default='',
                        help="Prefix for the data files")
    parser.add_argument('--rsr_dir', type=str,
                        default=os.path.join(os.getcwd(), 'rsr_data'),
                        help="Directory where the RSR data is stored")
    parser.add_argument('--n_samples', type=int, default=300000,
                        help="Number of samples to generate")
    parser.add_argument('--noise', type=float, default=0.0,
                        help="Noise to add to the simulated data")
    parser.add_argument('--dist_type', type=str, default='legacy',
                        help="Distribution type for the PROSAIL variables")
    parser.add_argument('--lai_corr_mode', type=str, default='v2',
                        help="Mode for the LAI correlation (v1 or v2)")
    parser.add_argument('--lai_thresh', type=float, default=None,
                        help="Threshold for LAI correlation")
    parser.add_argument('--prospect_version', type=str, default='5',
                        help="PROSPECT version to use (5, D, or PRO)")
    parser.add_argument('--sensor_type', type=str, default='s2',
                        help="Sensor type to simulate (s2 or enmap)")
    parser.add_argument('--r_down', type=int, default=1,
                        help="Downsampling factor for EnMAP bands (only used if sensor_type is enmap)")
    
    return parser


def main():
    parser = get_data_generation_parser()
    parser = parser.parse_args()
    
    data_dir = parser.data_dir
    lai_thresh = parser.lai_thresh
    
    print(f"Generating dataset with {parser.n_samples} samples")
    print(f"Sensor type: {parser.sensor_type}")
    print(f"Data directory: {data_dir}")
    print(f"RSR directory: {parser.rsr_dir}")
    print(f"File prefix: {parser.file_prefix}")
    print(f"Distribution type: {parser.dist_type}")
    print(f"LAI correlation mode: {parser.lai_corr_mode}")
    print(f"PROSPECT version: {parser.prospect_version}")
    
    if parser.sensor_type == "s2":
        print("Generating Sentinel-2 dataset...")
        save_dataset(
            data_dir,
            parser.file_prefix,
            parser.rsr_dir,
            parser.n_samples,
            parser.noise,
            uniform_mode=False,
            lai_corr=parser.lai_corr,
            prosail_var_dist_type=parser.dist_type,
            lai_var_dist=parser.lai_var_dist,
            lai_corr_mode=parser.lai_corr_mode,
            lai_thresh=lai_thresh,
            prospect_version=parser.prospect_version,
        )
    elif parser.sensor_type == "enmap":
        print("Generating EnMAP dataset...")
        print(f"Downsampling factor: {parser.r_down}")
        save_enmap_dataset(
            data_dir,
            parser.file_prefix,
            parser.rsr_dir,
            parser.n_samples,
            parser.noise,
            uniform_mode=False,
            lai_corr=True,
            prosail_var_dist_type=parser.dist_type,
            lai_corr_mode=parser.lai_corr_mode,
            lai_thresh=lai_thresh,
            prospect_version=parser.prospect_version,
            R_down=parser.r_down
        )
    else:
        print(f"Unknown sensor type: {parser.sensor_type}")
        print("Supported types: s2, enmap")
        sys.exit(1)


if __name__ == "__main__":
    if socket.gethostname() == "CELL200973":
        args = [
            # "-wd", "True",
            #   "-w", "True",
            "--data_dir",
            "/Users/princemensah/Desktop/prosailvae/data/simulated_dataset",
            "--dist_type",
            "new_v2",
            "--n_samples",
            "42000",
            "--lai_corr_mode",
            "v2",
            #   "--prospect_version", "D",
            "--sensor_type",
            "s2",
        ]
        main()
    else:
        main()
