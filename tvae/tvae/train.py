#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
transformer-VAE Training Script
==============================

This script implements the training pipeline for the TRANSFORMER-VAE model, a variational autoencoder
designed for biophysical parameter retrieval.

The model can work with both Sentinel-2 and EnMAP sensor data, and supports various
loss functions, learning rate schedules, and training optimizations.
"""

import os
import sys
import torch
import socket
import argparse
import shutil
import logging
import logging.config
import traceback
import time
import pandas as pd
import numpy as np
import torch.optim as optim
from tqdm import trange
from pathlib import Path
from dataclasses import dataclass
torch.autograd.set_detect_anomaly(True)
from prosailvae import __path__ as PPATH
TOP_PATH = os.path.join(PPATH[0], os.pardir)
from tqdm.contrib.logging import logging_redirect_tqdm
from prosailvae.prosail_vae import (load_prosail_vae_with_hyperprior, get_prosail_vae_config, ProsailVAEConfig, load_params)
from dataset.loaders import (get_simloader, get_enmap_simloader)
from metrics.results import (save_results_on_sim_data, get_res_dir_path, save_validation_results, plot_losses)
from utils.utils import (save_dict, get_RAM_usage, get_total_RAM, plot_grad_flow, load_standardize_coeffs)
from prosailvae.ProsailSimus import get_bands_idx, EnMapSensorSimulator

CUDA_LAUNCH_BLOCKING=1
LOGGER_NAME = 'Transformer-VAE logger'
PC_SOCKET_NAME = 'CELL200973'

@dataclass
class DatasetConfig:
    """
    Configuration parameters for the dataset.
    
    Attributes:
        dataset_file_prefix (str): Prefix for dataset files.
        sensor_type (str): Type of sensor data ('sentinel2' or 'enmap').
    """
    dataset_file_prefix: str = ""
    sensor_type: str = "sentinel2"  # or "enmap"

@dataclass
class TrainingConfig:
    """
    Configuration parameters for model training.
    
    Attributes:
        batch_size (int): Number of samples per batch.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for optimization.
        test_size (float): Fraction of data to use for testing.
        valid_ratio (float): Fraction of data to use for validation.
        k_fold (int): Current fold number in k-fold cross validation.
        n_fold (int | None): Total number of folds for cross validation.
        n_samples (int): Number of Monte Carlo samples for variational inference.
        sensor_type (str): Type of sensor data ('sentinel2' or 'enmap').
    """
    batch_size: int = 128
    epochs: int = 1
    lr: float = 0.0001
    test_size: float = 0.01
    valid_ratio: float = 0.01
    k_fold: int = 0
    n_fold: int | None = None
    n_samples: int = 2
    sensor_type: str = "sentinel2"  # or "enmap"

def get_training_config(params):
    """
    Create a TrainingConfig instance from parameters dictionary.
    
    Args:
        params (dict): Dictionary containing training parameters.
        
    Returns:
        TrainingConfig: A configuration object for training.
    """
    return TrainingConfig(
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        lr=params["lr"],
        test_size=params["test_size"],
        valid_ratio=params["valid_ratio"],
        k_fold=params["k_fold"],
        n_fold=params["n_fold"],
        n_samples=params['n_samples'],
        sensor_type=params.get("sensor_type", "sentinel2")
    )

@dataclass
class ModelConfig:
    """
    Configuration parameters for the model architecture.
    
    Attributes:
        supervised (bool): Whether to use supervised learning component.
        beta_kl (float): Weight for KL divergence term in VAE loss.
        beta_index (float): Weight for index term in loss function.
    """
    supervised: bool = False
    beta_kl: float = 1
    beta_index: float = 1

def get_prosailvae_train_parser():
    """
    Creates a command-line argument parser for the transformer-VAE training script.
    
    The parser accepts various arguments for configuring the training process,
    including paths to configuration files, dataset directories, and various
    training options.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')

    parser.add_argument("-n", dest="n_fold",
                        help="number k of fold",
                        type=int, default=0)

    parser.add_argument("-c", dest="config_file",
                        help="name of config json file on config directory.",
                        type=str, default="config.json")
    
    parser.add_argument("-cd", dest="config_dir",
                        help="path to config directory",
                        type=str, default="")
    
    parser.add_argument("-x", dest="n_xp",
                        help="Number of experience (to use in case of kfold)",
                        type=int, default=1)

    parser.add_argument("-o", dest="overwrite_xp",
                        help="Allow overwrite of experiment (fold)",
                        type=bool, default=True)

    parser.add_argument("-r", dest="root_results_dir",
                        help="path to root results directory",
                        type=str, default="")

    parser.add_argument("-rsr", dest="rsr_dir",
                        help="directory of rsr_file",
                        type=str, default='/Users/princemensah/Desktop/transformervae/prosailvae/data')

    parser.add_argument("-a", dest="xp_array",
                        help="array training (false for single xp) ",
                        type=bool, default=False)

    parser.add_argument("-p", dest="plot_results",
                        help="toggle results plotting",
                        type=bool, default=False)

    parser.add_argument("-w", dest="weiss_mode",
                        help="removes B2 and B8 bands for validation with weiss data",
                        type=bool, default=False)
    return parser

def switch_loss(epoch, n_epoch, PROSAIL_VAE, swith_ratio = 0.75):
    """
    Switches the loss type for the transformer-VAE model based on training progress.
    
    Changes from hybrid_nll loss to full_nll loss after a certain ratio of
    total training epochs have been completed.
    
    Args:
        epoch (int): Current training epoch.
        n_epoch (int): Total number of training epochs.
        PROSAIL_VAE: The transformer-VAE model instance.
        swith_ratio (float, optional): Ratio of total epochs after which to switch loss type.
            Defaults to 0.75.
    """
    loss_type = PROSAIL_VAE.decoder.loss_type
    if loss_type == "hybrid_nll":
        if epoch > swith_ratio * n_epoch:
            PROSAIL_VAE.decoder.loss_type = "full_nll"

def initialize_by_training(n_models: int,
                           n_epochs: int,
                           n_samples: int,
                           train_loader,
                           valid_loader,
                           lr: float,
                           logger,
                           pv_config: ProsailVAEConfig,
                           pv_config_hyper: ProsailVAEConfig | None = None,
                           break_at_rec_loss=None,
                           max_sec=3600):
    """
    Initialize transformer-VAE by training multiple models and selecting the best.
    
    This function trains multiple transformer-VAE models with different initializations
    for a small number of epochs, and keeps the best model based on validation loss.
    This helps to avoid poor local minima during training.
    
    Args:
        n_models (int): Number of models to initialize and train.
        n_epochs (int): Number of epochs to train each model.
        n_samples (int): Number of Monte Carlo samples for variational inference.
        train_loader: DataLoader for training data.
        valid_loader: DataLoader for validation data.
        lr (float): Learning rate for the optimizer.
        logger: Logger instance for recording progress.
        pv_config (ProsailVAEConfig): Configuration for the transformer-VAE model.
        pv_config_hyper (ProsailVAEConfig, optional): Configuration for hyperprior model.
            Defaults to None.
        break_at_rec_loss (float, optional): Reconstruction loss threshold to stop initialization.
            Defaults to None.
        max_sec (int, optional): Maximum time in seconds to spend on initialization.
            Defaults to 3600.
    
    Returns:
        bool: True if initialization was stopped due to achieving target reconstruction loss.
    """
    min_valid_loss = torch.inf
    broke_at_rec = False
    logger.info(f"Intializing by training {n_models} models for {n_epochs} epochs:")
    best_model_idx = 0
    t0=time.time()
    for i in range(n_models):
        logger.info(f'=========================== Model {i} ============================')
        prosail_vae = load_prosail_vae_with_hyperprior(pv_config=pv_config,
                                                       pv_config_hyper=pv_config_hyper,
                                                       logger_name=LOGGER_NAME)
        optimizer = optim.Adam(prosail_vae.parameters(), lr=lr, weight_decay=1e-2)
        _, all_valid_loss_df, _, _ = training_loop(
            prosail_vae,
            optimizer,
            n_epochs,
            train_loader,
            valid_loader,
            res_dir=None,
            n_samples=n_samples,
            lr_recompute=None,
            exp_lr_decay=-1,
            plot_gradient=False,
            lr_recompute_mode=False,
            cycle_training=False,
            accum_iter=1,
            lrs_threshold=0.01,
            max_sec=None
        )
        
        model_min_loss = all_valid_loss_df['loss_sum'].values.min()
        if min_valid_loss > model_min_loss:
            min_valid_loss = model_min_loss
            best_model_idx = i
            prosail_vae.save_ae(n_epochs, optimizer, model_min_loss, pv_config.vae_save_file_path)
        if break_at_rec_loss is not None:
            if all_valid_loss_df['rec_loss'].values.min() <= break_at_rec_loss:
                logger.info(f"Model {i} has gone under threshold loss {all_valid_loss_df['rec_loss'].values.min()} < {break_at_rec_loss}.")
                broke_at_rec = True
                break
        if time.time() - t0 > max_sec:
            break
    logger.info(f'Best model is model {best_model_idx}.')
    logger.info(f'=====================================================================')
    return broke_at_rec

def training_loop(prosail_vae,
                  optimizer,
                  n_epoch,
                  train_loader,
                  valid_loader,
                  res_dir=None,
                  n_samples=20,
                  lr_recompute=None,
                  exp_lr_decay=0,
                  plot_gradient=False,
                  lr_recompute_mode=True,
                  cycle_training=False,
                  accum_iter=1,
                  lrs_threshold=0.01,
                  lr_init=5e-4,
                  validation_at_every_epoch=None,
                  validation_dir=None,
                  frm4veg_data_dir=None,
                  frm4veg_2021_data_dir=None,
                  belsar_data_dir=None,
                  lai_cyclical_loader=None,
                  max_sec=None):
    """
    Main training loop for the transformer-VAE model.
    
    This function handles the complete training process, including epoch iteration,
    optimizer updates, loss tracking, validation, learning rate scheduling, and
    saving the best model.
    
    Args:
        prosail_vae: The transformer-VAE model instance.
        optimizer: The optimizer for training the model.
        n_epoch (int): Number of epochs to train.
        train_loader: DataLoader for training data.
        valid_loader: DataLoader for validation data.
        res_dir (str, optional): Directory to save results. Defaults to None.
        n_samples (int, optional): Number of Monte Carlo samples for variational inference. 
            Defaults to 20.
        lr_recompute (int, optional): Patience for learning rate scheduler. Defaults to None.
        exp_lr_decay (float, optional): Exponential learning rate decay factor. Defaults to 0.
        plot_gradient (bool, optional): Whether to plot gradient flow. Defaults to False.
        lr_recompute_mode (bool, optional): Learning rate scheduler mode. Defaults to True.
        cycle_training (bool, optional): Whether to use cyclical training. Defaults to False.
        accum_iter (int, optional): Gradient accumulation steps. Defaults to 1.
        lrs_threshold (float, optional): Learning rate scheduler threshold. Defaults to 0.01.
        lr_init (float, optional): Initial learning rate for cyclical training. Defaults to 5e-4.
        validation_at_every_epoch (int, optional): Run validation every N epochs. Defaults to None.
        validation_dir (str, optional): Directory to save validation results. Defaults to None.
        frm4veg_data_dir (str, optional): Directory with frm4veg data. Defaults to None.
        frm4veg_2021_data_dir (str, optional): Directory with frm4veg 2021 data. Defaults to None.
        belsar_data_dir (str, optional): Directory with Belsar data. Defaults to None.
        lai_cyclical_loader: DataLoader for LAI cyclical evaluation. Defaults to None.
        max_sec (int, optional): Maximum training time in seconds. Defaults to None.
    
    Returns:
        tuple: Tuple containing training loss DataFrame, validation loss DataFrame,
               learning rate info DataFrame, and cyclical RMSE values.
    """
    t_init = time.time()
    cyclical_lai_precomputed = True
    if lai_cyclical_loader is None:
        lai_cyclical_loader = valid_loader
        cyclical_lai_precomputed = False

    logger = logging.getLogger(LOGGER_NAME)
    tbeg = time.time()
    if prosail_vae.decoder.loss_type == 'mse':
        n_samples = 1
        logger.info('MSE Loss enabled, setting number of monte-carlo samples to 1')

    all_train_loss_df = pd.DataFrame()
    all_valid_loss_df = pd.DataFrame()
    info_df = pd.DataFrame()
    best_val_loss = torch.inf
    total_ram = get_total_RAM()
    old_lr = optimizer.param_groups[0]['lr']

    # Set up learning rate scheduling
    if exp_lr_decay > 0:
        if lr_recompute_mode:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=exp_lr_decay
            )
        else:
            if lr_recompute is not None:
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer, patience=lr_recompute,
                    threshold=lrs_threshold, threshold_mode='abs'
                )

    # Limit samples for PC environment to improve development speed
    max_train_samples_per_epoch = 50
    max_valid_samples_per_epoch = 200
    if socket.gethostname() == PC_SOCKET_NAME:
        max_train_samples_per_epoch = 5
        max_valid_samples_per_epoch = 2

    all_cyclical_rmse = []

    with logging_redirect_tqdm():
        for epoch in trange(n_epoch, desc='transformer-VAE training', leave=True):
            # Perform validation on specified epochs if configured
            if validation_at_every_epoch is not None:
                if epoch % validation_at_every_epoch == 0:
                    validation_dir_at_epoch = os.path.join(validation_dir, f"epoch_{epoch}")
                    _, cyclical_rmse = prosail_vae.get_cyclical_metrics_from_loader(
                        lai_cyclical_loader, lai_precomputed=cyclical_lai_precomputed
                    )
                    all_cyclical_rmse.append(cyclical_rmse.cpu().item())
                    save_validation_results(
                        prosail_vae, validation_dir_at_epoch,
                        frm4veg_data_dir=frm4veg_data_dir,
                        frm4veg_2021_data_dir=frm4veg_2021_data_dir,
                        belsar_data_dir=belsar_data_dir,
                        model_name=f"pvae_{epoch}",
                        method="simple_interpolate",
                        mode="sim_tg_mean", 
                        remove_files=True, 
                        plot_results=False,  # or True if you want
                        save_reconstruction=False
                    )

            t0 = time.time()
            # Handle learning rate for cyclical training
            if optimizer.param_groups[0]['lr'] < 5e-8:
                if not cycle_training:
                    break  # stop training if lr too low
                for g in optimizer.param_groups:
                    g['lr'] = lr_init
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer,
                    patience=lr_recompute,
                    threshold=0.01,
                    threshold_mode='abs'
                )
            if lr_recompute_mode:
                raise NotImplementedError  # same logic as your original code

            # Record learning rate
            info_df = pd.concat([
                info_df,
                pd.DataFrame({
                    'epoch': epoch,
                    "lr": optimizer.param_groups[0]['lr']
                }, index=[0])
            ], ignore_index=True)

            # Training step
            try:
                train_loss_dict = prosail_vae.fit(
                    train_loader, optimizer,
                    n_samples=n_samples,
                    max_samples=max_train_samples_per_epoch,
                    accum_iter=accum_iter
                )
                if plot_gradient and res_dir is not None:
                    if not os.path.isdir(os.path.join(res_dir, "gradient_flows")):
                        os.makedirs(os.path.join(res_dir, "gradient_flows"))
                    plot_grad_flow(
                        prosail_vae,
                        savefile=os.path.join(res_dir, "gradient_flows", f"grad_flow_{epoch}.svg")
                    )
            except Exception as exc:
                logger.error(f"Error during Training at epoch {epoch} !")
                logger.error('Original error :')
                logger.error(str(exc))
                print(f"Error during Training at epoch {epoch} !")
                print('Original error :')
                print(str(exc))
                traceback.print_exc()
                break

            # Validation step
            try:
                valid_loss_dict = prosail_vae.validate(
                    valid_loader,
                    n_samples=n_samples,
                    max_samples=max_valid_samples_per_epoch
                )
                if exp_lr_decay > 0:
                    if lr_recompute_mode:
                        lr_scheduler.step()
                    else:
                        lr_scheduler.step(valid_loss_dict['loss_sum'])
            except Exception as exc:
                logger.error(f"Error during Validation at epoch {epoch} !")
                logger.error('Original error :')
                logger.error(str(exc))
                print(f"Error during Validation at epoch {epoch} !")
                print('Original error :')
                print(str(exc))
                traceback.print_exc()

            # Log epoch results
            t1 = time.time()
            ram_usage = get_RAM_usage()
            train_loss_info = '- '.join([
                f"{key}: {'{:.2E}'.format(train_loss_dict[key])} "
                for key in train_loss_dict.keys()
            ])
            valid_loss_info = '- '.join([
                f"{key}: {'{:.2E}'.format(valid_loss_dict[key])} "
                for key in valid_loss_dict.keys()
            ])
            logger.info(
                f"{epoch} -- RAM: {ram_usage} / {total_ram} -- "
                f"lr: {'{:.2E}'.format(optimizer.param_groups[0]['lr'])} -- "
                f"{'{:.1f}'.format(t1 - t0)} s -- {train_loss_info} -- {valid_loss_info}"
            )

            # Store loss values
            train_loss_dict['epoch'] = epoch
            valid_loss_dict['epoch'] = epoch
            all_train_loss_df = pd.concat([
                all_train_loss_df,
                pd.DataFrame(train_loss_dict, index=[0])
            ], ignore_index=True)
            all_valid_loss_df = pd.concat([
                all_valid_loss_df,
                pd.DataFrame(valid_loss_dict, index=[0])
            ], ignore_index=True)

            # Save best model
            if valid_loss_dict['loss_sum'] < best_val_loss:
                best_val_loss = valid_loss_dict['loss_sum']
                if res_dir is not None:
                    prosail_vae.save_ae(epoch, optimizer, best_val_loss,
                                        os.path.join(res_dir, "prosailvae_weights.tar"))

            # Check time limit
            t_end = time.time()
            if max_sec is not None:
                if t_end - t_init > max_sec:
                    logger.info(f"Time limit of {max_sec} seconds over, finishing training early.")
                    break

    # Handle the case of no training (n_epoch < 1)
    if n_epoch < 1:
        all_train_loss_df = pd.DataFrame(data={"loss_sum": 10000, "epoch": 0}, index=[0])
        all_valid_loss_df = pd.DataFrame(data={"loss_sum": 10000, "epoch": 0}, index=[0])
        info_df = pd.DataFrame(data={"lr": 10000, "epoch": 0}, index=[0])
        if res_dir is not None:
            prosail_vae.save_ae(0, optimizer, 0, os.path.join(res_dir, "prosailvae_weights.tar"))

    tend = time.time()
    logger.info(f'Total training time: {tend - tbeg:.1f} seconds')
    return all_train_loss_df, all_valid_loss_df, info_df, all_cyclical_rmse

def setup_training():
    """
    Parse arguments, load config, and set up the environment for training.
    
    This function handles command-line argument parsing, configuration loading,
    directory creation, logging setup, and preparation of all necessary components
    for training the transformer-VAE model.
    
    Returns:
        tuple: A tuple containing:
            - params (dict): Model and training parameters
            - parser (argparse.Namespace): Parsed command-line arguments
            - res_dir (str): Results directory path
            - sim_data_dir (str): Simulated data directory path
            - s2_data_dir (str): Sentinel-2 data directory path
            - params_sup_kl_model (dict): Supervised KL model parameters
            - job_array_dir (str): Job array directory path
            - sup_kl_io_coeffs: I/O standardization coefficients for supervised KL model
            - frm4veg_data_dir (str): FRM4VEG data directory path
            - frm4veg_2021_data_dir (str): FRM4VEG 2021 data directory path
            - belsar_dir (str): BELSAR data directory path
            - model_name (str): Name of the model
    """
    if socket.gethostname() == PC_SOCKET_NAME:
        args = [
            "-n", "0",
            "-c", "config.json",
            "-x", "1",
            "-o", "True",
            "-r", "",
            "-rsr", '/Users/princemensah/Desktop/transformervae/prosailvae/data',
            "-a", "False",
            "-p", "False",
            "-cd", '/Users/princemensah/Desktop/transformervae/tvae/config/config/'
        ]
        parser = get_prosailvae_train_parser().parse_args(args)
    else:
        parser = get_prosailvae_train_parser().parse_args()

    xp_array = parser.xp_array
    job_array_dir = None
    if xp_array:
        job_array_dir = os.path.join(parser.root_results_dir, os.pardir)
    config_dir = parser.config_dir

    # Load parameters from config file
    params = load_params(config_dir, config_file=parser.config_file, parser=parser)
    model_name = parser.config_file[:-5]

    sim_data_dir = params["sim_data_dir"]
    s2_data_dir = params["s2_data_dir"]
    assert parser.n_fold < parser.n_xp

    # Set up results directory
    if len(parser.root_results_dir) == 0:
        root_results_dir = os.path.join(TOP_PATH, "results/")
    else:
        root_results_dir = parser.root_results_dir

    res_dir = get_res_dir_path(root_results_dir, params, parser.n_xp, parser.overwrite_xp)
    # print(f"\nres_dir: {res_dir}\n")
    # print(f"\n{root_results_dir}\n")
    save_dict(params, res_dir + "/config.json")
    params["vae_save_file_path"] = res_dir + "/prosailvae_weights.tar"

    # Configure logging
    logging.basicConfig(filename=res_dir + '/training_log.log',
                              level=logging.INFO, force=True)
    logger_name = LOGGER_NAME
    logger = logging.getLogger(logger_name)
    logger.info('Starting training of transformer-VAE.')
    logger.info('========================================================================')
    logger.info('Parameters are : ')
    for _, key in enumerate(params):
        logger.info(f'{key} : {params[key]}')
    logger.info('========================================================================')

    # Handle supervised KL model if enabled
    if params["supervised_kl"]:
        logger.info("Supervised KL loss (hyperprior) enabled.")
        logger.info(f"copying {params['supervised_config_file']} into {res_dir+'/sup_kl_model_config.json'}")
        shutil.copyfile(params['supervised_config_file'], res_dir + "/sup_kl_model_config.json")
        logger.info(f"copying {params['supervised_weight_file']} into {res_dir+'/sup_kl_model_weights.tar'}")
        shutil.copyfile(params['supervised_weight_file'], res_dir + "/sup_kl_model_weights.tar")
        params_sup_kl_model = load_params(res_dir, "/sup_kl_model_config.json", parser=None)
        params_sup_kl_model['vae_load_file_path'] = res_dir + "/sup_kl_model_weights.tar"
        params_sup_kl_model["load_model"] = True

        sup_kl_io_coeffs = load_standardize_coeffs(os.path.dirname(params["supervised_config_file"]))
        torch.save(sup_kl_io_coeffs.bands.loc, res_dir + "/norm_mean.pt")
        torch.save(sup_kl_io_coeffs.bands.scale, res_dir + "/norm_std.pt")
        torch.save(sup_kl_io_coeffs.idx.loc, res_dir + "/idx_loc.pt")
        torch.save(sup_kl_io_coeffs.idx.scale, res_dir + "/idx_scale.pt")
        torch.save(sup_kl_io_coeffs.angles.loc, res_dir + "/angles_loc.pt")
        torch.save(sup_kl_io_coeffs.angles.scale, res_dir + "/angles_scale.pt")
    else:
        params_sup_kl_model = None
        sup_kl_io_coeffs = None

    frm4veg_data_dir = params["frm4veg_data_dir"]
    frm4veg_2021_data_dir = params["frm4veg_2021_data_dir"]
    belsar_dir = params["belsar_dir"]

    return (params, parser, res_dir, sim_data_dir, s2_data_dir, params_sup_kl_model,
            job_array_dir, sup_kl_io_coeffs, frm4veg_data_dir,
            frm4veg_2021_data_dir, belsar_dir, model_name)

def get_training_data(params, logger):
    """
    Load and prepare training and validation data.
    
    This function sets up data loaders for either Sentinel-2 or EnMAP data,
    and handles standardization coefficients for the data.
    
    Args:
        params (dict): Dictionary containing data loading parameters.
        logger: Logger instance for recording progress.
    
    Returns:
        tuple: A tuple containing:
            - train_loader: DataLoader for training data
            - valid_loader: DataLoader for validation data
            - io_coeffs: Input/output standardization coefficients
            - bands: Spectral bands information
            - prosail_bands: PROSAIL bands information
    
    Raises:
        ValueError: If batch_size is less than 2 for simulated data.
    """
    sim_data_dir = params['sim_data_dir']
    enmap_data_dir = params.get('enmap_data_dir', sim_data_dir)
    bands, prosail_bands = get_bands_idx(params["weiss_bands"])
    
    sensor_type = params.get("sensor_type", "sentinel2")
    
    if sensor_type == "enmap":
        logger.info(f"Loading EnMAP training and validation loader in {enmap_data_dir}/{params['dataset_file_prefix']}...")
        if params["batch_size"] < 2:
            raise ValueError("With simulated data, batch_size cannot be < 2.")
        train_loader, valid_loader = get_enmap_simloader(
            valid_ratio=params["valid_ratio"],
            file_prefix=params["dataset_file_prefix"],
            sample_ids=None,
            batch_size=params["batch_size"],
            data_dir=enmap_data_dir,
            cat_angles=True
        )
        data_dir = enmap_data_dir
    else:
        logger.info(f"Loading Sentinel-2 training and validation loader in {sim_data_dir}/{params['dataset_file_prefix']}...")
        if params["batch_size"] < 2:
            raise ValueError("With simulated data, batch_size cannot be < 2.")
        train_loader, valid_loader = get_simloader(
            valid_ratio=params["valid_ratio"],
            file_prefix=params["dataset_file_prefix"],
            sample_ids=None,
            batch_size=params["batch_size"],
            data_dir=sim_data_dir
        )
        data_dir = sim_data_dir

    # Load or create standardization coefficients
    if params["apply_norm_rec"]:
        io_coeffs = load_standardize_coeffs(
            data_dir,
            params["dataset_file_prefix"],
            n_idx=0 if params["weiss_bands"] else 4
        )
    else:
        io_coeffs = load_standardize_coeffs(
            None,
            params["dataset_file_prefix"],
            n_idx=0 if params["weiss_bands"] else 4
        )

    logger.info(f"Training ({len(train_loader.dataset)} samples) and validation ({len(valid_loader.dataset)} samples) loaders loaded.")
    print(f"\n{train_loader.dataset.tensors[0].shape}\n")

    return train_loader, valid_loader, io_coeffs, bands, prosail_bands

def train_prosailvae(params, parser, res_dir, params_sup_kl_model,
                     sup_kl_io_coeffs, validation_dir=None,
                     frm4veg_data_dir=None, frm4veg_2021_data_dir=None, belsar_data_dir=None,
                     lai_cyclical_loader=None):
    """
    Train a transformer-VAE model with the given configuration.
    
    This function handles the complete training process for a transformer-VAE model,
    including data loading, model initialization, training loop execution,
    and saving results.
    
    Args:
        params (dict): Dictionary containing model and training parameters.
        parser (argparse.Namespace): Parsed command-line arguments.
        res_dir (str): Path to results directory.
        params_sup_kl_model (dict): Parameters for supervised KL model (hyperprior).
        sup_kl_io_coeffs: I/O standardization coefficients for supervised KL model.
        validation_dir (str, optional): Directory for validation results. Defaults to None.
        frm4veg_data_dir (str, optional): Directory with FRM4VEG data. Defaults to None.
        frm4veg_2021_data_dir (str, optional): Directory with FRM4VEG 2021 data. Defaults to None.
        belsar_data_dir (str, optional): Directory with BELSAR data. Defaults to None.
        lai_cyclical_loader: DataLoader for LAI cyclical evaluation. Defaults to None.
    
    Returns:
        tuple: A tuple containing:
            - prosail_vae: Trained transformer-VAE model
            - all_train_loss_df: DataFrame with training loss history
            - all_valid_loss_df: DataFrame with validation loss history
            - info_df: DataFrame with training information (learning rates, etc.)
    """
    logger = logging.getLogger(LOGGER_NAME)
    (train_loader, valid_loader, io_coeffs, bands, prosail_bands) = get_training_data(params, logger)

    # Handle model loading if requested
    if params["load_model"]:
        vae_load_file_path = os.path.join(params["vae_load_dir_path"], "prosailvae_weights.tar")
        io_coeffs = load_standardize_coeffs(
            params["vae_load_dir_path"],
            n_idx=0 if params["weiss_bands"] else 4
        )
    else:
        vae_load_file_path = None

    # Save standardization coefficients
    torch.save(io_coeffs.bands.loc, res_dir + "/norm_mean.pt")
    torch.save(io_coeffs.bands.scale, res_dir + "/norm_std.pt")
    torch.save(io_coeffs.idx.loc, res_dir + "/idx_loc.pt")
    torch.save(io_coeffs.idx.scale, res_dir + "/idx_scale.pt")
    torch.save(io_coeffs.angles.loc, res_dir + "/angles_loc.pt")
    torch.save(io_coeffs.angles.scale, res_dir + "/angles_scale.pt")

    logger.info(f"io_coeffs.bands.loc : {io_coeffs.bands.loc}")
    logger.info(f"io_coeffs.bands.scale : {io_coeffs.bands.scale}")
    logger.info(f"io_coeffs.idx.loc : {io_coeffs.idx.loc}")
    logger.info(f"io_coeffs.idx.scale : {io_coeffs.idx.scale}")

    params["vae_load_file_path"] = vae_load_file_path
    training_config = get_training_config(params)

    # Create model configuration
    pv_config = get_prosail_vae_config(
        params,
        bands=bands,
        prosail_bands=prosail_bands,
        inference_mode=False,
        rsr_dir=parser.rsr_dir,
        io_coeffs=io_coeffs
    )

    # Set up hyperprior configuration if needed
    pv_config_hyper = None
    if params_sup_kl_model is not None:
        bands_hyper, prosail_bands_hyper = get_bands_idx(params_sup_kl_model["weiss_bands"])
        pv_config_hyper = get_prosail_vae_config(
            params_sup_kl_model,
            bands=bands_hyper,
            prosail_bands=prosail_bands_hyper,
            inference_mode=True,
            rsr_dir=parser.rsr_dir,
            io_coeffs=sup_kl_io_coeffs
        )

    # Initialize model using multiple quick training runs if requested
    if params['init_model']:
        n_models = params["n_init_models"]
        lr = params['init_lr']
        n_epochs = params["n_init_epochs"]
        broke_at_rec = initialize_by_training(
            n_models=n_models,
            n_epochs=n_epochs,
            train_loader=train_loader,
            valid_loader=valid_loader,
            lr=lr,
            logger=logger,
            n_samples=training_config.n_samples,
            pv_config=pv_config,
            pv_config_hyper=pv_config_hyper,
            break_at_rec_loss=params["break_init_at_rec_loss"]
        )
        # Try with higher learning rate if initialization didn't reach target loss
        if params["break_init_at_rec_loss"] is not None and not broke_at_rec:
            broke_at_rec = initialize_by_training(
                n_models=n_models,
                n_epochs=n_epochs,
                train_loader=train_loader,
                valid_loader=valid_loader,
                lr=1e-3,
                logger=logger,
                n_samples=training_config.n_samples,
                pv_config=pv_config,
                pv_config_hyper=pv_config_hyper,
                break_at_rec_loss=params["break_init_at_rec_loss"]
            )
            if not broke_at_rec:
                logger.info("No good initialization was found!")
        # Set config to load the best model
        params["load_model"] = True
        params["vae_load_file_path"] = params["vae_save_file_path"]
        if os.path.exists(params["vae_load_file_path"]):
            pv_config = get_prosail_vae_config(
                params,
                bands=bands,
                prosail_bands=prosail_bands,
                inference_mode=False,
                rsr_dir=parser.rsr_dir,
                io_coeffs=io_coeffs
            )
        else:
            logger.warning(f"Model file {params['vae_load_file_path']} does not exist, skipping loading.")
            params["load_model"] = False
            params["vae_load_file_path"] = None

    # Load or create transformer-VAE model
    prosail_vae = load_prosail_vae_with_hyperprior(
        pv_config=pv_config,
        pv_config_hyper=pv_config_hyper,
        logger_name=LOGGER_NAME
    )

    # Initialize optimizer
    lr = params['lr']
    lr_recompute_mode = params["lr_recompute_mode"]
    optimizer = optim.Adam(prosail_vae.parameters(), lr=lr, weight_decay=1e-2)
    logger.info('transformer-VAE and optimizer initialized.')
    
    logger.info(f"Starting Training loop for {params['epochs']} epochs.")

    # Execute training loop
    (all_train_loss_df, all_valid_loss_df, info_df, all_cyclical_rmse) = training_loop(
        prosail_vae,
        optimizer,
        params['epochs'],
        train_loader,
        valid_loader,
        res_dir=res_dir,
        n_samples=params["n_samples"],
        lr_recompute=params['lr_recompute'],
        exp_lr_decay=params["exp_lr_decay"],
        plot_gradient=False,  # parser.plot_results if you want
        lr_recompute_mode=lr_recompute_mode,
        cycle_training=params["cycle_training"],
        accum_iter=params["accum_iter"],
        lrs_threshold=params['lrs_threshold'],
        lr_init=params['lr'],
        validation_at_every_epoch=params["validation_at_every_epoch"],
        validation_dir=validation_dir,
        frm4veg_data_dir=frm4veg_data_dir,
        frm4veg_2021_data_dir=frm4veg_2021_data_dir,
        belsar_data_dir=belsar_data_dir,
        lai_cyclical_loader=lai_cyclical_loader,
        max_sec=10.5 * 3600
    )
    logger.info("Training Completed!")

    # Load the best model if it exists
    saved_model_path = params["vae_save_file_path"]
    if os.path.exists(saved_model_path):
        params["load_model"] = True
        params["vae_load_file_path"] = saved_model_path
        pv_config = get_prosail_vae_config(
            params,
            bands=bands,
            prosail_bands=prosail_bands,
            inference_mode=False,
            rsr_dir=parser.rsr_dir,
            io_coeffs=io_coeffs
        )

        prosail_vae = load_prosail_vae_with_hyperprior(
            pv_config=pv_config,
            pv_config_hyper=pv_config_hyper,
            logger_name=LOGGER_NAME
        )
    else:
        logger.warning(f"Model file {saved_model_path} does not exist, skipping loading.")
    
    # Save cyclical RMSE values if available
    if len(all_cyclical_rmse):
        pd.DataFrame(all_cyclical_rmse).to_csv(os.path.join(res_dir, "cyclical_rmse.csv"))
    return prosail_vae, all_train_loss_df, all_valid_loss_df, info_df

def configureEmissionTracker(parser):
    """
    Configure carbon emissions tracking for the training process.
    
    Uses the codecarbon package to track computational carbon emissions
    during model training if available.
    
    Args:
        parser (argparse.Namespace): Parsed command-line arguments.
    
    Returns:
        tuple: A tuple containing:
            - tracker: OfflineEmissionsTracker instance or None if unavailable
            - useEmissionTracker (bool): Whether emissions tracking is enabled
    """
    logger = logging.getLogger(LOGGER_NAME)
    try:
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(country_iso_code="FRA", output_dir=parser.root_results_dir)
        tracker.start()
        useEmissionTracker = True
    except:
        logger.error("Couldn't start codecarbon! Emissions not tracked for this execution.")
        useEmissionTracker = False
        tracker = None
    return tracker, useEmissionTracker

def save_array_xp_path(job_array_dir, res_dir):
    """
    Save results directory path to a tracking file for array jobs.
    
    This function is used for experiments that are part of a job array,
    to keep track of all result directories.
    
    Args:
        job_array_dir (str): Directory for the job array.
        res_dir (str): Results directory path to save.
    """
    if job_array_dir is not None:
        if not os.path.isfile(job_array_dir + "/results_directory_names.txt"):
            with open(job_array_dir + "/results_directory_names.txt", 'w') as outfile:
                outfile.write(f"{res_dir}\n")
        else:
            with open(job_array_dir + "/results_directory_names.txt", 'a') as outfile:
                outfile.write(f"{res_dir}\n")

def main():
    """
    Main execution function for training a transformer-VAE model.
    
    This function orchestrates the complete training process:
    1. Sets up the environment and loads configurations
    2. Configures emissions tracking if available
    3. Trains the transformer-VAE model
    4. Saves training results and generates validation metrics
    5. Exports results for simulated data evaluation
    
    The function handles errors gracefully and ensures proper cleanup.
    """
    # Setup environment and load configurations
    (params,
     parser,
     res_dir,
     sim_data_dir,
     s2_data_dir,
     params_sup_kl_model,
     job_array_dir,
     sup_kl_io_coeffs,
     frm4veg_data_dir,
     frm4veg_2021_data_dir,
     belsar_data_dir,
     model_name) = setup_training()

    # Configure emissions tracking
    tracker, useEmissionTracker = configureEmissionTracker(parser)
    spatial_encoder_types = ['cnn', 'rcnn']
    try:
        # Setup validation directory
        lai_cyclical_loader = None
        validation_dir = os.path.join(res_dir, "validation")
        if not Path(validation_dir).exists():
            os.makedirs(validation_dir)

        # Train the model
        (prosail_vae,
         all_train_loss_df,
         all_valid_loss_df,
         info_df) = train_prosailvae(
            params,
            parser,
            res_dir,
            params_sup_kl_model,
            sup_kl_io_coeffs=sup_kl_io_coeffs,
            validation_dir=validation_dir,
            frm4veg_data_dir=frm4veg_data_dir,
            frm4veg_2021_data_dir=frm4veg_2021_data_dir,
            belsar_data_dir=belsar_data_dir,
            lai_cyclical_loader=lai_cyclical_loader
        )

        # Plot training and validation losses
        plot_losses(
            res_dir,
            all_train_loss_df,
            all_valid_loss_df,
            info_df,
            LOGGER_NAME=LOGGER_NAME,
            plot_results=parser.plot_results
        )
        min_loss = all_valid_loss_df['rec_loss'].min() if 'rec_loss' in all_valid_loss_df.columns else all_valid_loss_df['loss_sum'].min()
        min_loss_df = pd.DataFrame({"Loss": [min_loss]})

        # Run validation on real data if available
        if (True and not socket.gethostname()==PC_SOCKET_NAME and
            frm4veg_data_dir is not None and
            frm4veg_2021_data_dir is not None and
            belsar_data_dir is not None):
            global_validation_metrics = save_validation_results(
                prosail_vae,
                validation_dir,
                frm4veg_data_dir=frm4veg_data_dir,
                frm4veg_2021_data_dir=frm4veg_2021_data_dir,
                belsar_data_dir=belsar_data_dir,
                model_name="pvae",
                method="simple_interpolate",
                mode="sim_tg_mean",
                remove_files=True,
                plot_results=parser.plot_results,
                save_reconstruction=False
            )

        # Create default metrics dataframe
        cyclical_rmse_df = pd.DataFrame(data={"cyclical_rmse": [1.0]})

        # Check if using EnMAP or Sentinel-2 configuration
        is_enmap = isinstance(prosail_vae.decoder.ssimulator, EnMapSensorSimulator)
        logger = logging.getLogger(LOGGER_NAME)

        if is_enmap:
            logger.info("Using EnMAP configuration")
            global_validation_metrics = {}
        else:
            logger.info("Using Sentinel-2 configuration")
            cyclical_rmse_df = pd.DataFrame(data={"cyclical_rmse": [0.0]})

        # Combine all metrics into a single dataframe
        global_results_df = pd.concat((
            pd.DataFrame({'model': [model_name]}),
            cyclical_rmse_df,
            min_loss_df
        ), axis=1)

        # Add validation metrics if available
        if 'global_validation_metrics' in locals() and global_validation_metrics:
            for variable, metrics in global_validation_metrics.items():
                global_results_df = pd.concat((
                    global_results_df,
                    metrics['rmse'],
                    metrics["picp"],
                    metrics['mpiw'],
                    metrics['mestdr']
                ), axis=1)

        # Save global results to CSV
        res_df_filename = os.path.join(
            os.path.join(os.path.join(res_dir, os.pardir), os.pardir),
            "model_results.csv"
        )
        if not os.path.isfile(res_df_filename):
            global_results_df.to_csv(res_df_filename, header=global_results_df.columns, index=False)
        else:
            global_results_df.to_csv(res_df_filename, mode='a', index=False, header=False)

        # Save results on simulated data for non-spatial encoders
        if not params['encoder_type'] in spatial_encoder_types:
            if is_enmap:
                # For EnMAP, skip if no test data
                if os.path.exists(os.path.join(params["sim_data_dir"], "test_prosail_enmap_sim_refl.pt")):
                    logger.info("Saving results on EnMAP simulated data")
                    save_results_on_sim_data(
                        prosail_vae,
                        res_dir,
                        params["sim_data_dir"],
                        all_train_loss_df,
                        all_valid_loss_df,
                        info_df,
                        LOGGER_NAME=LOGGER_NAME,
                        plot_results=parser.plot_results,
                        n_samples=params["n_samples"]
                    )
                else:
                    logger.info("Skipping sim data results for EnMAP: test data doesn't exist")
            else:
                # Sentinel-2
                save_results_on_sim_data(
                    prosail_vae,
                    res_dir,
                    params["sim_data_dir"],
                    all_train_loss_df,
                    all_valid_loss_df,
                    info_df,
                    LOGGER_NAME=LOGGER_NAME,
                    plot_results=parser.plot_results,
                    n_samples=params["n_samples"]
                )

        # Save job array paths
        save_array_xp_path(job_array_dir, res_dir)
        if params["k_fold"] > 1:
            save_array_xp_path(os.path.join(res_dir, os.pardir), res_dir)

    except Exception as exc:
        traceback.print_exc()
        print(exc)

    # Stop emissions tracking
    if useEmissionTracker:
        tracker.stop()

if __name__ == "__main__":
    # Execute the main training function when script is run directly
    main()
