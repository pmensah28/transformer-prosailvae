import os
import logging
import argparse
import socket
import torch
from prosailvae import __path__ as PPATH
TOP_PATH = os.path.join(PPATH[0], os.pardir)
from .metrics_utils import get_metrics, save_metrics
from .prosail_plots import (plot_metrics, plot_rec_and_latent, loss_curve, plot_param_dist, plot_pred_vs_tgt, 
                                    plot_refl_dist, pair_plot, plot_rec_error_vs_angles, plot_lat_hist2D, plot_rec_hist2D, 
                                    plot_metric_boxplot, plot_single_lat_hist_2D,
                                    all_loss_curve, regression_plot, regression_plot_2hues)
from dataset.loaders import  get_simloader
from prosailvae.ProsailSimus import PROSAILVARS, BANDS

from utils.utils import load_dict, save_dict
from utils.image_utils import get_encoded_image_from_batch, crop_s2_input
from prosailvae.prosail_vae import load_prosail_vae_with_hyperprior

from validation.validation import (get_all_campaign_lai_results, get_belsar_x_frm4veg_lai_results, get_frm4veg_ccc_results, 
                                   get_validation_global_metrics)
from datetime import datetime 
import shutil
from time import sleep
import warnings
from torchutils.patches import patchify, unpatchify 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchutils.patches import patchify

LOGGER_NAME = "PROSAIL-VAE results logger"
PC_SOCKET_NAME = 'CELL200973' # toggle options for dev and debug on PC

def get_prosailvae_results_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')

    parser.add_argument("-c", dest="config_file",
                        help="name of config json file on config directory.",
                        type=str, default="config.json")

    parser.add_argument("-d", dest="data_dir",
                        help="path to data direcotry",
                        type=str, default="/Users/princemensah/Desktop/prosailvae/data/")
    
    parser.add_argument("-r", dest="root_results_dir",
                        help="path to root results direcotry",
                        type=str, default="")

    parser.add_argument("-rsr", dest="rsr_dir",
                        help="directory of rsr_file",
                        type=str, default='/Users/princemensah/Desktop/prosailvae/data/simulated_dataset')
    
    parser.add_argument("-t", dest="tensor_dir",
                        help="directory of mmdc tensor files",
                        type=str, default="/Users/princemensah/Desktop/prosailvae/data/simulated_dataset")

    parser.add_argument("-p", dest="plot_results",
                        help="toggle results plotting",
                        type=bool, default=False)  
    return parser


    if isinstance(frm4veg_barrax_lai_pred, torch.Tensor):
        frm4veg_barrax_lai_pred = frm4veg_barrax_lai_pred.numpy()
    gdf_barrax_lai, _, _, _, _ = load_frm4veg_data(frm4veg_data_dir, frm4veg_barrax_filename, variable=variable)
    barrax_ref = gdf_barrax_lai[variable].values.reshape(-1)
    # ref_uncert = gdf_barrax_lai["uncertainty"].values
    x_idx = gdf_barrax_lai["x_idx"].values.astype(int)
    y_idx = gdf_barrax_lai["y_idx"].values.astype(int)
    barrax_pred_at_site = frm4veg_barrax_lai_pred[:, y_idx, x_idx].reshape(-1)
    site = site + ["Spain"] * len(barrax_ref)

def save_validation_results(model, res_dir,
                            frm4veg_data_dir="/Users/princemensah/Desktop/prosailvae/data/frm4veg_validation",
                            frm4veg2021_data_dir="/Users/princemensah/Desktop/prosailvae/data/frm4veg_2021_validation",
                            belsar_data_dir="/Users/princemensah/Desktop/prosailvae/databelsar_validation",
                            belsar_pred_dir=None,
                            model_name="pvae",
                            method="simple_interpolate",
                            mode="sim_tg_mean", 
                            save_reconstruction=True, 
                            remove_files=False, plot_results=False):

    # If belsar_pred_dir is not specified, use res_dir as default
    if belsar_pred_dir is None:
        belsar_pred_dir = res_dir

    (barrax_results, barrax_2021_results, wytham_results, belsar_results, all_belsar
     ) = get_all_campaign_lai_results(model, frm4veg_data_dir, frm4veg2021_data_dir, belsar_data_dir, belsar_pred_dir,
                                      mode=mode, method=method, model_name=model_name, 
                                      save_reconstruction=save_reconstruction, get_all_belsar=plot_results, 
                                      remove_files=remove_files)
    results = {}
    results["lai"] = get_belsar_x_frm4veg_lai_results(belsar_results, barrax_results, barrax_2021_results, wytham_results,
                                                      frm4veg_lai="lai", get_reconstruction_error=save_reconstruction,
                                                      bands_idx=model.encoder.bands)
    hue_elem = pd.unique(results["lai"]["Land cover"])
    hue2_elem = pd.unique(results["lai"]["Campaign"])
    hue_color_dict= {}
    for j, h_e in enumerate(hue_elem):
        hue_color_dict[h_e] = f"C{j}"
    default_markers = ["o", "v", "D", "s", "+", ".", "^", "1"]
    hue2_markers_dict= {}
    for j, h2_e in enumerate(hue2_elem):
        hue2_markers_dict[h2_e] = default_markers[j]

    results["ccc"] = get_frm4veg_ccc_results(barrax_results, barrax_2021_results, wytham_results,
                                             frm4veg_ccc="ccc", get_reconstruction_error=save_reconstruction, 
                                             bands_idx=model.encoder.bands)
    lai_dir = os.path.join(res_dir, "lai_scatter")
    os.makedirs(lai_dir, exist_ok=True)
    ccc_dir = os.path.join(res_dir, "ccc_scatter")
    os.makedirs(ccc_dir, exist_ok=True)
    scatter_dir = {"lai":lai_dir, "ccc":ccc_dir}
    global_metrics = {}

    for variable in ["lai", "ccc"]:
        (global_rmse_dict, global_picp_dict, 
         global_mpiw_dict, global_mestdr_dict) = get_validation_global_metrics(results[variable], 
                                                                               decompose_along_columns = ["Campaign"], #["Site", "Land cover", "Campaign"], 
                                                                               n_sigma=3,
                                                                               variable=variable)
        global_metrics[variable] = {"rmse":global_rmse_dict["Campaign"], "picp":global_picp_dict["Campaign"], 
                                    "mpiw":global_mpiw_dict["Campaign"], "mestdr": global_mestdr_dict["Campaign"]}
        for key, rmse_df in global_rmse_dict.items():
            rmse_df.to_csv(os.path.join(scatter_dir[variable], f"{model_name}_{key}_{variable}_validation_rmse.csv"))
        for key, pcip_df in global_picp_dict.items():
            pcip_df.to_csv(os.path.join(scatter_dir[variable], f"{model_name}_{key}_{variable}_validation_picp.csv"))
        for key, mpiw_df in global_mpiw_dict.items():
            mpiw_df.to_csv(os.path.join(scatter_dir[variable], f"{model_name}_{key}_{variable}_validation_mpiw.csv"))
        for key, mestdr_df in global_mestdr_dict.items():
            mestdr_df.to_csv(os.path.join(scatter_dir[variable], f"{model_name}_{key}_{variable}_validation_mestdr.csv"))
    
    if plot_results:
        time_series_dir = os.path.join(res_dir, "time_series")
        os.makedirs(time_series_dir, exist_ok=True)
        
        # Check if all_belsar is available for plotting
        if all_belsar is not None and all_belsar:
            # Create plots for time series
            fig, axs = plt.subplots(10, 1, dpi=150, sharex=True, tight_layout=True, figsize=(10, 2*10))
            sites = ["W1", "W2", "W3", "W4", "W5", "W6", "M1", "M2", "M3", "M4"]
            for i, site in enumerate(sites):
                fig, ax = get_belsar_sites_time_series(all_belsar, belsar_data_dir, site=site, fig=fig, ax=axs[i], 
                                                      label=f"{model_name}")
            
            fig.savefig(os.path.join(time_series_dir, f"{model_name}_belsar_time_series.pdf"))
            plt.close(fig)
            
            sites = ["W1", "W2", "W3", "W4", "W5"]
            fig, ax = plt.subplots(dpi=150, tight_layout=True)
            for i, site in enumerate(sites):
                fig, ax = get_belsar_sites_time_series(all_belsar, belsar_data_dir, site=site, fig=fig, ax=ax, 
                                                      label=f"{site}")
            
            fig.savefig(os.path.join(time_series_dir, f"{model_name}_wheat_time_series.pdf"))
            plt.close(fig)
            
            sites = ["M1", "M2", "M3", "M4", "M5"]
            fig, ax = plt.subplots(dpi=150, tight_layout=True)
            for i, site in enumerate(sites):
                fig, ax = get_belsar_sites_time_series(all_belsar, belsar_data_dir, site=site, fig=fig, ax=ax, 
                                                      label=f"{site}")
            
            fig.savefig(os.path.join(time_series_dir, f"{model_name}_maize_time_series.pdf"))
            plt.close(fig)
            
            # Create grouped plots
            fig, axs = plt.subplots(2, 1, sharex=True, dpi=150, tight_layout=True, figsize=(7, 2*2.5))
            sites = ["W1", "W2", "W3", "W4", "W5", "W6"]
            for i, site in enumerate(sites):
                fig, ax = get_belsar_sites_time_series(all_belsar, belsar_data_dir, site=site, fig=fig, ax=axs[0], 
                                                      label=f"{site}")
            
            sites = ["M1", "M2", "M3", "M4", "M5"]
            for i, site in enumerate(sites):
                fig, ax = get_belsar_sites_time_series(all_belsar, belsar_data_dir, site=site, fig=fig, ax=axs[1], 
                                                      label=f"{site}")
            axs[0].set_title("Wheat fields")
            axs[1].set_title("Maize fields")
            
            fig.savefig(os.path.join(time_series_dir, f"{model_name}_belsar_time_series_by_crop.pdf"))
            plt.close(fig)
        else:
            logger = logging.getLogger(LOGGER_NAME)
            logger.warning("Skipping BelSAR time series plots because all_belsar data is not available")
        
        # Create regression plots for each variable
        for variable in ["lai", "ccc"]:
            # By Campaign
            fig, ax = plt.subplots(dpi=150)
            
            _, _ = regression_plot(results[variable], x=f"Predicted {variable}", y=f"{variable}", 
                                 fig=fig, ax=ax, hue="Campaign")
            tikzplotlib_fix_ncols(fig)
            if TIKZPLOTLIB_AVAILABLE:
                tikzplotlib.save(os.path.join(scatter_dir[variable], f"{model_name}_{variable}_regression_Campaign.tex"))
            fig.savefig(os.path.join(scatter_dir[variable], f"{model_name}_{variable}_regression_Campaign.png"))
            plt.close(fig)
            
            # By Land cover
            fig, ax = plt.subplots(dpi=150)
            
            _, _ = regression_plot(results[variable], x=f"Predicted {variable}", y=f"{variable}", 
                                 fig=fig, ax=ax, hue="Land cover")
            tikzplotlib_fix_ncols(fig)
            if TIKZPLOTLIB_AVAILABLE:
                tikzplotlib.save(os.path.join(scatter_dir[variable], f"{model_name}_{variable}_regression_Land_cover.tex"))
            fig.savefig(os.path.join(scatter_dir[variable], f"{model_name}_{variable}_regression_Land_cover.png"))
            plt.close(fig)
            
            # By Site
            fig, ax = plt.subplots(dpi=150)
            
            _, _ = regression_plot(results[variable], x=f"Predicted {variable}", y=f"{variable}", 
                                 fig=fig, ax=ax, hue="Site")
            tikzplotlib_fix_ncols(fig)
            if TIKZPLOTLIB_AVAILABLE:
                tikzplotlib.save(os.path.join(scatter_dir[variable], f"{model_name}_{variable}_regression_Site.tex"))
            fig.savefig(os.path.join(scatter_dir[variable], f"{model_name}_{variable}_regression_Site.png"))
            plt.close(fig)
            
            # With two hues
            fig, ax = plt.subplots(dpi=150)
            
            _, _ = regression_plot_2hues(results[variable], x=f"Predicted {variable}", y=f"{variable}", 
                                      fig=fig, ax=ax, hue="Land cover", hue2="Campaign",
                                      hue_color_dict=hue_color_dict, hue2_markers_dict=hue2_markers_dict,
                                      title_hue="Land cover", title_hue2="Campaign")
            tikzplotlib_fix_ncols(fig)
            if TIKZPLOTLIB_AVAILABLE:
                tikzplotlib.save(os.path.join(scatter_dir[variable], f"{model_name}_{variable}_regression_2hues.tex"))
            fig.savefig(os.path.join(scatter_dir[variable], f"{model_name}_{variable}_regression_2hues.png"))
            plt.close(fig)
            
            # Error vs reconstruction error
            fig, ax = plt.subplots(dpi=150)
            
            _, _ = regression_plot(results[variable], x=f"Reconstruction error", y=f"abs error {variable}", 
                                 fig=fig, ax=ax, hue="Campaign")
            tikzplotlib_fix_ncols(fig)
            if TIKZPLOTLIB_AVAILABLE:
                tikzplotlib.save(os.path.join(scatter_dir[variable], f"{model_name}_{variable}_error_vs_reconstruction_error_Campaign.tex"))
            fig.savefig(os.path.join(scatter_dir[variable], f"{model_name}_{variable}_error_vs_reconstruction_error_Campaign.png"))
            plt.close('all')

    return global_metrics
        
    
def get_rec_var(PROSAIL_VAE, loader, max_batch=50, n_samples=10, sample_dim=1, bands_dim=2, n_bands=10):
    with torch.no_grad():
        all_rec_var = []
        for i, batch in enumerate(loader):
            if i==max_batch:
                break
            s2_r = patchify(batch[0].squeeze(0), patch_size=32, margin=0).to(PROSAIL_VAE.device)
            s2_r = s2_r.reshape(-1, *s2_r.shape[2:])
            s2_a = patchify(batch[1].squeeze(0), patch_size=32, margin=0).to(PROSAIL_VAE.device)
            s2_a = s2_a.reshape(-1, *s2_a.shape[2:])
            for j in range(s2_a.size(0)):
                _, _, _, rec = PROSAIL_VAE.forward(s2_r[j,...].unsqueeze(0), 
                                                   n_samples=n_samples, 
                                                   angles=s2_a[j,...].unsqueeze(0))
                rec_var = rec.var(sample_dim)
                rec_var = rec_var.transpose(bands_dim,0).reshape(n_bands,-1)
            all_rec_var.append(rec_var.cpu())
    return torch.cat(all_rec_var, 1)

def plot_losses(res_dir, all_train_loss_df=None, all_valid_loss_df=None, info_df=None, 
                LOGGER_NAME='PROSAIL-VAE logger', plot_results=False,):
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("Saving Loss")
    # Saving Loss
    loss_dir = res_dir + "/loss/"
    if not os.path.isdir(loss_dir):
        os.makedirs(loss_dir)
    
    if all_train_loss_df is not None:
        all_train_loss_df.to_csv(loss_dir + "train_loss.csv")
        if plot_results:
            loss_curve(all_train_loss_df, save_file=loss_dir + "train_loss.svg")
            loss_curve(all_train_loss_df[["epoch", "loss_sum"]], save_file=loss_dir + "train_loss_sum.svg")
    if all_valid_loss_df is not None:
        all_valid_loss_df.to_csv(loss_dir + "valid_loss.csv")
        if plot_results:
            loss_curve(all_valid_loss_df, save_file=loss_dir+"valid_loss.svg")
            loss_curve(all_valid_loss_df[["epoch", "loss_sum"]], save_file=loss_dir + "train_loss_sum.svg")
    if info_df is not None:
        if plot_results:
            loss_curve(info_df, save_file=loss_dir+"lr.svg")
            all_loss_curve(all_train_loss_df[["epoch", "loss_sum"]], all_valid_loss_df[["epoch", "loss_sum"]], 
                           info_df, save_file=loss_dir+"all_loss_sum.svg")
            all_loss_curve(all_train_loss_df, all_valid_loss_df, info_df, save_file=loss_dir+"all_loss.svg")


def save_results_on_sim_data(PROSAIL_VAE, res_dir, data_dir, all_train_loss_df=None,
                 all_valid_loss_df=None, info_df=None, LOGGER_NAME='PROSAIL-VAE logger', 
                 plot_results=False, juan_validation=True, n_samples=1,
                 lai_cyclical_loader=None):
    
    bands_name = np.array(BANDS)[PROSAIL_VAE.encoder.bands.cpu()].tolist()
    device = PROSAIL_VAE.device
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("Saving Loss")
    # Saving Loss
    loss_dir = res_dir + "/loss/"
    if not os.path.isdir(loss_dir):
        os.makedirs(loss_dir)
    
    if all_train_loss_df is not None:
        all_train_loss_df.to_csv(loss_dir + "train_loss.csv")
        if plot_results:
            loss_curve(all_train_loss_df, save_file=loss_dir+"train_loss.svg")
    if all_valid_loss_df is not None:
        all_valid_loss_df.to_csv(loss_dir + "valid_loss.csv")
        if plot_results:
            loss_curve(all_valid_loss_df, save_file=loss_dir+"valid_loss.svg")
    if info_df is not None:
        if plot_results:
            loss_curve(info_df, save_file=loss_dir+"lr.svg")
            all_loss_curve(all_train_loss_df, all_valid_loss_df, info_df, 
                           save_file=loss_dir+"all_loss.svg")

    logger.info("Loading test loader...")
    loader = get_simloader(file_prefix="test_", data_dir=data_dir)
    logger.info("Test loader, loaded.")
    
    alpha_pi = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    alpha_pi.reverse()
    PROSAIL_VAE.eval()
    logger.info("Computing inference metrics with test dataset...")
    test_loss = PROSAIL_VAE.validate(loader, n_samples=n_samples)
    pd.DataFrame(test_loss, index=[0]).to_csv(loss_dir + "/test_loss.csv")
    nlls = PROSAIL_VAE.compute_lat_nlls(loader).mean(0).squeeze()
    torch.save(nlls, res_dir + "/params_nll.pt")

    if plot_results:
        plot_rec_hist2D(PROSAIL_VAE, loader, res_dir, nbin=50, bands_name=bands_name)
    (mae, mpiw, picp, mare, 
    sim_dist, tgt_dist, rec_dist,
    angles_dist, s2_r_dist,
    sim_pdfs, sim_supports, ae_percentiles, 
    are_percentiles, piw_percentiles) = get_metrics(PROSAIL_VAE, loader, 
                              n_pdf_sample_points=3001,
                              alpha_conf=alpha_pi)
    logger.info("Metrics computed.")

    save_metrics(res_dir, mae, mpiw, picp, alpha_pi, 
                ae_percentiles, are_percentiles, piw_percentiles, var_bounds_type=PROSAIL_VAE.sim_space.var_bounds)
    maer = pd.read_csv(res_dir+"/metrics/maer.csv").drop(columns=["Unnamed: 0"])
    mpiwr = pd.read_csv(res_dir+"/metrics/mpiwr.csv").drop(columns=["Unnamed: 0"])
    if plot_results:
        # Plotting results
        metrics_dir = res_dir + "/metrics_plot/"
        if not os.path.isdir(metrics_dir):
            os.makedirs(metrics_dir)
        
        logger.info("Plotting metrics.")
        
        plot_metrics(metrics_dir, alpha_pi, maer, mpiwr, picp, mare)
        plot_metric_boxplot(ae_percentiles, res_dir, metric_name='ae', logscale=True)
        plot_metric_boxplot(are_percentiles, res_dir, metric_name='are')
        # plot_metric_boxplot(piw_percentiles, res_dir, metric_name='piw')
        rec_dir = res_dir + "/reconstruction/"
        if not os.path.isdir(rec_dir):
            os.makedirs(rec_dir)
        logger.info("Plotting reconstructions")
        plot_rec_and_latent(PROSAIL_VAE, loader, rec_dir, n_plots=20, bands_name=bands_name)
        
        logger.info("Plotting PROSAIL parameter distributions")
        plot_param_dist(metrics_dir, sim_dist, tgt_dist, var_bounds_type=PROSAIL_VAE.sim_space.var_bounds)
        logger.info("Plotting PROSAIL parameters, reference vs prediction")
        plot_lat_hist2D(tgt_dist, sim_pdfs, sim_supports, res_dir, nbin=50)
        plot_pred_vs_tgt(metrics_dir, sim_dist, tgt_dist, var_bounds_type=PROSAIL_VAE.sim_space.var_bounds)
        ssimulator = PROSAIL_VAE.decoder.ssimulator
        refl_dist = loader.dataset[:][0]
        plot_refl_dist(rec_dist, refl_dist, res_dir, normalized=False, 
                    ssimulator=PROSAIL_VAE.decoder.ssimulator)
        
        normed_rec_dist =  ssimulator.normalize(rec_dist.to(device))
        normed_refl_dist =  ssimulator.normalize(refl_dist.to(device)) 
        logger.info("Plotting reflectance distribution")
        plot_refl_dist(normed_rec_dist, normed_refl_dist, metrics_dir, normalized=True, 
                       ssimulator=PROSAIL_VAE.decoder.ssimulator, bands_name=bands_name)
        logger.info("Plotting reconstructed reflectance components pair plots")
        pair_plot(normed_rec_dist, tensor_2=None, features = BANDS, 
                res_dir=metrics_dir, filename='normed_rec_pair_plot.png')
        logger.info("Plotting reference reflectance components pair plots")
        pair_plot(normed_refl_dist, tensor_2=None, features = BANDS, 
                res_dir=metrics_dir, filename='normed_s2bands_pair_plot.png')
        logger.info("Plotting inferred PROSAIL parameters pair plots")
        pair_plot(sim_dist.squeeze(), tensor_2=None, features = PROSAILVARS, 
                res_dir=metrics_dir, filename='sim_prosail_pair_plot.png')
        logger.info("Plotting reference PROSAIL parameters pair plots")
        pair_plot(tgt_dist.squeeze(), tensor_2=None, features = PROSAILVARS, 
                res_dir=metrics_dir, filename='ref_prosail_pair_plot.png')
        logger.info("Plotting reconstruction error against angles")
        plot_rec_error_vs_angles(s2_r_dist, rec_dist, angles_dist,  res_dir=metrics_dir)
    
    logger.info("Program completed.")
    return


def check_fold_res_dir(fold_dir, n_xp, params):
    same_fold = ""
    all_dirs = os.listdir(fold_dir)
    for d in all_dirs:
        if d.startswith(f"{n_xp}_kfold_{params['k_fold']}_n_{params['n_fold']}") :
            same_fold = d
    return same_fold

def get_res_dir_path(root_results_dir, params, n_xp=None, overwrite_xp=False):
    
    if not os.path.exists(root_results_dir):
        os.makedirs(root_results_dir)
    if not os.path.exists(root_results_dir+"n_xp.json"):    
        save_dict({"xp":0}, root_results_dir+"n_xp.json")
    if n_xp is None:
        n_xp = load_dict(root_results_dir+"n_xp.json")['xp']+1
    save_dict({"xp":n_xp}, root_results_dir+"n_xp.json")
    date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if params['k_fold']>1:
        k_fold_dir = f"{root_results_dir}/{n_xp}_kfold_{params['k_fold']}_supervised_{params['supervised']}_{params['dataset_file_prefix']}"
        if not params['supervised']:
            k_fold_dir + f"kl_{params['beta_kl']}"
        if not os.path.exists(k_fold_dir):
            os.makedirs(k_fold_dir)    
        res_dir = f"{k_fold_dir}/{n_xp}_kfold_{params['k_fold']}_n_{params['n_fold']}_d{date}_supervised_{params['supervised']}_{params['dataset_file_prefix']}"
        same_fold_dir = check_fold_res_dir(k_fold_dir, n_xp, params)
        if len(same_fold_dir)>0:
            if overwrite_xp:
                warnings.warn("WARNING: Overwriting existing fold experiment in 5s")
                sleep(5)
                shutil.rmtree(k_fold_dir + "/"+ same_fold_dir)
            else:
                raise ValueError(f"The same experiment (fold) has already been carried out at {same_fold_dir}.\n Please change the number of fold or allow overwrite")
    else:
        if not socket.gethostname()==PC_SOCKET_NAME:
            res_dir = f"{root_results_dir}/{n_xp}"#_d{date}_supervised_{params['supervised']}_{params['dataset_file_prefix']}"
        else:
            res_dir = f"{root_results_dir}/{n_xp}_d{date}_supervised_{params['supervised']}_{params['dataset_file_prefix']}"
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)    
    return res_dir

# def setupResults():
#     if socket.gethostname()==PC_SOCKET_NAME:
#         args=["-d", "/Users/princemensah/Desktop/prosailvae/data/",
#               "-r", "",
#               "-rsr", '/Users/princemensah/Desktop/prosailvae/data/',
#               "-t", "/Users/princemensah/Desktop/prosailvae/data/"]
        
#         parser = get_prosailvae_results_parser().parse_args(args)    
#     else:
#         parser = get_prosailvae_results_parser().parse_args()
#     root_dir = os.path.join(os.path.dirname(prosailvae.__file__), os.pardir)

#     if len(parser.data_dir)==0:
#         data_dir = os.path.join(root_dir,"data/")
#     else:
#         data_dir = parser.data_dir

#     if len(parser.root_results_dir)==0:
#         res_dir = os.path.join(os.path.join(os.path.dirname(prosailvae.__file__),
#                                                      os.pardir),"results/")
#     else:
#         res_dir = parser.root_results_dir    
#     params = load_dict(res_dir + "/config.json")
#     if params["supervised"]:
#         params["simulated_dataset"]=True
#     params["n_fold"] = parser.n_fold if params["k_fold"] > 1 else None

#     params_sup_kl_model = None
#     if params["supervised_kl"]:
#         params_sup_kl_model = load_dict(res_dir+"/sup_kl_model_config.json")
#         params_sup_kl_model['sup_model_weights_path'] = res_dir+"/sup_kl_model_weights.tar"
    
#     logging.basicConfig(filename=res_dir+'/results_log.log', 
#                               level=logging.INFO, force=True)
#     logger_name = LOGGER_NAME
#     # create logger
#     logger = logging.getLogger(logger_name)
#     logger.info('Starting computation of results of PROSAIL-VAE.')
#     logger.info('========================================================================')
#     logger.info('Parameters are : ')
#     for _, key in enumerate(params):
#         logger.info(f'{key} : {params[key]}')
#     logger.info('========================================================================')

#     return params, parser, res_dir, data_dir, params_sup_kl_model
    
def configureEmissionTracker(parser):
    logger = logging.getLogger(LOGGER_NAME)
    try:
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(country_iso_code="FRA", output_dir=parser.root_results_dir)
        tracker.start()
        useEmissionTracker = True
    except:
        logger.error("Couldn't start codecarbon ! Emissions not tracked for this execution.")
        useEmissionTracker = False
        tracker = None
    return tracker, useEmissionTracker

# def main():
#     params, parser, res_dir, data_dir, params_sup_kl_model = setupResults()
#     tracker, useEmissionTracker = configureEmissionTracker(parser)
#     try:
#         vae_file_path = res_dir + '/prosailvae_weights.tar'
#         PROSAIL_VAE = load_prosail_vae_with_hyperprior(params, parser.rsr_dir, data_dir, 
#                                 logger_name=LOGGER_NAME, vae_file_path=vae_file_path, params_sup_kl_model=params_sup_kl_model)
#         save_results(PROSAIL_VAE, res_dir, data_dir, LOGGER_NAME=LOGGER_NAME, plot_results=parser.plot_results)
#     except Exception as e:
#         traceback.print_exc()
#         print(e)
#     if useEmissionTracker:
#         tracker.stop()
#     pass
#     pass

# if __name__ == "__main__":
#     main()