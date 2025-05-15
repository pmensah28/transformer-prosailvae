import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os 
from prosailvae.ProsailSimus import PROSAILVARS
from prosailvae.prosail_var_dists import get_prosail_vars_interval_width
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from utils.image_utils import get_encoded_image_from_batch
from sklearn.metrics import r2_score
import logging

def save_metrics(res_dir, mae, mpiw, picp, alpha_pi, ae_percentiles, are_percentiles, piw_percentiles, var_bounds_type="legacy"):
    metrics_dir = res_dir + "/metrics/"
    if not os.path.isdir(metrics_dir):
        os.makedirs(metrics_dir)
    pd.DataFrame(data=mae.view(1,len(PROSAILVARS)).detach().cpu().numpy(), columns=PROSAILVARS, 
                 index=[0]).to_csv(metrics_dir + "/mae.csv")
    df_mpwi = pd.DataFrame(data=mpiw.view(-1, len(PROSAILVARS)).detach().cpu().numpy(), 
                           columns=PROSAILVARS)
    df_mpwi["alpha"] = alpha_pi
    df_mpwi.to_csv(metrics_dir + "/mpiw.csv")
    df_picp = pd.DataFrame(data=picp.view(-1, len(PROSAILVARS)).detach().cpu().numpy(), 
                           columns=PROSAILVARS)
    df_picp["alpha"] = alpha_pi
    df_picp.to_csv(metrics_dir + "/picp.csv")
    
    interval_length = get_prosail_vars_interval_width(bounds_type=var_bounds_type).to(mpiw.device)
    mpiwr = (mpiw / interval_length.view(-1,1)).transpose(0,1)
    maer = mae / interval_length
    df_maer = pd.DataFrame(data=maer.view(-1, len(PROSAILVARS)).detach().cpu().numpy(), 
                           columns=PROSAILVARS)
    df_mpiwr = pd.DataFrame(data=mpiwr.view(-1, len(PROSAILVARS)).detach().cpu().numpy(), 
                           columns=PROSAILVARS)
    df_maer.to_csv(metrics_dir + "/maer.csv")
    df_mpiwr.to_csv(metrics_dir + "/mpiwr.csv")
    torch.save(ae_percentiles, metrics_dir + '/ae_percentiles.pt')
    aer_percentiles = ae_percentiles / interval_length.view(1,-1).detach().cpu().numpy()
    torch.save(aer_percentiles, metrics_dir + '/aer_percentiles.pt')
    torch.save(are_percentiles, metrics_dir + '/are_percentiles.pt')
    # torch.save(piw_percentiles, '/piw_percentiles.pt')

def get_percentiles_from_box_plots(bp):
    percentiles = torch.zeros((5,len(bp['boxes'])))
    for i in range(len(bp['boxes'])):
        percentiles[0,i] = torch.from_numpy(np.asarray(bp['caps'][2*i].get_ydata()[0]))
        percentiles[1,i] = torch.from_numpy(np.asarray(bp['boxes'][i].get_ydata()[0]))
        percentiles[2,i] = torch.from_numpy(np.asarray(bp['medians'][i].get_ydata()[0]))
        percentiles[3,i] = torch.from_numpy(np.asarray(bp['boxes'][i].get_ydata()[2]))
        percentiles[4,i] = torch.from_numpy(np.asarray(bp['caps'][2*i + 1].get_ydata()[0]))
                        #    (bp['fliers'][i].get_xdata(),
                        #     bp['fliers'][i].get_ydata()))
    return percentiles

def get_box_plot_percentiles(tensor):
    fig, ax = plt.subplots()
    all_tensor_percentiles = torch.zeros((5, tensor.size(1)))
    for i in range(tensor.size(1)):
        bp = ax.boxplot([tensor[:,i].numpy(),])
        percentiles = get_percentiles_from_box_plots(bp)
        all_tensor_percentiles[:,i] = percentiles.squeeze()
    return all_tensor_percentiles

def get_metrics(PROSAIL_VAE, loader,  
                n_pdf_sample_points=3001,
                alpha_conf=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]):
    
    
    device = PROSAIL_VAE.device
    error = torch.tensor([]).to(device)
    rel_error = torch.tensor([]).to(device)
    pic = torch.tensor([]).to(device)
    piw = torch.tensor([]).to(device)
    sim_dist = torch.tensor([]).to(device)
    pi_lower = (np.array(alpha_conf)/2).tolist()
    pi_upper = (1-np.array(alpha_conf)/2).tolist()
    tgt_dist = torch.tensor([]).to(device)
    rec_dist = torch.tensor([]).to(device)
    s2_r_dist = torch.tensor([]).to(device)
    angles_dist = torch.tensor([]).to(device)
    sim_pdfs = torch.tensor([]).to(device)
    sim_supports = torch.tensor([]).to(device)
    ssimulator = PROSAIL_VAE.decoder.ssimulator
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc='Computing metrics', leave=True)):
            s2_r = batch[0].to(device)
            s2_r_dist = torch.concat([s2_r_dist, s2_r], axis=0)
            angles = batch[1].to(device)
            tgt = batch[2].to(device)
            dist_params, z_mode, prosail_params_mode, rec = PROSAIL_VAE.point_estimate_rec(s2_r, angles, mode='sim_mode')
            lat_pdfs, lat_supports = PROSAIL_VAE.lat_space.latent_pdf(dist_params)
            sim_pdfs_i, sim_supports_i = PROSAIL_VAE.sim_space.sim_pdf(lat_pdfs, lat_supports, n_pdf_sample_points=n_pdf_sample_points)
            sim_pdfs = torch.concat([sim_pdfs, sim_pdfs_i], axis=0)
            sim_supports = torch.concat([sim_supports, sim_supports_i], axis=0)
            pheno_pi_lower = PROSAIL_VAE.sim_space.sim_quantiles(lat_pdfs, lat_supports, alpha=pi_lower, n_pdf_sample_points=n_pdf_sample_points)
            pheno_pi_upper = PROSAIL_VAE.sim_space.sim_quantiles(lat_pdfs, lat_supports, alpha=pi_upper, n_pdf_sample_points=n_pdf_sample_points)
            error_i = prosail_params_mode.squeeze() - tgt
            tgt_dist = torch.concat([tgt_dist, tgt], axis=0)
            error = torch.concat([error, error_i], axis=0)
            sim_dist = torch.concat([sim_dist, prosail_params_mode], axis=0)
            rec_dist = torch.concat([rec_dist, ssimulator.normalize(rec.squeeze(), bands_dim=1)], axis=0)
            rel_error_i = (prosail_params_mode.squeeze() - tgt).abs() / (tgt.abs()+1e-10)
            rel_error = torch.concat([rel_error, rel_error_i], axis=0)
            piw_i = pheno_pi_upper - pheno_pi_lower
            piw = torch.concat([piw, piw_i], axis=0)
            pic_i = torch.logical_and(tgt.unsqueeze(2) > pheno_pi_lower, 
                                      tgt.unsqueeze(2) < pheno_pi_upper).float()
            pic = torch.concat([pic, pic_i], axis=0)
            angles_dist = torch.concat([angles_dist, angles], axis=0)
    mae = error.abs().mean(axis=0)     
    ae_percentiles = get_box_plot_percentiles(error.abs().detach().cpu())
    picp = pic.mean(axis=0)    
    mpiw = piw.mean(axis=0)
    piw_percentiles = None # get_box_plot_percentiles(piw.detach().cpu())
    mare = rel_error.mean(axis=0)
    are_percentiles = get_box_plot_percentiles(rel_error.detach().cpu())

    return (mae, mpiw, picp, mare, sim_dist, tgt_dist, rec_dist, angles_dist, s2_r_dist, sim_pdfs, 
            sim_supports, ae_percentiles, are_percentiles, piw_percentiles)

def regression_metrics(x_ref, x):
    """
    Compute linear regression metrics between reference and prediction.
    
    Parameters:
    -----------
    x_ref : np.ndarray or torch.Tensor
        Reference values
    x : np.ndarray or torch.Tensor
        Predicted values
        
    Returns:
    --------
    m : float
        Slope of the regression line
    b : float
        Intercept of the regression line
    r2 : float
        Coefficient of determination
    rmse : float
        Root mean squared error
    """
    logger = logging.getLogger("Metrics")
    
    # Convert torch tensors to numpy arrays
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x_ref, torch.Tensor):
        x_ref = x_ref.detach().cpu().numpy()
    
    assert isinstance(x_ref, np.ndarray)
    assert isinstance(x, np.ndarray)
    
    try:
        # Check for NaN or infinite values
        if np.isnan(x_ref).any() or np.isnan(x).any() or np.isinf(x_ref).any() or np.isinf(x).any():
            logger.warning("NaN or infinite values detected in regression inputs, using fallback approach")
            m, b = 1.0, 0.0  # Identity line as fallback
        else:
            # Try regular polyfit
            m, b = np.polyfit(x_ref, x, 1)
        
        # Calculate R² and RMSE
        # For R², handle case where all values are the same (division by zero)
        if np.allclose(x_ref, x_ref.mean()):
            r2 = 1.0 if np.allclose(x, x_ref) else 0.0
        else:
            r2 = r2_score(x_ref, x)
        
        rmse = np.sqrt(np.mean((x_ref - x)**2))
        
    except (np.linalg.LinAlgError, ValueError) as e:
        # Fallback to simple metrics if regression fails
        logger.warning(f"Regression calculation failed: {str(e)}. Using fallback approach.")
        m, b = 1.0, 0.0  # Identity line as fallback
        r2 = 0.0
        rmse = np.sqrt(np.mean((x_ref - x)**2)) if not np.isnan(x_ref).any() and not np.isnan(x).any() else 0.0
    
    return m, b, r2, rmse

