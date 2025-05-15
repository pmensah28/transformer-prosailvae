#!/usr/bin/env python3
"""
PROSAIL simulation module for generating and processing spectral data.
"""

from pathlib import Path
from typing import Tuple, Optional, List, Dict, Union
import numpy as np
import prosail
import torch

# from prosail import spectral_lib
from prosail.sail_model import init_prosail_spectra
from scipy.interpolate import interp1d
from scipy.signal import decimate

from prosailvae.spectral_indices import get_spectral_idx
from utils.utils import gaussian_nll_loss, standardize, unstandardize


def subsample_spectra(tensor, R_down=1, axis=0, method="interp"):
    """
    Subsample spectral data along a specified axis.
    
    Args:
        tensor (torch.Tensor): Input spectral data tensor
        R_down (int): Downsampling factor
        axis (int): Axis along which to perform subsampling
        method (str): Subsampling method ("interp", "block_mean", or "decimate")
        
    Returns:
        torch.Tensor: Subsampled tensor
    """
    if R_down > 1 :
        assert 2100 % R_down == 0
        if method=='block_mean':
            if tensor.size(0)==2101:
                tensor = tensor[:-1].reshape(-1, R_down).mean(1)
        elif method=="decimate":
            device = tensor.device
            decimated_array = decimate(tensor.detach().cpu().numpy(), R_down).copy()
            tensor = torch.from_numpy(decimated_array).to(device)
        elif method == "interp":
            device = tensor.device
            f = interp1d(np.arange(400,2501), tensor.detach().cpu().numpy())
            sampling = np.arange(400, 2501, R_down)
            array = np.apply_along_axis(f, 0, sampling)
            tensor = torch.from_numpy(array).float().to(device)
        else:
            raise NotImplementedError
    return tensor

PROSAILVARS = ["N", "cab", "car", "cbrown", "cw", "cm",
               "lai", "lidfa", "hspot", "psoil", "rsoil"]
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

def get_bands_idx(weiss_bands=False):
    """
    Get indices of bands to use in reflectance tensor.
    
    Args:
        weiss_bands (bool): Whether to use reduced set of bands (removing B2 and B8)
        
    Returns:
        Tuple[torch.Tensor, List[int]]: (band indices, prosail band indices)
    """
    # Default: all 10 bands
    bands = torch.arange(10)  # Bands are supposed to come in number order, not by resolution group
    prosail_bands = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
    
    if weiss_bands:  # Removing B2 and B8
        print("Using reduced band set (without B2 and B8)")
        bands = torch.tensor([1, 2, 3, 4, 5, 7, 8, 9])  # removing b2 and b8
        prosail_bands = [2, 3, 4, 5, 6, 8, 11, 12]
    
    return bands, prosail_bands

def apply_along_axis(function, x, fn_arg, axis: int = 0):
    """
    Apply a function along a specified axis of a tensor.
    
    Args:
        function: Function to apply
        x (torch.Tensor): Input tensor
        fn_arg: Arguments to pass to function
        axis (int): Axis along which to apply function
        
    Returns:
        torch.Tensor: Result of applying function
    """
    return torch.stack([
        function(x_i, fn_arg) for x_i in torch.unbind(x, dim=axis)
    ], dim=axis)

def decimate_1Dtensor(tensor, R_down=1):
    """
    Decimate a 1D tensor by a factor R_down.
    
    Args:
        tensor (torch.Tensor): Input 1D tensor
        R_down (int): Decimation factor
        
    Returns:
        torch.Tensor: Decimated tensor
    """
    device=tensor.device
    decimated_array = decimate(tensor.detach().cpu().numpy(), R_down).copy()
    return torch.from_numpy(decimated_array).to(device)

# def subsample_tensor(tensor, R_down=1, axis=0, method='block_mean'):
#     if R_down > 1 :
#         assert 2100 % R_down == 0
#         if method=='block_mean':
#             size = torch.as_tensor(tensor.size())
#             axis_len = size[axis]
#             if axis_len == 2101:
#                 tensor = tensor.gather(axis, torch.arange(2100))
#                 size = torch.as_tensor(tensor.size())
#             resized = torch.zeros(len(size)+1)
#             resized[axis] = 2100//R_down
#             resized[axis+1] = R_down
#             resized[axis+2:] = size[axis+1:]
#             tensor = tensor.reshape(resized.int().numpy().tolist()).mean(axis)
#         else:
#             if len(tensor.size())==1:
#                 tensor = decimate_1Dtensor(tensor, R_down=R_down)
#             elif len(tensor.size())==2:
#                 axis = 0 if axis == 1 else 1
#                 tensor = apply_along_axis(decimate_1Dtensor, tensor, R_down, axis=axis)
#             else:
#                 raise NotImplementedError
#     return tensor

class SensorSimulator():
    """
    Base class for simulating sensor responses from PROSAIL model output.
    
    This class handles the conversion of full spectrum reflectance data into
    sensor-specific band responses, including:
    - Loading and applying spectral response functions (RSR)
    - Normalizing reflectance values
    - Computing spectral indices
    - Handling different sensor configurations
    
    Attributes:
        rsr (torch.Tensor): Relative spectral response functions
        bands (List[int]): Band indices to use
        device (str): Device to use for computations
        prospect_range (Tuple[int, int]): Wavelength range for PROSAIL
        bands_loc (torch.Tensor): Location parameter for band normalization
        bands_scale (torch.Tensor): Scale parameter for band normalization
        apply_norm (bool): Whether to apply normalization
        R_down (int): Downsampling factor
    """

    def __init__(self,
                 rsr_file: str,
                 prospect_range: Tuple[int, int] = (400, 2500),
                 bands=[1, 2, 3, 4, 5, 6, 7, 8, 11, 12],
                 device='cpu',
                 bands_loc=None,
                 bands_scale=None,
                 idx_loc=None,
                 idx_scale=None,
                 apply_norm=True,
                 R_down=1,
                 rsr_dir=None):
        """
        Initialize sensor simulator.
        
        Args:
            rsr_file (str): Path to relative spectral response file
            prospect_range (Tuple[int, int]): PROSAIL wavelength range
            bands (List[int]): Band indices to use
            device (str): Computation device ('cpu' or 'cuda')
            bands_loc (torch.Tensor, optional): Band normalization location
            bands_scale (torch.Tensor, optional): Band normalization scale
            idx_loc (torch.Tensor, optional): Index normalization location
            idx_scale (torch.Tensor, optional): Index normalization scale
            apply_norm (bool): Whether to apply normalization
            R_down (int): Downsampling factor
            rsr_dir (str, optional): Directory containing RSR file
        """
        super().__init__()
        self.R_down = R_down
        self.bands = bands
        self.device = device
        self.prospect_range = prospect_range
        
        # Handle RSR file path
        if rsr_dir is not None:
            rsr_path = Path(rsr_dir) / rsr_file
        else:
            rsr_path = rsr_file
            
        # Load and process RSR data
        self.rsr = torch.from_numpy(np.loadtxt(rsr_path, unpack=True)).to(device)
        self.nb_bands = self.rsr.shape[0] - 2
        self.rsr_range = (int(self.rsr[0, 0].item() * 1000),
                          int(self.rsr[0, -1].item() * 1000))
        self.nb_lambdas = prospect_range[1] - prospect_range[0] + 1
        
        # Create PROSAIL wavelength grid
        self.rsr_prospect = torch.zeros([self.rsr.shape[0], self.nb_lambdas]).to(device)
        self.rsr_prospect[0, :] = torch.linspace(prospect_range[0],
                                                prospect_range[1],
                                                self.nb_lambdas).to(device)
        
        # Copy RSR data to appropriate range
        self.rsr_prospect[1:, 
                          :-(self.prospect_range[1] -
                                self.rsr_range[1])] = self.rsr[1:, (
                                    self.prospect_range[0] -
                                    self.rsr_range[0]):]

        # Extract solar spectrum and RSR
        self.solar = self.rsr_prospect[1, :].unsqueeze(0)
        self.rsr = self.rsr_prospect[2:, :].unsqueeze(0)
        self.rsr = self.rsr[:,bands,:]

        # Set up normalization parameters
        bands_loc = bands_loc if bands_loc is not None else torch.zeros((len(bands)))
        bands_scale = bands_scale if bands_scale is not None else torch.ones((len(bands)))
        idx_loc = idx_loc if idx_loc is not None else torch.zeros((5))
        idx_scale = idx_scale if idx_scale is not None else torch.ones((5))
                
        self.bands_loc = bands_loc.float().to(device)
        self.bands_scale = bands_scale.float().to(device)
        self.idx_loc = idx_loc.float().to(device)
        self.idx_scale = idx_scale.float().to(device)
        self.apply_norm = apply_norm
        
        # Calculate normalization factors
        self.s2norm_factor_d = (self.rsr * self.solar).sum(axis=2)
        self.s2norm_factor_n = self.rsr * self.solar
        
        # Handle downsampling if needed
        if self.R_down > 1:
            self.s2norm_factor_n = self.s2norm_factor_n[:,:,:-1].reshape(
                1, len(bands), -1, R_down).mean(3) * self.R_down

    def change_device(self, device):
        """Change the device for all tensors in the simulator."""
        self.device = device
        self.rsr = self.rsr.to(device)
        self.bands_loc = self.bands_loc.to(device)
        self.bands_scale = self.bands_scale.to(device)
        self.idx_loc = self.idx_loc.to(device)
        self.idx_scale = self.idx_scale.to(device)
        self.s2norm_factor_d = self.s2norm_factor_d.to(device)
        self.s2norm_factor_n = self.s2norm_factor_n.to(device)
        self.solar = self.solar.to(device)

    def __call__(self, prosail_output: torch.Tensor, apply_norm=None):
        """Forward pass alias."""
        return self.forward(prosail_output, apply_norm=apply_norm)
    
    def apply_s2_sensor(self, prosail_output: torch.Tensor) -> torch.Tensor:
        """
        Apply sensor spectral response to PROSAIL output.
        
        Args:
            prosail_output (torch.Tensor): PROSAIL reflectance output
            
        Returns:
            torch.Tensor: Sensor-specific band reflectances
        """
        x = prosail_output
        if len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        simu = (self.s2norm_factor_n * x).sum(axis=2) / self.s2norm_factor_d 
        return simu
    
    def normalize(self, s2_r, bands_dim=1):
        """Normalize reflectance values using stored parameters."""
        return standardize(s2_r, self.bands_loc, self.bands_scale, dim=bands_dim)
        
    def unnormalize(self, s2_r, bands_dim=1):
        """Reverse normalization of reflectance values."""
        return unstandardize(s2_r, self.bands_loc, self.bands_scale, dim=bands_dim)

    def forward(self, prosail_output: torch.Tensor, apply_norm=None) -> torch.Tensor:
        """
        Forward pass applying sensor simulation and optional normalization.
        
        Args:
            prosail_output (torch.Tensor): PROSAIL model output
            apply_norm (bool, optional): Override default normalization setting
            
        Returns:
            torch.Tensor: Processed reflectance values
        """
        simu = self.apply_s2_sensor(prosail_output)
        if apply_norm is None:
            apply_norm = self.apply_norm
        if apply_norm:
            simu = self.normalize(simu)
        return simu  # type: ignore
    
    def index_loss(self, s2_r, s2_rec, lossfn=None,
                   normalize_idx=True, s2_r_bands_dim=1, rec_bands_dim=1):
        """
        Calculate the loss between spectral indices of the target and reconstructed reflectances.
        Uses hyperspectral indices for EnMAP data.
        
        Args:
            s2_r: Target reflectance tensor
            s2_rec: Reconstructed reflectance tensor
            lossfn: Loss function to apply (None for MSE, can be gaussian_nll_loss)
            normalize_idx: Whether to normalize the indices
            s2_r_bands_dim: Dimension for bands in s2_r
            rec_bands_dim: Dimension for bands in s2_rec
            
        Returns:
            Loss value
        """
        import torch.nn.functional as F
        from utils.utils import gaussian_nll_loss
        
        # For EnMAP, we don't need to unnormalize since we use regional spectral averages
        # Just ensure the dimensions are correct
        
        # Handle dimension order for s2_r
        if s2_r_bands_dim != 1 and len(s2_r.shape) > 1:
            s2_r = s2_r.transpose(s2_r_bands_dim, 1)
            
        # Handle dimension order for s2_rec
        if rec_bands_dim != 1:
            if len(s2_rec.shape) == 3:  # [batch, channels, samples]
                s2_rec = s2_rec.transpose(rec_bands_dim, 1)
            elif len(s2_rec.shape) == 2:  # [batch, channels]
                s2_rec = s2_rec.transpose(0, 1) if rec_bands_dim == 0 else s2_rec
        
        # Calculate indices for target
        indices_tgt = self.calculate_hyperspectral_indices(s2_r)
        
        # Check if we're using gaussian_nll_loss
        using_gnll = lossfn is gaussian_nll_loss
        
        # Handle multi-sample case
        if len(s2_rec.shape) == 3:  # [batch, channels, samples]
            # Calculate indices for each sample
            indices_rec_samples = []
            for i in range(s2_rec.shape[2]):
                sample_indices = self.calculate_hyperspectral_indices(s2_rec[:, :, i])
                indices_rec_samples.append(sample_indices)
            
            # For gaussian_nll_loss, keep samples separate
            if using_gnll:
                indices_rec = torch.stack(indices_rec_samples, dim=2)  # [batch, n_indices, n_samples]
            else:
                # For other losses, average across samples
                indices_rec = torch.stack(indices_rec_samples, dim=0).mean(dim=0)
        else:
            # Calculate hyperspectral indices for single sample
            indices_rec = self.calculate_hyperspectral_indices(s2_rec)
        
        # Normalize indices if requested
        if normalize_idx and indices_tgt.shape[1] > 0:
            # Calculate mean and std across batch dimension for normalization
            mean = indices_tgt.mean(dim=0, keepdim=True)
            std = indices_tgt.std(dim=0, keepdim=True) + 1e-6  # Avoid division by zero
            
            # Only normalize for MSE loss
            if not using_gnll or len(s2_rec.shape) != 3:
                indices_tgt = (indices_tgt - mean) / std
                indices_rec = (indices_rec - mean) / std
        
        # Calculate loss
        if lossfn is None:
            # Default to MSE
            loss = F.mse_loss(indices_tgt, indices_rec)
        elif using_gnll and len(s2_rec.shape) == 3:
            # Gaussian NLL loss with samples
            loss = gaussian_nll_loss(indices_tgt, indices_rec)
        else:
            # Other custom loss function
            loss = lossfn(indices_tgt, indices_rec)
        
        return loss


class EnMapSensorSimulator(SensorSimulator):
    """Simulates the reflectances of the EnMAP hyperspectral sensor from a full spectrum.
    
    The EnMAP sensor has 224 spectral bands ranging from 420 to 2450 nm.
    This class handles the specifics of the EnMAP sensor, including its
    spectral response functions and normalization.
    """
    def __init__(self,
                 rsr_file: str = "enmap.rsr",
                 rsr_dir: str = "data",
                 prospect_range: Tuple[int, int] = (400, 2500),
                 device='cpu',
                 bands_loc=None,
                 bands_scale=None,
                 idx_loc=None,
                 idx_scale=None,
                 apply_norm=True,
                 R_down=1):
        
        # Set up path
        if rsr_dir is not None:
            rsr_path = Path(rsr_dir) / rsr_file
        else:
            rsr_path = rsr_file
            
        # Load RSR data directly
        rsr_data = np.loadtxt(str(rsr_path), unpack=True)
        
        # Get number of bands (columns 2 onwards in the RSR file)
        n_bands = rsr_data.shape[0] - 2
        all_bands = list(range(n_bands))
        
        # Initialize variables directly instead of using parent constructor
        self.R_down = R_down
        self.bands = all_bands
        self.device = device
        self.prospect_range = prospect_range
        
        # Convert RSR data to tensor
        self.rsr = torch.from_numpy(rsr_data).to(device)
        self.nb_bands = n_bands
        self.rsr_range = (int(self.rsr[0, 0].item() * 1000),
                          int(self.rsr[0, -1].item() * 1000))
        self.nb_lambdas = prospect_range[1] - prospect_range[0] + 1
        
        # Create tensor for PROSAIL wavelengths
        self.rsr_prospect = torch.zeros([self.rsr.shape[0], self.nb_lambdas]).to(device)
        self.rsr_prospect[0, :] = torch.linspace(prospect_range[0],
                                                prospect_range[1],
                                                self.nb_lambdas).to(device)
        
        # Copy RSR data to appropriate range
        offset = self.prospect_range[0] - self.rsr_range[0]
        if offset >= 0:
            length = min(self.rsr_range[1] - self.rsr_range[0] + 1 - offset, 
                         self.prospect_range[1] - self.prospect_range[0] + 1)
            self.rsr_prospect[1:, :length] = self.rsr[1:, offset:offset+length]
            
        # Extract solar spectrum and RSR
        self.solar = self.rsr_prospect[1, :].unsqueeze(0)
        self.rsr = self.rsr_prospect[2:, :].unsqueeze(0)
        
        # Normalization parameters
        bands_loc = bands_loc if bands_loc is not None else torch.zeros((len(all_bands)))
        bands_scale = bands_scale if bands_scale is not None else torch.ones((len(all_bands)))
        idx_loc = idx_loc if idx_loc is not None else torch.zeros((8))  # For 8 indices
        idx_scale = idx_scale if idx_scale is not None else torch.ones((8))
                
        self.bands_loc = bands_loc.float().to(device)
        self.bands_scale = bands_scale.float().to(device)
        self.idx_loc = idx_loc.float().to(device)
        self.idx_scale = idx_scale.float().to(device)
        self.apply_norm = apply_norm
        
        # Calculate normalization factors
        self.s2norm_factor_d = (self.rsr * self.solar).sum(axis=2)
        self.s2norm_factor_n = self.rsr * self.solar
        
        # Handle downsampling if needed
        if self.R_down > 1:
            self.s2norm_factor_n = self.s2norm_factor_n[:,:,:-1].reshape(1, len(all_bands), -1, R_down).mean(3) * self.R_down
            
        # Store the actual wavelengths for reference
        self.bands_wl = self.rsr_prospect[0, :]
        
    def get_refl(self, prosail_output: torch.Tensor, apply_norm=None) -> torch.Tensor:
        """Get EnMAP reflectances from PROSAIL output"""
        return self.forward(prosail_output, apply_norm=apply_norm)
    
    def calculate_hyperspectral_indices(self, reflectance: torch.Tensor) -> torch.Tensor:
        """
        Calculate hyperspectral indices from reflectance data using the spectral_indices module.
        
        Args:
            reflectance: Tensor of shape [batch_size, num_bands] or [num_bands]
        
        Returns:
            Tensor of shape [batch_size, num_indices] containing the calculated indices
        """
        from prosailvae.spectral_indices import get_enmap_spectral_idx
        
        # Use the standard function from spectral_indices.py
        # Make sure we're passing the correct wavelength values matching the current bands
        wavelengths = self.rsr_prospect[0, :].detach()
        
        # The wavelengths need to be aligned with the band reflectances
        # Create a proper wavelength array for the bands
        wavelengths_bands = torch.linspace(420, 2450, reflectance.shape[1], device=self.device)
        
        return get_enmap_spectral_idx(reflectance, wavelengths_bands)
    
    def index_loss(self, s2_r, s2_rec, lossfn=None,
                   normalize_idx=True, s2_r_bands_dim=1, rec_bands_dim=1):
        """
        Calculate the loss between spectral indices of the target and reconstructed reflectances.
        Uses hyperspectral indices for EnMAP data.
        
        Args:
            s2_r: Target reflectance tensor
            s2_rec: Reconstructed reflectance tensor
            lossfn: Loss function to apply (None for MSE, can be gaussian_nll_loss)
            normalize_idx: Whether to normalize the indices
            s2_r_bands_dim: Dimension for bands in s2_r
            rec_bands_dim: Dimension for bands in s2_rec
            
        Returns:
            Loss value
        """
        import torch.nn.functional as F
        from utils.utils import gaussian_nll_loss
        
        # For EnMAP, we don't need to unnormalize since we use regional spectral averages
        # Just ensure the dimensions are correct
        
        # Handle dimension order for s2_r
        if s2_r_bands_dim != 1 and len(s2_r.shape) > 1:
            s2_r = s2_r.transpose(s2_r_bands_dim, 1)
            
        # Handle dimension order for s2_rec
        if rec_bands_dim != 1:
            if len(s2_rec.shape) == 3:  # [batch, channels, samples]
                s2_rec = s2_rec.transpose(rec_bands_dim, 1)
            elif len(s2_rec.shape) == 2:  # [batch, channels]
                s2_rec = s2_rec.transpose(0, 1) if rec_bands_dim == 0 else s2_rec
        
        # Calculate indices for target
        indices_tgt = self.calculate_hyperspectral_indices(s2_r)
        
        # Check if we're using gaussian_nll_loss
        using_gnll = lossfn is gaussian_nll_loss
        
        # Handle multi-sample case
        if len(s2_rec.shape) == 3:  # [batch, channels, samples]
            # Calculate indices for each sample
            indices_rec_samples = []
            for i in range(s2_rec.shape[2]):
                sample_indices = self.calculate_hyperspectral_indices(s2_rec[:, :, i])
                indices_rec_samples.append(sample_indices)
            
            # For gaussian_nll_loss, keep samples separate
            if using_gnll:
                indices_rec = torch.stack(indices_rec_samples, dim=2)  # [batch, n_indices, n_samples]
            else:
                # For other losses, average across samples
                indices_rec = torch.stack(indices_rec_samples, dim=0).mean(dim=0)
        else:
            # Calculate hyperspectral indices for single sample
            indices_rec = self.calculate_hyperspectral_indices(s2_rec)
        
        # Normalize indices if requested
        if normalize_idx and indices_tgt.shape[1] > 0:
            # Calculate mean and std across batch dimension for normalization
            mean = indices_tgt.mean(dim=0, keepdim=True)
            std = indices_tgt.std(dim=0, keepdim=True) + 1e-6  # Avoid division by zero
            
            # Only normalize for MSE loss
            if not using_gnll or len(s2_rec.shape) != 3:
                indices_tgt = (indices_tgt - mean) / std
                indices_rec = (indices_rec - mean) / std
        
        # Calculate loss
        if lossfn is None:
            # Default to MSE
            loss = F.mse_loss(indices_tgt, indices_rec)
        elif using_gnll and len(s2_rec.shape) == 3:
            # Gaussian NLL loss with samples
            loss = gaussian_nll_loss(indices_tgt, indices_rec)
        else:
            # Other custom loss function
            loss = lossfn(indices_tgt, indices_rec)
        
        return loss


class ProsailSimulator():
    """
    Wrapper class for running PROSAIL radiative transfer model simulations.
    
    This class provides a convenient interface for:
    - Initializing PROSAIL spectra
    - Running PROSAIL simulations with different parameter sets
    - Managing device placement of model components
    - Supporting both PROSPECT-5 and PROSPECT-D versions
    
    Attributes:
        factor (str): Type of reflectance factor to compute
        typelidf (int): Leaf inclination distribution function type
        device (str): Computation device
        R_down (int): Downsampling factor
        prospect_version (str): PROSPECT model version to use
    """
    
    def __init__(self, factor: str = "SDR", 
                 typelidf: int = 2, 
                 device='cpu', 
                 R_down:int=1, 
                 prospect_version="5"):
        """
        Initialize ProsailSimulator.
        
        Args:
            factor (str): Type of reflectance factor ("SDR", "BHR", or "DHR")
            typelidf (int): Leaf inclination distribution function type
            device (str): Device for computations ('cpu' or 'cuda')
            R_down (int): Downsampling factor
            prospect_version (str): PROSPECT version ("5" or "D")
        """
        super().__init__()
        self.factor = factor
        self.typelidf = typelidf
        self.device = device
        self.R_down = R_down
        self.prospect_version = prospect_version
        
        # Initialize PROSAIL spectra
        [self.soil_spectrum1,
         self.soil_spectrum2,
         self.nr,
         self.kab,
         self.kcar,
         self.kbrown,
         self.kw,
         self.km,
         self.kant,
         self.kprot,
         self.kcbc,
         self.lambdas] = init_prosail_spectra(R_down=self.R_down, 
                                             device=self.device, 
                                             prospect_version=self.prospect_version)
        
    def __call__(self, params):
        """Forward pass alias."""
        return self.forward(params)

    def change_device(self, device):
        """
        Move all model components to specified device.
        
        Args:
            device (str): Target device ('cpu' or 'cuda')
        """
        self.device = device
        self.soil_spectrum1 = self.soil_spectrum1.to(device)
        self.soil_spectrum2 = self.soil_spectrum2.to(device)
        self.nr = self.nr.to(device)
        self.kab = self.kab.to(device)
        self.kcar = self.kcar.to(device)
        self.kbrown = self.kbrown.to(device)
        self.kw = self.kw.to(device)
        self.km = self.km.to(device)
        self.kant = self.kant.to(device)
        self.kprot = self.kprot.to(device)
        self.kcbc = self.kcbc.to(device)
        self.lambdas = self.lambdas.to(device)

    def forward(self, params: torch.Tensor):
        """
        Run PROSAIL simulation with given parameters.
        
        Args:
            params (torch.Tensor): Parameter tensor [N, cab, car, cbrown, cw, cm,
                                                   lai, lidfa, hspot, rsoil, psoil,
                                                   tts, tto, psi]
                                 Shape: [..., 14] for single or batch input
        
        Returns:
            torch.Tensor: Simulated reflectance spectrum
        """
        assert params.shape[-1] == 14, f"{params.shape[-1]}"
        if len(params.shape) == 1:
            params = params.unsqueeze(0)
            
        prosail_refl = prosail.run_prosail(
            N=params[...,0].unsqueeze(-1),
            cab=params[...,1].unsqueeze(-1), 
            car=params[...,2].unsqueeze(-1), 
            cbrown=params[...,3].unsqueeze(-1), 
            cw=params[...,4].unsqueeze(-1), 
            cm=params[...,5].unsqueeze(-1), 
            lai=params[...,6].unsqueeze(-1), 
            lidfa=params[...,7].unsqueeze(-1), 
            hspot=params[...,8].unsqueeze(-1), 
            rsoil=params[...,9].unsqueeze(-1), 
            psoil=params[...,10].unsqueeze(-1), 
            tts=params[...,11].unsqueeze(-1), 
            tto=params[...,12].unsqueeze(-1), 
            psi=params[...,13].unsqueeze(-1), 
            typelidf=torch.as_tensor(self.typelidf),
            factor=self.factor,
            device=self.device,
            soil_spectrum1=self.soil_spectrum1,
            soil_spectrum2=self.soil_spectrum2,
            nr=self.nr,
            kab=self.kab,
            kcar=self.kcar,
            kbrown=self.kbrown,
            kw=self.kw,
            km=self.km,
            kant=self.kant,
            kprot=self.kprot,
            kcbc=self.kcbc,
            lambdas=self.lambdas,
            R_down=1, 
            init_spectra=False,
            prospect_version=self.prospect_version
        ).float()
        
        return prosail_refl
    
    def run_prospect(self, params: torch.Tensor):
        """
        Run only the PROSPECT portion of the simulation.
        
        Args:
            params (torch.Tensor): First 6 PROSPECT parameters
                                 [N, cab, car, cbrown, cw, cm]
                                 Shape: [..., 6] for single or batch input
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Wavelengths
                - Leaf reflectance
                - Leaf transmittance
        """
        if len(params.shape) == 1:
            params = params.unsqueeze(0)
            
        wv, refl, trans = prosail.run_prospect(
            N=params[...,0].unsqueeze(-1),
            cab=params[...,1].unsqueeze(-1), 
            car=params[...,2].unsqueeze(-1), 
            cbrown=params[...,3].unsqueeze(-1), 
            cw=params[...,4].unsqueeze(-1), 
            cm=params[...,5].unsqueeze(-1), 
            device=self.device,
            nr=self.nr,
            kab=self.kab,
            kcar=self.kcar,
            kbrown=self.kbrown,
            kw=self.kw,
            km=self.km,
            kant=self.kant,
            kprot=self.kprot,
            kcbc=self.kcbc,
            lambdas=self.lambdas,
            R_down=1, 
            init_spectra=False,
            prospect_version=self.prospect_version
        )
        
        return wv.float(), refl.float(), trans.float()







