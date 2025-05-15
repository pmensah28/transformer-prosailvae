#!/usr/bin/env python3
"""
This module implements the LinearVarSpace class, which is responsible for
mapping latent space representations to physical parameters using a linear transformation.
The LinearVarSpace class provides methods for converting between latent space and physical
parameter space, as well as for computing probability density functions (PDFs) and
cumulative distribution functions (CDFs) for the physical parameters.
It also includes methods for computing point estimates (mean, median, mode) and
expectations of the physical parameters based on the latent space representations.
The LinearVarSpace class is designed to work with the PROSAIL model, which is a
radiative transfer model used for simulating the reflectance of vegetation and soil
canopies.

Credit: yoel zerah
"""
import numpy as np
import torch
import torch.nn as nn

from utils.image_utils import batchify_batch_latent, unbatchify
from utils.TruncatedNormal import TruncatedNormal
from utils.utils import torch_select_unsqueeze

from prosailvae.dist_utils import cdfs2quantiles, convolve_pdfs, pdfs2cdfs
from prosailvae.prosail_var_dists import (
    get_prosail_var_bounds,
    get_prosailparams_pdf_span,
    get_z2prosailparams_bound,
    get_z2prosailparams_mat,
    get_z2prosailparams_offset,
)


class SimVarSpace(nn.Module):
    """
    Abstract base class for simulation variable space transformations.
    
    This class defines the interface for transforming between latent space
    and simulation parameter space.
    """
    
    def lat2sim(self):
        """Transform from latent space to simulation parameter space."""
        raise NotImplementedError


class LinearVarSpace(SimVarSpace):
    """
    Linear transformation between latent space and simulation parameters.
    
    This class implements a linear mapping between the latent space of a VAE
    and the physical parameter space of the PROSAIL model. It handles:
    - Linear transformations between spaces
    - PDF/CDF computations
    - Statistical estimates (mode, median, expectation)
    - Distribution parameter conversions
    
    Attributes:
        device (str): Computation device ('cpu' or 'cuda')
        eps (float): Small value for numerical stability
        latent_dim (int): Dimension of latent space
        var_bounds: Parameter bounds from PROSAIL model
        z2sim_mat (torch.Tensor): Linear transformation matrix
        z2sim_offset (torch.Tensor): Translation vector
        sim_pdf_support_span (torch.Tensor): PDF support range
        inv_z2sim_mat (torch.Tensor): Inverse transformation matrix
    """

    def __init__(
        self,
        latent_dim=6,
        device="cpu",
        var_bounds_type="legacy",
    ):
        """
        Initialize LinearVarSpace.
        
        Args:
            latent_dim (int): Dimension of latent space
            device (str): Computation device ('cpu' or 'cuda')
            var_bounds_type (str): Type of variable bounds ("legacy" or "new")
        """
        super().__init__()
        self.device = device
        self.eps = 1e-3
        self.latent_dim = latent_dim
        
        # Get parameter bounds and transformations
        self.var_bounds = get_prosail_var_bounds(var_bounds_type)
        self.z2sim_mat = get_z2prosailparams_mat(self.var_bounds).to(device)
        self.z2sim_offset = get_z2prosailparams_offset(self.var_bounds).to(device)
        self.sim_pdf_support_span = get_prosailparams_pdf_span(self.var_bounds).to(
            device
        )
        
        # Compute inverse transformation matrix
        self.inv_z2sim_mat = torch.from_numpy(
            np.linalg.inv(self.z2sim_mat.detach().cpu().numpy())
        ).to(self.device)

    def change_device(self, device):
        """
        Move all tensors to specified device.
        
        Args:
            device (str): Target device ('cpu' or 'cuda')
        """
        self.device = device
        self.z2sim_mat = self.z2sim_mat.to(device)
        self.z2sim_offset = self.z2sim_offset.to(device)
        self.sim_pdf_support_span = self.sim_pdf_support_span.to(device)
        self.inv_z2sim_mat = self.inv_z2sim_mat.to(device)

    def get_distribution_from_lat_params(
        self,
        lat_params,
        distribution_type="tn",
        dist_idx=1,
    ):
        """
        Convert latent parameters to a distribution in simulation space.
        
        Args:
            lat_params: Latent space parameters
            distribution_type (str): Distribution type ("tn" for truncated normal)
            dist_idx (int): Index for distribution parameters
            
        Returns:
            TruncatedNormal: Distribution in simulation space
            
        Raises:
            NotImplementedError: For unsupported distribution types
        """
        if distribution_type == "tn":
            # Extract mean and convert to simulation space
            lat_mu = lat_params.select(dist_idx, 0)
            sim_mu = unbatchify(
                self.z2sim(batchify_batch_latent(lat_mu).unsqueeze(2))
            ).squeeze(0)
            
            # Convert variance to simulation space
            lat_sigma = lat_params.select(dist_idx, 1)
            lat_sigma2 = lat_sigma.pow(2)
            sim_sigma2 = (
                torch_select_unsqueeze(
                    torch.diag(self.z2sim_mat).pow(2), 1, len(lat_sigma2.size())
                )
                * lat_sigma2
            )
            sim_sigma = sim_sigma2.sqrt()
            
            # Get bounds for truncated normal
            high = torch_select_unsqueeze(
                get_z2prosailparams_bound("high"), 1, len(lat_sigma2.size())
            ).to(sim_mu.device)
            low = torch_select_unsqueeze(
                get_z2prosailparams_bound("low"), 1, len(lat_sigma2.size())
            ).to(sim_mu.device)
            
            distribution = TruncatedNormal(
                loc=sim_mu, scale=sim_sigma, low=low, high=high
            )
        else:
            raise NotImplementedError
        return distribution

    def z2sim(self, z):
        """
        Transform from latent space to simulation space.
        
        Args:
            z (torch.Tensor): Points in latent space
            
        Returns:
            torch.Tensor: Transformed points in simulation space
        """
        sim = torch.matmul(self.z2sim_mat, z) + self.z2sim_offset
        return sim

    def sim2z(self, sim):
        """
        Transform from simulation space to latent space.
        
        Args:
            sim (torch.Tensor): Points in simulation space
            
        Returns:
            torch.Tensor: Transformed points in latent space
        """
        if len(sim.size()) == 2:
            sim = sim.unsqueeze(2)
        z = torch.matmul(self.inv_z2sim_mat, sim - self.z2sim_offset)
        return z

    def sim_pdf(self, pdfs, supports, n_pdf_sample_points=3001):
        """
        Compute PDFs in simulation space.
        
        Applies the linear transformation to PDFs from latent space
        to get corresponding PDFs in simulation space.
        
        Args:
            pdfs (torch.Tensor): Input PDFs in latent space
            supports (torch.Tensor): Support points for input PDFs
            n_pdf_sample_points (int): Number of points to sample
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (transformed PDFs, support points)
        """
        sim_pdfs = torch.zeros(
            (pdfs.size(0), self.z2sim_mat.size(0), n_pdf_sample_points)
        ).to(self.device)
        sim_supports = torch.zeros(
            (pdfs.size(0), self.z2sim_mat.size(0), n_pdf_sample_points)
        ).to(self.device)
        
        # Transform each dimension
        for i in range(self.latent_dim):
            transfer_mat_line = self.z2sim_mat[i]
            sim_pdf, sim_support = convolve_pdfs(
                pdfs,
                supports,
                transfer_mat_line,
                n_pdf_sample_points=n_pdf_sample_points,
                support_max=self.sim_pdf_support_span[i],
            )
            sim_support = sim_support + self.z2sim_offset[i].item()
            sim_pdfs[:, i, :] = sim_pdf
            sim_supports[:, i, :] = sim_support
            
        return sim_pdfs, sim_supports

    def sim_mode(self, pdfs, supports, n_pdf_sample_points=3001):
        """
        Compute modes of distributions in simulation space.
        
        Args:
            pdfs (torch.Tensor): Input PDFs
            supports (torch.Tensor): Support points for PDFs
            n_pdf_sample_points (int): Number of points to sample
            
        Returns:
            torch.Tensor: Modes of transformed distributions
        """
        batch_size = pdfs.size(0)
        latent_size = pdfs.size(1)
        sim_pdfs, sim_supports = self.sim_pdf(
            pdfs, supports, n_pdf_sample_points=n_pdf_sample_points
        )
        
        # Find maximum of each PDF
        max_index = (
            sim_pdfs.view(batch_size * latent_size, -1).argmax(dim=1).view(-1, 1)
        )
        sim_mode = torch.gather(
            sim_supports.view(batch_size * latent_size, -1), dim=1, index=max_index
        ).view(batch_size, latent_size, -1)
        
        return sim_mode

    def sim_quantiles(self, pdfs, supports, alpha=[0.5], n_pdf_sample_points=3001):
        """
        Compute quantiles of distributions in simulation space.
        
        Args:
            pdfs (torch.Tensor): Input PDFs
            supports (torch.Tensor): Support points for PDFs
            alpha (List[float]): Quantile levels (e.g., [0.5] for median)
            n_pdf_sample_points (int): Number of points to sample
            
        Returns:
            torch.Tensor: Quantiles at specified levels
        """
        sim_pdfs, sim_supports = self.sim_pdf(
            pdfs, supports, n_pdf_sample_points=n_pdf_sample_points
        )
        sim_cdfs = pdfs2cdfs(sim_pdfs)
        quantiles = torch.zeros((pdfs.size(0), pdfs.size(1), len(alpha))).to(
            pdfs.device
        )
        
        # Compute quantiles for each batch
        for i in range(pdfs.size(0)):
            quantiles[i, :, :] = cdfs2quantiles(
                sim_cdfs[i, :, :].cpu(), sim_supports[i, :, :].cpu(), alpha=alpha
            ).to(pdfs.device)
            
        return quantiles

    def sim_median(self, pdfs, supports, n_pdf_sample_points=3001):
        """
        Compute medians of distributions in simulation space.
        
        Args:
            pdfs (torch.Tensor): Input PDFs
            supports (torch.Tensor): Support points for PDFs
            n_pdf_sample_points (int): Number of points to sample
            
        Returns:
            torch.Tensor: Medians of transformed distributions
        """
        sim_median = self.sim_quantiles(
            pdfs, supports, n_pdf_sample_points=n_pdf_sample_points, alpha=[0.5]
        ).view(1, -1, 1)
        return sim_median

    def sim_expectation(self, pdfs, supports, n_pdf_sample_points=3001):
        """
        Compute expectations of distributions in simulation space.
        
        Args:
            pdfs (torch.Tensor): Input PDFs
            supports (torch.Tensor): Support points for PDFs
            n_pdf_sample_points (int): Number of points to sample
            
        Returns:
            torch.Tensor: Expected values of transformed distributions
        """
        sim_pdfs, sim_supports = self.sim_pdf(
            pdfs, supports, n_pdf_sample_points=n_pdf_sample_points
        )
        sampling = sim_supports[:, 1] - sim_supports[:, 0]
        sim_expected = (sim_pdfs * sim_supports) * sampling.view(-1, 1)
        return sim_expected.sum(1).view(1, -1, 1)

    def sim_all_point_estimates(self, pdfs, supports, n_pdf_sample_points=3001):
        """
        Compute all point estimates (mode, median, mean) in simulation space.
        
        Args:
            pdfs (torch.Tensor): Input PDFs
            supports (torch.Tensor): Support points for PDFs
            n_pdf_sample_points (int): Number of points to sample
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                (modes, medians, expected values)
        """
        sim_pdfs, sim_supports = self.sim_pdf(
            pdfs, supports, n_pdf_sample_points=n_pdf_sample_points
        )
        
        # Calculate expected value
        sampling = sim_supports[:, 1] - sim_supports[:, 0]
        sim_expected = (sim_pdfs * sim_supports) * sampling.view(-1, 1)
        sim_expected = sim_expected.sum(1).view(-1, 1)
        
        # Calculate modes
        sim_modes = torch.gather(
            sim_supports, dim=1, index=sim_pdfs.argmax(dim=1).view(-1, 1)
        )
        
        # Calculate medians
        sim_cdfs = pdfs2cdfs(sim_pdfs)
        sim_median = cdfs2quantiles(sim_cdfs, sim_supports, alpha=[0.5])[0]

        return sim_modes, sim_median, sim_expected
