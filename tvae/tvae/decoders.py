#!/usr/bin/env python3
"""
Decoders Module
===============

This module implements decoder architectures for the transformer-VAE model, which maps
latent space representations back to spectral reflectance data using physics-based
radiative transfer modeling.
"""

import torch
import torch.nn as nn

from utils.utils import select_rec_loss_fn


class Decoder(nn.Module):
    """
    This class defines the interface that all decoder implementations must follow.
    Decoders are responsible for mapping latent space representations back to 
    spectral reflectance data through physics-based simulation.
    
    Methods
    -------
    decode()
        Abstract method to decode latent vectors into spectral reflectance.
        Must be implemented by subclasses.
    loss()
        Abstract method to compute reconstruction loss.
        Must be implemented by subclasses.
    """
    def decode(self):
        raise NotImplementedError()

    def loss(self):
        raise NotImplementedError()


class ProsailSimulatorDecoder(Decoder):
    """
    Physics-based decoder using PROSAIL radiative transfer model.
    
    This decoder implements the physically-based decoding process using the PROSAIL
    radiative transfer model and sensor-specific simulators. It maps latent vectors 
    to spectral reflectance by:
    1. Combining latent vectors with viewing angles
    2. Running PROSAIL simulation on the combined input
    3. Applying sensor-specific simulation to the PROSAIL output
    
    Parameters
    ----------
    prosailsimulator : ProsailSimulator
        Instance of PROSAIL simulator for radiative transfer modeling
    ssimulator : SensorSimulator
        Sensor-specific simulator (e.g., Sentinel-2, EnMAP)
    device : str, optional
        Device to place the model on, by default "cpu"
    loss_type : str, optional
        Type of reconstruction loss to use, by default "diag_nll"
        
    Attributes
    ----------
    device : str
        Current device the model is on
    prosailsimulator : ProsailSimulator
        PROSAIL radiative transfer model simulator
    ssimulator : SensorSimulator
        Sensor-specific simulator
    loss_type : str
        Type of reconstruction loss being used
    nbands : int
        Number of spectral bands in sensor
    rec_loss_fn : callable
        Selected reconstruction loss function
    """

    def __init__(
        self, prosailsimulator, ssimulator, device="cpu", loss_type="diag_nll"
    ):
        """
        Initialize the ProsailSimulatorDecoder.
        
        Parameters
        ----------
        prosailsimulator : ProsailSimulator
            Instance of PROSAIL simulator for radiative transfer modeling
        ssimulator : SensorSimulator
            Sensor-specific simulator (e.g., Sentinel-2, EnMAP)
        device : str, optional
            Device to place the model on, by default "cpu"
        loss_type : str, optional
            Type of reconstruction loss to use, by default "diag_nll"
        """
        super().__init__()
        self.device = device
        self.prosailsimulator = prosailsimulator
        self.ssimulator = ssimulator
        self.loss_type = loss_type
        self.nbands = len(ssimulator.bands)
        self.rec_loss_fn = select_rec_loss_fn(self.loss_type)

    def change_device(self, device):
        """
        Move the model and its components to specified device.
        
        Parameters
        ----------
        device : str
            Target device ('cpu' or 'cuda')
        """
        self.device = device
        self.ssimulator.change_device(device)
        self.prosailsimulator.change_device(device)
        pass

    def decode(self, z, angles, apply_norm=None):
        """
        Decode latent vectors into spectral reflectance using PROSAIL simulation.
        
        This method:
        1. Combines latent vectors with viewing angles
        2. Runs PROSAIL simulation on the combined input
        3. Applies sensor-specific simulation to get final reflectance
        
        Parameters
        ----------
        z : torch.Tensor
            Latent vectors of shape [batch_size, latent_dim, n_samples]
        angles : torch.Tensor
            Viewing angles of shape [batch_size, 3]
        apply_norm : bool, optional
            Whether to apply normalization to output, by default None
            
        Returns
        -------
        torch.Tensor
            Reconstructed spectral reflectance of shape 
            [batch_size, n_bands, n_samples]
        """
        n_samples = z.size(2)
        batch_size = z.size(0)

        sim_input = (
            torch.concat((z, angles.unsqueeze(2).repeat(1, 1, n_samples)), axis=1)
            .transpose(1, 2)
            .reshape(n_samples * batch_size, -1)
        )
        prosail_output = self.prosailsimulator(sim_input)
        rec = (
            self.ssimulator(prosail_output, apply_norm=apply_norm)
            .reshape(batch_size, n_samples, -1)
            .transpose(1, 2)
        )
        return rec

    def loss(self, tgt, rec):
        """
        Compute reconstruction loss between target and reconstructed reflectance.
        
        Parameters
        ----------
        tgt : torch.Tensor
            Target spectral reflectance
        rec : torch.Tensor
            Reconstructed spectral reflectance
            
        Returns
        -------
        torch.Tensor
            Computed reconstruction loss
            
        Notes
        -----
        If self.ssimulator.apply_norm is True, the target will be normalized
        before computing the loss.
        """
        if self.ssimulator.apply_norm:
            tgt = self.ssimulator.normalize(tgt)
        rec_loss = self.rec_loss_fn(tgt, rec)
        return rec_loss
