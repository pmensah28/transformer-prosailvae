#!/usr/bin/env python3
"""
Encoders Module
================

This module implements various encoder architectures including the RNN, CNN, and Transformer
which is designed to retrieve biophysical parameters from Sentinel-2 satellite imagery.
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from prosailvae.spectral_indices import get_spectral_idx
from utils.image_utils import batchify_batch_latent, check_is_patch
from utils.utils import IOStandardizeCoeffs, deg2rad, standardize


@dataclass
class EncoderConfig:
    """
    This dataclass defines all the parameters needed to initialize and configure
    various encoder architectures (RNN, CNN, Transformer).
    """
    io_coeffs: IOStandardizeCoeffs
    encoder_type: str = "rnn"
    input_size: int = 16
    output_size: int = 12
    device: str = "cpu"
    bands: torch.Tensor | None = torch.arange(10)
    last_activation: nn.Module | None = None
    n_latent_params: int = 2
    layer_sizes: list[int] | None = field(default_factory=lambda: [128])
    kernel_sizes: list[int] = field(default_factory=lambda: [3])
    padding: str = "valid"

    # For MLP/RNN-based encoders
    first_layer_kernel: int = 3
    first_layer_size: int = 128
    block_layer_sizes: list[int] = field(default_factory=lambda: [128, 128])
    block_layer_depths: list[int] = field(default_factory=lambda: [2, 2])
    block_kernel_sizes: list[int] = field(default_factory=lambda: [3, 1])
    block_n: list[int] = field(default_factory=lambda: [1, 2])
    disable_s2_r_idx: bool = False  # If True, do not use spectral indices

    # === NEW FIELDS FOR TRANSFORMER ===
    # (used only if encoder_type="transformer")
    n_transformer_layers: int = 2
    d_model: int = 64
    n_heads: int = 4
    dropout: float = 0.1


class Encoder(nn.Module):
    """
    This class defines the interface that all encoder implementations must follow.
    Encoders are responsible for mapping input spectral data and viewing angles 
    to a latent space representation.
    
    Methods
    -------
    encode(x)
        Abstract method to encode input data into latent space.
        Must be implemented by subclasses.
    """

    def encode(self):
        raise NotImplementedError


class EncoderResBlock(nn.Module):
    """
    A residual MLP encoder block with configurable depth and activation.
    
    This class implements a residual block that maintains the input dimensionality
    through a series of linear layers with optional activation functions. The block
    adds the input to the output (residual connection) to help with gradient flow.
    
    Parameters
    ----------
    hidden_layers_size : int, optional
        Size of the hidden layers, by default 128
    depth : int, optional
        Number of linear layers in the block, by default 2
    last_activation : nn.Module, optional
        Activation function to apply after the last layer, by default None
    device : str | torch.device, optional
        Device to place the model on, by default torch.device("cpu")
        
    Attributes
    ----------
    net : nn.Sequential
        Sequential container of linear layers and activations
    device : str | torch.device
        Device the model is currently on
        
    Methods
    -------
    change_device(device: str)
        Move the model to a different device
    forward(x: torch.Tensor) -> torch.Tensor
        Forward pass adding residual connection
    """

    def __init__(
        self,
        hidden_layers_size: int = 128,
        depth: int = 2,
        last_activation=None,
        device: str | torch.device = torch.device("cpu"),
    ):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.append(
                nn.Linear(
                    in_features=hidden_layers_size, out_features=hidden_layers_size
                )
            )
            if i < depth - 1:
                layers.append(nn.ReLU())
        if last_activation is not None:
            layers.append(last_activation)
        self.device = device
        self.net = nn.Sequential(*layers).to(device)

    def change_device(self, device: str):
        """
        Move the class attributes to desired device
        """
        self.device = device
        self.net = self.net.to(device)

    def forward(self, x: torch.Tensor):
        y = self.net(x)
        return y + x


class ProsailRNNEncoder(Encoder):
    """
    This encoder uses a series of residual MLP blocks to process spectral data 
    and viewing angles, mapping them to a latent space representation. It handles
    input normalization and optional spectral indices computation.
    """

    def __init__(self, config: EncoderConfig, device: str | torch.device | None = None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bands = config.bands
        if bands is None:
            bands = torch.arange(10)
        self.bands = bands.to(device)
        resnet = []
        # First Layer
        resnet.append(
            nn.Linear(
                in_features=config.input_size, out_features=config.block_layer_sizes[0]
            )
        )
        resnet.append(nn.ReLU())
        # Residual connexion blocks
        n_groups = len(config.block_n)
        for i in range(n_groups):
            for _ in range(config.block_n[i]):
                resblock = EncoderResBlock(
                    hidden_layers_size=config.block_layer_sizes[i],
                    depth=config.block_layer_depths[i],
                    last_activation=None,
                    device=device,
                )
                resnet.append(resblock)
                resnet.append(nn.ReLU())
        # Last layer
        resnet.append(
            nn.Linear(
                in_features=config.block_layer_sizes[-1],
                out_features=config.n_latent_params * config.output_size,
            )
        )
        if config.last_activation is not None:
            resnet.append(config.last_activation)
        self.output_size = config.output_size
        self.device = device
        self.net = nn.Sequential(*resnet).to(device)

        bands_loc = config.io_coeffs.bands.loc
        idx_loc = config.io_coeffs.idx.loc
        angles_loc = config.io_coeffs.angles.loc
        bands_scale = config.io_coeffs.bands.scale
        idx_scale = config.io_coeffs.idx.scale
        angles_scale = config.io_coeffs.angles.scale
        bands_loc = (
            bands_loc if bands_loc is not None else torch.zeros(config.input_size)
        )
        bands_scale = (
            bands_scale if bands_scale is not None else torch.ones(config.input_size)
        )
        idx_loc = idx_loc if idx_loc is not None else torch.zeros(5)
        idx_scale = idx_scale if idx_scale is not None else torch.ones(5)
        angles_loc = angles_loc if angles_loc is not None else torch.zeros(3)
        angles_scale = angles_scale if angles_scale is not None else torch.ones(3)
        self.bands_loc = bands_loc.float().to(device)
        self.bands_scale = bands_scale.float().to(device)
        self.idx_loc = idx_loc.float().to(device)
        self.idx_scale = idx_scale.float().to(device)
        self.angles_loc = angles_loc.float().to(device)
        self.angles_scale = angles_scale.float().to(device)
        self._spatial_encoding = False
        self.nb_enc_cropped_hw = 0
        self.disable_s2_r_idx = config.disable_s2_r_idx

    def get_spatial_encoding(self):
        return self._spatial_encoding

    def change_device(self, device):
        self.device = device
        self.bands_loc = self.bands_loc.to(device)
        self.bands_scale = self.bands_scale.to(device)
        self.idx_loc = self.idx_loc.to(device)
        self.idx_scale = self.idx_scale.to(device)
        self.angles_loc = self.angles_loc.to(device)
        self.angles_scale = self.angles_scale.to(device)
        self.net = self.net.to(device)

    def encode(self, s2_refl, angles):
        if len(s2_refl.size()) == 4:
            s2_refl = batchify_batch_latent(s2_refl)
            angles = batchify_batch_latent(angles)

        if s2_refl.size(1) == self.bands_loc.size(0):
            normed_refl = standardize(
                s2_refl, loc=self.bands_loc, scale=self.bands_scale, dim=1
            )
            if len(self.bands) < normed_refl.size(1):
                normed_refl = normed_refl[:, self.bands]
        elif len(self.bands) == self.bands_loc.size(0):
            normed_refl = standardize(
                s2_refl[:, self.bands],
                loc=self.bands_loc,
                scale=self.bands_scale,
                dim=1,
            )
        else:
            raise NotImplementedError

        normed_angles = standardize(
            torch.cos(deg2rad(angles)), self.angles_loc, self.angles_scale, dim=1
        )
        encoder_input = torch.concat((normed_refl, normed_angles), axis=1)
        if not self.disable_s2_r_idx:
            spectral_idx = get_spectral_idx(s2_refl, bands_dim=1)
            encoder_input = torch.concat((encoder_input, spectral_idx), axis=1)

        encoder_output = self.net(encoder_input)
        return encoder_output, angles

    def forward(self, s2_refl, angles):
        return self.encode(s2_refl, angles)


class EncoderCResNetBlock(Encoder):
    """
    A residual CNN block for the encoder architecture.
    
    This class implements a convolutional residual block that maintains spatial dimensions
    through a series of CNN layers with optional activation functions. The block includes
    a residual connection that adds the input to the output.
    
    Parameters
    ----------
    output_size : int, optional
        Number of output channels, by default 128
    depth : int, optional
        Number of convolutional layers in the block, by default 2
    kernel_size : int, optional
        Size of convolutional kernels, by default 3
    last_activation : nn.Module, optional
        Activation function to apply after last layer, by default None
    device : str | torch.device, optional
        Device to place the model on, by default torch.device("cpu")  
    input_size : int, optional
        Number of input channels, by default 10
    stride : int, optional
        Stride for convolutions, by default 1
    padding : str, optional
        Padding mode ('valid' or 'same'), by default 'valid'
        
    Attributes
    ----------
    net : nn.Sequential
        Sequential container of CNN layers and activations
    nb_enc_cropped_hw : int
        Number of pixels cropped from height/width due to valid padding
    device : str | torch.device
        Device the model is currently on
    """

    def __init__(
        self,
        output_size: int = 128,
        depth: int = 2,
        kernel_size: int = 3,
        last_activation=None,
        device: str | torch.device = torch.device("cpu"),
        input_size: int = 10,
        stride: int = 1,
        padding: str = "valid",
    ):
        super().__init__()
        layers = []
        input_sizes = [input_size] + [output_size for i in range(depth - 1)]
        for i in range(depth):
            layers.append(
                nn.Conv2d(
                    input_sizes[i],
                    output_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                )
            )
            if i < depth - 1:
                layers.append(nn.ReLU())
        if last_activation is not None:
            layers.append(last_activation)
        self.device = device
        self.net = nn.Sequential(*layers).to(device)
        self.nb_enc_cropped_hw = 0
        for _ in range(depth):
            self.nb_enc_cropped_hw += kernel_size // 2

    def change_device(self, device):
        self.device = device
        self.net = self.net.to(device)

    def forward(self, x):
        y = self.net(x)
        x_cropped = x
        patch_size = x.size(-1)
        if self.nb_enc_cropped_hw > 0:
            x_cropped = x[
                ...,
                self.nb_enc_cropped_hw : patch_size - self.nb_enc_cropped_hw,
                self.nb_enc_cropped_hw : patch_size - self.nb_enc_cropped_hw,
            ]
        return y + x_cropped


class ProsailResCNNEncoder(nn.Module):
    """
    This encoder uses convolutional layers with residual connections to process
    spectral-spatial data, mapping reflectance and angle inputs to a latent space
    representation. Designed for patch-based processing of satellite imagery.
    
    Parameters
    ----------
    config : EncoderConfig
        Configuration object containing model hyperparameters
    device : str, optional
        Device to place the model on, by default "cpu"
        
    Attributes
    ----------
    bands : torch.Tensor
        Indices of spectral bands used by the encoder
    output_size : int
        Size of the output latent space
    cnet : nn.Sequential
        Main convolutional network with residual blocks
    mu_conv : nn.Conv2d
        Convolution layer for mean prediction
    logvar_conv : nn.Conv2d
        Convolution layer for variance prediction
    mu_logvar_conv : nn.Conv2d
        Combined convolution for mean and variance
    nb_enc_cropped_hw : int
        Number of pixels cropped from height/width due to valid padding
    _spatial_encoding : bool
        Whether this encoder processes spatial information
    disable_s2_r_idx : bool
        Whether spectral indices computation is disabled
        
    Methods
    -------
    encode(s2_refl: torch.Tensor, angles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        Encode spectral reflectance and viewing angles into latent space
    change_device(device: str)
        Move the model to a different device
    get_spatial_encoding() -> bool
        Return whether this is a spatial encoder
    """

    def __init__(self, config: EncoderConfig, device="cpu"):
        super().__init__()
        bands = config.bands
        if bands is None:
            bands = torch.arange(10)
        self.bands = bands.to(device)
        self.device = device
        network = []
        self.output_size = config.output_size
        network.append(
            nn.Conv2d(
                config.input_size,
                config.first_layer_size,
                config.first_layer_kernel,
                padding=config.padding,
            )
        )
        network.append(nn.ReLU())
        input_sizes = [config.first_layer_size] + config.block_layer_sizes
        assert len(config.block_layer_sizes) == len(config.block_layer_depths)
        assert len(config.block_layer_depths) == len(config.block_kernel_sizes)
        assert len(config.block_kernel_sizes) == len(config.block_n)
        n_groups = len(config.block_n)
        for i in range(n_groups):
            for _ in range(config.block_n[i]):
                network.append(
                    EncoderCResNetBlock(
                        output_size=config.block_layer_sizes[i],
                        depth=config.block_layer_depths[i],
                        kernel_size=config.block_kernel_sizes[i],
                        input_size=input_sizes[i],
                        padding=config.padding,
                    )
                )
                network.append(nn.ReLU())
        self.cnet = nn.Sequential(*network).to(device)
        self.mu_conv = nn.Conv2d(
            input_sizes[-1], config.output_size, kernel_size=1, padding=config.padding
        ).to(device)
        self.logvar_conv = nn.Conv2d(
            input_sizes[-1], config.output_size, kernel_size=1, padding=config.padding
        ).to(device)
        self.mu_logvar_conv = nn.Conv2d(
            input_sizes[-1],
            config.n_latent_params * config.output_size,
            kernel_size=1,
            padding=config.padding,
        ).to(device)
        bands_loc = config.io_coeffs.bands.loc
        idx_loc = config.io_coeffs.idx.loc
        angles_loc = config.io_coeffs.angles.loc
        bands_scale = config.io_coeffs.bands.scale
        idx_scale = config.io_coeffs.idx.scale
        angles_scale = config.io_coeffs.angles.scale
        bands_loc = (
            bands_loc if bands_loc is not None else torch.zeros(config.input_size)
        )
        bands_scale = (
            bands_scale if bands_scale is not None else torch.ones(config.input_size)
        )
        idx_loc = idx_loc if idx_loc is not None else torch.zeros(5)
        idx_scale = idx_scale if idx_scale is not None else torch.ones(5)
        angles_loc = angles_loc if angles_loc is not None else torch.zeros(3)
        angles_scale = angles_scale if angles_scale is not None else torch.ones(3)
        self.bands_loc = bands_loc.float().to(device)
        self.bands_scale = bands_scale.float().to(device)
        self.idx_loc = idx_loc.float().to(device)
        self.idx_scale = idx_scale.float().to(device)
        self.angles_loc = angles_loc.float().to(device)
        self.angles_scale = angles_scale.float().to(device)

        self.nb_enc_cropped_hw = config.first_layer_kernel // 2
        for i in range(n_groups):
            for _ in range(config.block_n[i]):
                for _ in range(config.block_layer_depths[i]):
                    self.nb_enc_cropped_hw += config.block_kernel_sizes[i] // 2
        self._spatial_encoding = True
        self.disable_s2_r_idx = config.disable_s2_r_idx

    def get_spatial_encoding(self):
        return self._spatial_encoding

    def encode(self, s2_refl, angles):
        is_patch = check_is_patch(s2_refl)
        if not is_patch:
            raise AttributeError(
                "Input data is a not a patch: spatial encoder can only"
                " take patches as input"
            )

        normed_refl = standardize(s2_refl, self.bands_loc, self.bands_scale, dim=1)[
            :, self.bands, ...
        ]
        if len(normed_refl.size()) == 3:
            normed_refl = normed_refl.unsqueeze(0)
        if len(angles.size()) == 3:
            angles = angles.unsqueeze(0)

        normed_angles = standardize(
            torch.cos(deg2rad(angles)), self.angles_loc, self.angles_scale, dim=1
        )
        encoder_input = torch.concat((normed_refl, normed_angles), axis=1)
        if not self.disable_s2_r_idx:
            spectral_idx = get_spectral_idx(s2_refl, bands_dim=1)
            encoder_input = torch.concat((encoder_input, spectral_idx), axis=1)
        y = self.cnet(encoder_input)

        # Optionally, separate mu/logvar:
        # y_mu_logvar = torch.concat([self.mu_conv(y), self.logvar_conv(y)], axis=1)
        # Or use single conv:
        y_mu_logvar = self.mu_logvar_conv(y)

        angles = angles[
            :,
            :,
            self.nb_enc_cropped_hw : -self.nb_enc_cropped_hw,
            self.nb_enc_cropped_hw : -self.nb_enc_cropped_hw,
        ]
        return batchify_batch_latent(y_mu_logvar), batchify_batch_latent(angles)

    def forward(self, s2_refl, angles):
        return self.encode(s2_refl, angles)

    def change_device(self, device):
        self.device = device
        self.bands_loc = self.bands_loc.to(device)
        self.bands_scale = self.bands_scale.to(device)
        self.idx_loc = self.idx_loc.to(device)
        self.idx_scale = self.idx_scale.to(device)
        self.angles_loc = self.angles_loc.to(device)
        self.angles_scale = self.angles_scale.to(device)
        self.cnet = self.cnet.to(device)
        self.mu_conv = self.mu_conv.to(device)
        self.logvar_conv = self.logvar_conv.to(device)
        self.mu_logvar_conv = self.mu_logvar_conv.to(device)
        self.bands = self.bands.to(device)


class ProsailTransformerEncoder(Encoder):
    """
    A Transformer-based encoder for processing PROSAIL reflectances and viewing angles.
    
    This encoder uses a transformer architecture to learn complex relationships between
    spectral bands and viewing angles. It processes input data as a sequence where each
    spectral band is treated as a token, with positional embeddings to maintain order.
    
    Parameters
    ----------
    config : EncoderConfig
        Configuration object containing model hyperparameters including transformer-specific settings
    device : str | torch.device | None, optional
        Device to place the model on (auto-selects CUDA if available), by default None
        
    Attributes
    ----------
    d_model : int
        Dimension of the transformer model's hidden states
    n_heads : int
        Number of attention heads in transformer layers
    num_layers : int
        Number of transformer encoder layers
    dropout : float
        Dropout rate for transformer layers
    input_size : int
        Size of input features (reflectance + angles + spectral indices)
    output_size : int
        Size of output latent space
    token_embedding : nn.Linear
        Embedding layer for spectral bands
    pos_embedding : nn.Embedding
        Positional embeddings for sequence order
    transformer_encoder : nn.TransformerEncoder
        Main transformer encoder stack
    linear_out : nn.Linear
        Output projection to latent space
        
    Methods
    -------
    encode(s2_refl: torch.Tensor, angles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        Encode spectral reflectance and viewing angles into latent space
    get_spatial_encoding() -> bool
        Return whether this is a spatial encoder (always False)
    change_device(device: str)
        Move the model to a different device
    forward(s2_refl: torch.Tensor, angles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        Forward pass, calls encode()
    """

    def __init__(self, config: EncoderConfig, device: str | torch.device | None = None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.bands = config.bands
        if self.bands is None:
            self.bands = torch.arange(10)
        self.bands = self.bands.to(device)

        # Transformer hyperparams
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.num_layers = config.n_transformer_layers
        self.dropout = config.dropout

        self.input_size = config.input_size  # reflectance + angles + possibly spectral idx
        self.output_size = config.output_size
        self.n_latent_params = config.n_latent_params

        # Normalization parameters, from config.io_coeffs
        # (mirroring how RNN & ResCNN handle them)
        self.bands_loc = config.io_coeffs.bands.loc
        self.bands_scale = config.io_coeffs.bands.scale
        self.angles_loc = config.io_coeffs.angles.loc
        self.angles_scale = config.io_coeffs.angles.scale

        if self.bands_loc is None:
            self.bands_loc = torch.zeros(self.input_size)
        if self.bands_scale is None:
            self.bands_scale = torch.ones(self.input_size)
        if self.angles_loc is None:
            self.angles_loc = torch.zeros(3)
        if self.angles_scale is None:
            self.angles_scale = torch.ones(3)

        self.bands_loc = self.bands_loc.float().to(device)
        self.bands_scale = self.bands_scale.float().to(device)
        self.angles_loc = self.angles_loc.float().to(device)
        self.angles_scale = self.angles_scale.float().to(device)

        self.disable_s2_r_idx = config.disable_s2_r_idx

        # Token + positional embeddings
        # We'll assume each band is 1 token, so token_embedding in_features=1
        self.token_embedding = nn.Linear(1, self.d_model)
        self.pos_embedding = nn.Embedding(self.input_size, self.d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True,  # shape [batch, seq_len, d_model]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)

        # Final linear projection from d_model -> (mu, logvar) or a single latent vector
        # Here, we follow the style of the RNN: 2 * output_size if we want mu/logvar
        self.latent_dim = config.n_latent_params * config.output_size
        self.linear_out = nn.Linear(self.d_model, self.latent_dim)

        self.to(device)

    def get_spatial_encoding(self):
        """
        Indicate whether this encoder processes spatial information.
        
        Returns
        -------
        bool
            Always False as this is not a spatial encoder
        """
        return False

    def encode(self, s2_refl: torch.Tensor, angles: torch.Tensor):
        """
        Encode spectral reflectance and viewing angles into latent space.
        
        This method:
        1. Normalizes input reflectance and angles
        2. Optionally computes and incorporates spectral indices
        3. Processes inputs through transformer encoder
        4. Maps encoded features to latent space parameters
        
        Parameters
        ----------
        s2_refl : torch.Tensor
            Spectral reflectance data of shape [B, #bands] or [B, #bands, ...]
        angles : torch.Tensor
            Viewing angles of shape [B, 3] or [B, #pix, 3]
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - Encoded latent representation (could be mu/logvar or direct latent)
            - Processed angles
        """
        # Flatten if 4D
        if len(s2_refl.size()) == 4:
            s2_refl = batchify_batch_latent(s2_refl)
            angles = batchify_batch_latent(angles)

        # Standardize reflectance
        if s2_refl.size(1) == self.bands_loc.size(0):
            normed_refl = standardize(s2_refl, self.bands_loc, self.bands_scale, dim=1)
            if len(self.bands) < normed_refl.size(1):
                normed_refl = normed_refl[:, self.bands]
        elif len(self.bands) == self.bands_loc.size(0):
            normed_refl = standardize(
                s2_refl[:, self.bands], self.bands_loc, self.bands_scale, dim=1
            )
        else:
            raise NotImplementedError

        normed_angles = standardize(
            torch.cos(deg2rad(angles)), self.angles_loc, self.angles_scale, dim=1
        )

        # Combine reflectance with spectral indices if not disabled
        encoder_input = normed_refl
        if not self.disable_s2_r_idx:
            spectral_idx = get_spectral_idx(s2_refl, bands_dim=1)
            encoder_input = torch.concat((encoder_input, spectral_idx), axis=1)

        # (Optional) incorporate angles as extra tokens or appended features
        # For simplicity, we'll skip that in this example, but you could incorporate them as well.

        # Reshape so each band is a "token"
        # shape: [B, seq_len, 1]
        encoder_input = encoder_input.unsqueeze(-1)

        # Embedding
        seq_len = encoder_input.size(1)
        x_emb = self.token_embedding(encoder_input)  # [B, seq_len, d_model]

        # Positional embedding
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0)  # [1, seq_len]
        pos_emb = self.pos_embedding(positions)  # [1, seq_len, d_model]
        x_emb = x_emb + pos_emb  # [B, seq_len, d_model]

        # TransformerEncoder
        encoded_seq = self.transformer_encoder(x_emb)  # [B, seq_len, d_model]

        # Pool the sequence dimension
        pooled = encoded_seq.mean(dim=1)  # [B, d_model]

        # Map to latent dimension
        latent_out = self.linear_out(pooled)  # [B, latent_dim]

        return latent_out, angles

    def forward(self, s2_refl, angles):
        """
        Forward pass of the encoder.
        
        Parameters
        ----------
        s2_refl : torch.Tensor
            Spectral reflectance data
        angles : torch.Tensor
            Viewing angles
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Results of encode() method
        """
        return self.encode(s2_refl, angles)

    def change_device(self, device):
        """
        Move the model and all its parameters to specified device.
        
        Parameters
        ----------
        device : str
            Target device ('cpu' or 'cuda')
        """
        self.device = device
        self.to(device)
        # Move any other parameters as needed
        self.bands_loc = self.bands_loc.to(device)
        self.bands_scale = self.bands_scale.to(device)
        self.angles_loc = self.angles_loc.to(device)
        self.angles_scale = self.angles_scale.to(device)
        self.bands = self.bands.to(device)


def get_encoder(config: EncoderConfig, device: str | torch.device = torch.device("cpu")):
    """
    Factory function to create appropriate encoder based on configuration.
    
    Creates and returns an encoder instance based on the specified encoder type
    in the configuration. Supports RNN, CNN, and Transformer architectures.
    
    Parameters
    ----------
    config : EncoderConfig
        Configuration object specifying encoder type and parameters
    device : str | torch.device, optional
        Device to place the model on, by default torch.device("cpu")
        
    Returns
    -------
    Encoder
        Instantiated encoder of the specified type
        
    Raises
    ------
    NotImplementedError
        If the specified encoder_type is not supported
        
    Notes
    -----
    Currently supported encoder types:
    - 'rnn': ProsailRNNEncoder
    - 'rcnn': ProsailResCNNEncoder
    - 'transformer': ProsailTransformerEncoder
    """
    if config.encoder_type == "nn":
        pass  # encoder = ProsailNNEncoder(config, device)
    elif config.encoder_type == "rnn":
        encoder = ProsailRNNEncoder(config, device)
    elif config.encoder_type == "rcnn":
        encoder = ProsailResCNNEncoder(config, device)
    elif config.encoder_type == "cnn":
        pass  # encoder = ProsailCNNEncoder(config, device)
    elif config.encoder_type == "transformer":
        # === NEW CASE FOR TRANSFORMER-BASED ENCODER ===
        encoder = ProsailTransformerEncoder(config, device)
    else:
        raise NotImplementedError(f"Encoder type '{config.encoder_type}' not supported.")
    return encoder
