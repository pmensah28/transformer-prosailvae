#!/usr/bin/env python3
"""
Updated to support a transformer-based encoder approach
while remaining compatible with the existing RNN/CNN encoders.

credit: yoel
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from prosailvae.latentspace import LatentSpace
from prosailvae.simspaces import SimVarSpace
from utils.image_utils import (
    batchify_batch_latent,
    check_is_patch,
    crop_s2_input,
    unbatchify,
)
from utils.utils import NaN_model_params, count_parameters, unstandardize


@dataclass
class SimVAEConfig:
    """
    Configuration for a simulator-based VAE (SimVAE).

    Attributes
    ----------
    encoder : nn.Module
        Torch module that encodes reflectances + angles (etc.) into latent params.
        This can be a CNN, RNN, or Transformer-based encoder.

    decoder : nn.Module
        Torch module that decodes samples from the sim_space parameters (e.g., PROSAIL),
        returning reconstructed reflectances.

    lat_space : LatentSpace
        Defines how the encoder's output is interpreted as a latent distribution.

    sim_space : SimVarSpace
        Maps latent distributions to simulator parameter distributions (e.g., PROSAIL vars).

    reconstruction_loss : nn.Module
        Handles the reconstruction loss function.

    deterministic : bool
        If True, the model may do no sampling in the latent space (sigma=0).

    index_loss : Any | None
        Additional index-based penalty if used.

    supervised : bool
        If True, the VAE is being trained with direct supervision of latent or simulator
        parameters.

    device : torch.device
        Device on which to run computations.

    beta_kl : float
        Weight for the KL divergence term.

    beta_index : float
        Weight for index-based constraints in reconstruction.

    logger_name : str
        Name for the logger to track training progress.

    beta_cyclical : float
        Weight for cyclical reconstruction loss.

    inference_mode : bool
        If True, the model simply encodes but does not decode or sample latents.

    lat_idx : torch.Tensor | list[int] | None
        Indices of the latent vars on which to apply KL. None means all.

    disabled_latent : list[int] | None
        Indices of latent variables to disable (set them to a fixed value).

    disabled_latent_values : list[float] | None
        Values to which the disabled latent variables are set.

    lat_nll : str
        Type of latent NLL used in certain specialized setups.
    """

    encoder: nn.Module
    decoder: nn.Module
    lat_space: LatentSpace
    sim_space: SimVarSpace
    reconstruction_loss: nn.Module
    deterministic: bool = False
    index_loss: Any | None = None
    supervised: bool = False
    device: torch.device = torch.device("cpu")
    beta_kl: float = 0
    beta_index: float = 0
    logger_name: str = "PROSAIL-VAE logger"
    beta_cyclical: float = 0.0
    inference_mode: bool = False
    lat_idx: torch.Tensor | list[int] | None = None
    disabled_latent = None
    disabled_latent_values = None
    lat_nll: str = "diag_nll"


class SimVAE(nn.Module):
    """
    A class for a simulator-based VAE that uses a learned encoder
    and a physically informed (or learned) simulator-decoder.

    The model can be used in unsupervised or supervised modes,
    and supports cyclical training strategies if desired.
    """

    def __init__(self, config: SimVAEConfig):
        """
        Initialize the SimVAE model.
        """
        # handle lat_idx
        if config.lat_idx is None:
            self.lat_idx = torch.tensor([])
        elif isinstance(config.lat_idx, torch.Tensor):
            self.lat_idx = config.lat_idx
        else:
            self.lat_idx = torch.tensor(list(config.lat_idx))

        if config.disabled_latent is None:
            config.disabled_latent = []
        if config.disabled_latent_values is None:
            config.disabled_latent_values = []

        super().__init__()
        # encoder, decoder, latent spaces
        self.encoder = config.encoder
        self.lat_space = config.lat_space
        self.sim_space = config.sim_space
        self.decoder = config.decoder
        self.reconstruction_loss = config.reconstruction_loss
        self.index_loss = config.index_loss
        self.encoder.eval()
        self.lat_space.eval()

        # global training modes
        self.supervised = config.supervised
        self.device = config.device
        self.beta_kl = config.beta_kl
        self.eval()
        self.logger = logging.getLogger(config.logger_name)
        self.logger.info(
            f"Number of trainable parameters: {count_parameters(self.encoder)}"
        )
        self.beta_index = config.beta_index
        self.inference_mode = config.inference_mode
        self.hyper_prior = None
        self.lat_nll = config.lat_nll

        # check if the encoder is "spatial" (e.g., CNN for patch-based data)
        self.spatial_mode = self.encoder.get_spatial_encoding()
        self.deterministic = config.deterministic
        self.beta_cyclical = config.beta_cyclical

    def set_hyper_prior(self, hyper_prior: nn.Module | None = None):
        """
        Optional hyper-prior, e.g. a teacher model used as a prior.
        """
        self.hyper_prior = hyper_prior

    def change_device(self, device: str | torch.device):
        """
        Move model and all submodules to desired device.
        """
        self.device = device
        self.encoder.change_device(device)
        self.lat_space.change_device(device)
        self.sim_space.change_device(device)
        self.decoder.change_device(device)
        if self.hyper_prior is not None:
            self.hyper_prior.change_device(device)

    def encode(self, s2_r, s2_a):
        """
        Encode reflectances + angles using the encoder.
        Returns (y, angles).
        """
        y, angles = self.encoder.encode(s2_r, s2_a)
        return y, angles

    def encode2lat_params(self, s2_r, s2_a, deterministic=False):
        """
        Encode data into latent distribution parameters.
        If deterministic=True, set sigma=0.
        """
        y, _ = self.encode(s2_r, s2_a)
        dist_params = self.lat_space.get_params_from_encoder(y)
        if deterministic:
            dist_params[..., 1] = 0.0
        return dist_params

    def sample_latent_from_params(self, dist_params, n_samples=1, deterministic=False):
        """
        Sample from latent distribution described by dist_params.
        If deterministic=True, skip random sampling.
        """
        z = self.lat_space.sample_latent_from_params(
            dist_params, n_samples=n_samples, deterministic=deterministic
        )
        return z

    def transfer_latent(self, z):
        """
        Map latent samples z to physical variables (simulator parameters).
        """
        sim = self.sim_space.z2sim(z)
        return sim

    def decode(self, sim, angles, apply_norm=None):
        """
        Decode simulator parameters into reflectances via self.decoder.
        """
        rec = self.decoder.decode(sim, angles, apply_norm=apply_norm)
        return rec

    def freeze_weigths(self):
        """
        Freeze all learnable parameters in this model.
        """
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, angles=None, n_samples=1, apply_norm=None):
        """
        Full forward pass through the VAE:
          1) encode -> 2) sample latents -> 3) decode -> 4) reconstruction
        """
        is_patch = check_is_patch(x)
        if angles is None:
            # last 3 columns might be angles if appended
            angles = x[:, -3:]
            x = x[:, :-3]
        batch_size = x.size(0)

        # 1) Encode
        y, angles = self.encode(x, angles)
        dist_params = self.lat_space.get_params_from_encoder(y)

        if self.inference_mode:
            # if just encoding, stop here
            return dist_params, None, None, None

        # 2) Sample latents
        z = self.sample_latent_from_params(
            dist_params, n_samples=n_samples, deterministic=self.deterministic
        )

        # 3) Transfer to simulator space
        sim = self.transfer_latent(z)

        # 4) Decode
        rec = self.decode(sim, angles, apply_norm=apply_norm)

        if is_patch:
            return (
                dist_params,
                unbatchify(z, batch_size=batch_size),
                unbatchify(sim, batch_size=batch_size),
                unbatchify(rec, batch_size=batch_size),
            )
        else:
            return dist_params, z, sim, rec

    def point_estimate_rec(self, x, angles, mode="random", apply_norm=False):
        """
        Forward pass with a "point estimate" of the latent distribution.
        E.g., random sample, mode, or median, etc.
        """
        is_patch = check_is_patch(x)
        if angles is None:
            angles = x[:, -3:]
            x = x[:, :-3]

        # 1) Encode
        y, angles = self.encode(x, angles)
        dist_params = self.lat_space.get_params_from_encoder(y)

        # 2) Choose how to get z (random / mode / median / etc.)
        if mode == "random":
            if self.inference_mode:
                return dist_params, None, None, None
            z = self.sample_latent_from_params(dist_params, n_samples=1)
            sim = self.transfer_latent(z)
        elif mode == "lat_mode":
            z = self.lat_space.mode(dist_params)
            sim = self.transfer_latent(z.unsqueeze(2))
        elif mode == "sim_tg_mean":
            z = self.lat_space.expectation(dist_params)
            sim = self.transfer_latent(z.unsqueeze(2))
        elif mode == "sim_mode":
            lat_pdfs, lat_supports = self.lat_space.latent_pdf(dist_params)
            sim = self.sim_space.sim_mode(lat_pdfs, lat_supports, n_pdf_sample_points=5001)
            z = self.sim_space.sim2z(sim)
        elif mode == "sim_median":
            lat_pdfs, lat_supports = self.lat_space.latent_pdf(dist_params)
            sim = self.sim_space.sim_median(lat_pdfs, lat_supports, n_samples=5001)
            z = self.sim_space.sim2z(sim)
        elif mode == "sim_expectation":
            lat_pdfs, lat_supports = self.lat_space.latent_pdf(dist_params)
            sim = self.sim_space.sim_expectation(lat_pdfs, lat_supports, n_samples=5001)
            z = self.sim_space.sim2z(sim)
        else:
            raise NotImplementedError(f"Unknown point estimate mode '{mode}'.")

        # 3) Decode
        rec = self.decode(sim, angles, apply_norm=apply_norm)

        if is_patch:
            batch_size = x.size(0)
            rec = unbatchify(rec, batch_size=batch_size)
            sim = unbatchify(sim, batch_size=batch_size)
            if mode != "random":
                dist_params = unbatchify(dist_params, batch_size=batch_size)

        return dist_params, z, sim, rec

    def point_estimate_sim(self, x, angles, mode="random", unbatch=True):
        """
        Similar to point_estimate_rec, but only returning the simulator parameters, not the reconstruction.
        """
        is_patch = check_is_patch(x)
        if angles is None:
            angles = x[:, -3:]
            x = x[:, :-3]
        y, angles = self.encode(x, angles)
        dist_params = self.lat_space.get_params_from_encoder(y)

        if mode == "random":
            if self.inference_mode:
                return dist_params, None, None
            z = self.sample_latent_from_params(dist_params, n_samples=1)
            sim = self.transfer_latent(z)
        elif mode == "lat_mode":
            z = self.lat_space.mode(dist_params)
            sim = self.transfer_latent(z.unsqueeze(2))
        elif mode == "sim_tg_mean":
            z = self.lat_space.expectation(dist_params)
            sim = self.transfer_latent(z.unsqueeze(2))
        elif mode == "sim_mode":
            lat_pdfs, lat_supports = self.lat_space.latent_pdf(dist_params)
            sim = self.sim_space.sim_mode(lat_pdfs, lat_supports, n_pdf_sample_points=5001)
            z = self.sim_space.sim2z(sim)
        elif mode == "sim_median":
            lat_pdfs, lat_supports = self.lat_space.latent_pdf(dist_params)
            sim = self.sim_space.sim_median(lat_pdfs, lat_supports, n_samples=5001)
            z = self.sim_space.sim2z(sim)
        elif mode == "sim_expectation":
            lat_pdfs, lat_supports = self.lat_space.latent_pdf(dist_params)
            sim = self.sim_space.sim_expectation(lat_pdfs, lat_supports, n_samples=5001)
            z = self.sim_space.sim2z(sim)
        else:
            raise NotImplementedError(f"Unknown point estimate mode '{mode}'.")

        if is_patch and unbatch:
            batch_size = x.size(0)
            if mode == "random":
                return unbatchify(dist_params, batch_size=batch_size), z, sim
            return (
                unbatchify(dist_params, batch_size=batch_size),
                z,
                unbatchify(sim, batch_size=batch_size),
            )
        return dist_params, z, sim

    def crop_patch(
        self, s2_r: torch.Tensor, s2_a: torch.Tensor, z: torch.Tensor, rec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Crop patches if needed after convolution-based encoders.
        (Patch edges get lost in conv layers, etc.)
        """
        assert check_is_patch(rec)
        s2_r = crop_s2_input(s2_r, self.encoder.nb_enc_cropped_hw)
        s2_a = crop_s2_input(s2_a, self.encoder.nb_enc_cropped_hw)
        z = crop_s2_input(z, self.encoder.nb_enc_cropped_hw)

    def compute_rec_loss(self, s2_r: torch.Tensor, rec: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss (e.g., NLL or MSE).
        If the sensor simulator applies normalization, do that before comparing.
        """
        if self.decoder.ssimulator.apply_norm:
            return self.reconstruction_loss(self.decoder.ssimulator.normalize(s2_r), rec)
        else:
            return self.reconstruction_loss(s2_r, rec)

    def compute_cyclical_loss(self, s2_r, s2_a, z, rec, batch_size, n_samples):
        """
        Helper for cyclical training strategies (re-encoding the reconstruction).
        """
        sample_dim = self.reconstruction_loss.sample_dim
        feature_dim = self.reconstruction_loss.feature_dim

        # "un-standardize" the reconstruction
        rec_cyc = unstandardize(
            rec, self.encoder.bands_loc, self.encoder.bands_scale, dim=feature_dim
        )
        # Flatten batch + sample dims
        rec_cyc = rec_cyc.transpose(sample_dim, 1)
        rec_cyc = rec_cyc.reshape(-1, *rec_cyc.shape[2:])

        s2_a_cyc = s2_a.unsqueeze(sample_dim)
        s2_a_cyc = s2_a_cyc.tile(
            [(n_samples if i == sample_dim else 1) for i in range(len(s2_a_cyc.size()))]
        )
        s2_a_cyc = s2_a_cyc.transpose(sample_dim, 1)
        s2_a_cyc = s2_a_cyc.reshape(-1, *s2_a_cyc.shape[2:])
        z_cyc = z.transpose(feature_dim, -1).reshape(-1, z.size(feature_dim))
        return (rec_cyc, s2_a_cyc, z_cyc)

    def unsupervised_batch_loss(
        self, batch, normalized_loss_dict, len_loader=1, n_samples=1
    ):
        """
        Compute unsupervised VAE loss (ELBO) on one batch.
        """
        # 1) Forward pass
        s2_r, s2_a, distri_params, z, sim, rec = self.generate_outputs(batch, n_samples)
        batch_size = s2_r.size(0)

        # 2) Crop if spatial
        if self.spatial_mode:
            s2_r, s2_a, z = self.crop_patch(s2_r, s2_a, z, rec)

        # 3) Reconstruction term
        rec_loss = self.compute_rec_loss(s2_r, rec)
        loss_dict = {"rec_loss": rec_loss.item()}
        loss_sum = rec_loss

        # 4) If cyclical training is enabled
        if self.beta_cyclical > 0:
            cyclical_batch = self.compute_cyclical_loss(
                s2_r, s2_a, z, rec, batch_size, n_samples
            )
            cyclical_loss, _ = self.supervised_batch_loss(
                cyclical_batch, {}, ref_is_lat=True
            )
            loss_sum += self.beta_cyclical * cyclical_loss
            loss_dict["cyclical_loss"] = cyclical_loss.item()

        # 5) KL term
        if self.beta_kl > 0:
            kl_loss = self.compute_kl_elbo(s2_r, s2_a, distri_params)
            loss_sum += kl_loss
            loss_dict["kl_loss"] = kl_loss.item()

        # 6) Index-based constraints
        if self.beta_index > 0:
            index_loss = self.beta_index * self.decoder.ssimulator.index_loss(
                s2_r,
                rec,
                lossfn=self.decoder.rec_loss_fn,
                normalize_idx=True,
                s2_r_bands_dim=1,
                rec_bands_dim=self.decoder.rec_loss_fn.feature_dim,
            )
            loss_sum += index_loss
            loss_dict["index_loss"] = index_loss.item()

        loss_dict["loss_sum"] = loss_sum.item()
        for loss_type, loss_val in loss_dict.items():
            if loss_type not in normalized_loss_dict:
                normalized_loss_dict[loss_type] = 0.0
            normalized_loss_dict[loss_type] += loss_val

        return loss_sum, normalized_loss_dict

    def supervised_batch_loss(
        self, batch, normalized_loss_dict, len_loader=1, ref_is_lat=False
    ):
        """
        Compute supervised loss on the latent parameters:
        e.g., if we have ground-truth simulator variables or latents.
        """
        s2_r = batch[0].to(self.device)
        s2_a = batch[1].to(self.device)
        ref_lat = batch[2].to(self.device)

        # ref_lat can be raw simulator variables or latent space variables
        if not ref_is_lat:
            ref_lat = self.sim_space.sim2z(ref_lat)

        encoder_output, _ = self.encode(s2_r, s2_a)
        if encoder_output.isnan().any() or encoder_output.isinf().any():
            nan_in_params = NaN_model_params(self)
            err_str = (
                "NaN encountered during encoding, "
                "but there is no NaN in network parameters!"
            )
            if nan_in_params:
                err_str = (
                    "NaN encountered during encoding, "
                    "and there are NaN in network parameters!"
                )
            raise ValueError(err_str)

        # Convert encoder output to distribution parameters
        params = self.lat_space.get_params_from_encoder(encoder_output=encoder_output)
        reduction_nll = "sum"
        if self.lat_nll == "lai_nll":
            reduction_nll = "lai"
        loss_sum = self.lat_space.supervised_loss(
            ref_lat, params, reduction_nll=reduction_nll
        )

        if loss_sum.isnan().any() or loss_sum.isinf().any():
            raise ValueError("NaN/inf in supervised loss.")

        all_losses = {"lat_loss": loss_sum.item(), "loss_sum": loss_sum.item()}
        for loss_type, loss_val in all_losses.items():
            if loss_type not in normalized_loss_dict:
                normalized_loss_dict[loss_type] = 0.0
            normalized_loss_dict[loss_type] += loss_val

        return loss_sum, normalized_loss_dict

    def compute_lat_nlls_batch(self, batch):
        """
        Compute negative log-likelihood for a batch
        if we have ground-truth latents or sim parameters.
        """
        s2_r = batch[0].to(self.device)
        s2_a = batch[1].to(self.device)
        ref_sim = batch[2].to(self.device)
        ref_lat = self.sim_space.sim2z(ref_sim)

        encoder_output, _ = self.encode(s2_r, s2_a)
        params = self.lat_space.get_params_from_encoder(encoder_output=encoder_output)
        nll = self.lat_space.supervised_loss(ref_lat, params, reduction=None, reduction_nll=None)
        if nll.isnan().any() or nll.isinf().any():
            raise ValueError("NaN/inf in lat_nll.")
        return nll

    def compute_lat_nlls(self, dataloader, batch_per_epoch=None):
        """
        Compute lat NLL over an entire dataloader, returning an array of nll values.
        """
        self.eval()
        all_nlls = []
        with torch.no_grad():
            if batch_per_epoch is None:
                batch_per_epoch = len(dataloader)
            for _, batch in zip(range(min(len(dataloader), batch_per_epoch)), dataloader):
                nll_batch = self.compute_lat_nlls_batch(batch)
                all_nlls.append(nll_batch)
                if torch.isnan(nll_batch).any():
                    self.logger.error("NaN Loss encountered in lat NLLs!")
        all_nlls = torch.vstack(all_nlls)
        return all_nlls

    def save_ae(self, epoch: int, optimizer, loss, path: str):
        """
        Save the model + optimizer state to disk.
        """
        hyper_prior = None
        if self.hyper_prior is not None:
            hyper_prior = self.hyper_prior
            self.set_hyper_prior(None)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            path,
        )
        if hyper_prior is not None:
            self.set_hyper_prior(hyper_prior)

    def load_ae(self, path: str, optimizer=None, weights_only: bool = False):
        """
        Load the model from disk. Optionally load optimizer state too.
        """
        hyper_prior = None
        if self.hyper_prior is not None:
            hyper_prior = self.hyper_prior
            self.set_hyper_prior(None)

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        try:
            self.load_state_dict(checkpoint["model_state_dict"])
        except Exception as exc:
            print("checkpoint state dict", checkpoint["model_state_dict"].keys())
            print("model state dict", self.state_dict().keys())
            raise ValueError from exc

        if optimizer is not None and not weights_only:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if hyper_prior is not None:
            self.set_hyper_prior(hyper_prior)

        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        return epoch, loss

    def fit(self, dataloader, optimizer, n_samples=1, max_samples=None, accum_iter=1):
        """
        Train for one epoch over a dataloader.
        """
        if max_samples is not None:
            accum_iter = min(accum_iter, max_samples)
        self.train()
        train_loss_dict = {}
        len_loader = len(dataloader.dataset)
        n_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            if NaN_model_params(self):
                self.logger.debug("NaN model parameters at batch %d!", batch_idx)
            if max_samples is not None and batch_idx == max_samples:
                break

            try:
                n_batches += batch[0].size(0)
                if not self.supervised:
                    loss_sum, train_loss_dict = self.unsupervised_batch_loss(
                        batch, train_loss_dict, n_samples=n_samples, len_loader=len_loader
                    )
                else:
                    loss_sum, train_loss_dict = self.supervised_batch_loss(
                        batch, train_loss_dict, len_loader=len_loader
                    )
            except Exception as exc:
                self.logger.error("Couldn't compute loss at batch %d!", batch_idx)
                self.logger.error("s2_r : %d NaN", torch.isnan(batch[0]).sum().item())
                self.logger.error("s2_a : %d NaN", torch.isnan(batch[1]).sum().item())
                self.logger.error(exc)
                raise ValueError(f"Couldn't compute loss at batch {batch_idx}!") from exc

            if torch.isnan(loss_sum).any():
                self.logger.error("NaN Loss encountered during training at batch %d!", batch_idx)

            loss_sum = loss_sum / accum_iter
            loss_sum.backward()
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(dataloader)):
                optimizer.step()
                if NaN_model_params(self):
                    self.logger.debug("NaN model parameters after batch %d!", batch_idx)
                optimizer.zero_grad()

        for loss_type, loss_val in train_loss_dict.items():
            train_loss_dict[loss_type] = loss_val / n_batches
        self.eval()
        return train_loss_dict

    def validate(self, dataloader, n_samples=1, batch_per_epoch=None, max_samples=None):
        """
        Validate for one epoch.
        """
        self.eval()
        valid_loss_dict = {}
        len_loader = len(dataloader.dataset)
        n_batches = 0

        with torch.no_grad():
            if batch_per_epoch is None:
                batch_per_epoch = len(dataloader)
            for i, batch in zip(range(min(len(dataloader), batch_per_epoch)), dataloader):
                n_batches += batch[0].size(0)
                if max_samples is not None and i == max_samples:
                    break
                if not self.supervised:
                    loss_sum, _ = self.unsupervised_batch_loss(
                        batch, valid_loss_dict, n_samples=n_samples, len_loader=len_loader
                    )
                else:
                    loss_sum, _ = self.supervised_batch_loss(
                        batch, valid_loss_dict, len_loader=len_loader
                    )
                if torch.isnan(loss_sum).any():
                    self.logger.error("NaN Loss encountered during validation!")

        for loss_type, loss_val in valid_loss_dict.items():
            valid_loss_dict[loss_type] = loss_val / n_batches
        return valid_loss_dict

    def get_cyclical_loss_from_batch(self, batch, n_samples=1):
        """
        Utility for cyclical training strategies:
        1) forward pass to get reconstruction
        2) re-encode the reconstruction to see if we get the same latents
        """
        s2_r = batch[0].to(self.device)
        s2_a = batch[1].to(self.device)
        input_is_patch = check_is_patch(s2_r)
        if self.spatial_mode:
            assert input_is_patch
        else:
            if input_is_patch:
                s2_r = batchify_batch_latent(s2_r)
                s2_a = batchify_batch_latent(s2_a)

        _, z, sim, rec = self.forward(s2_r, n_samples=n_samples, angles=s2_a)
        if self.spatial_mode:
            assert check_is_patch(rec)
            s2_a = crop_s2_input(s2_a, self.encoder.nb_enc_cropped_hw)
            z = crop_s2_input(z, self.encoder.nb_enc_cropped_hw)

        sample_dim = self.reconstruction_loss.sample_dim
        feature_dim = self.reconstruction_loss.feature_dim

        rec_cyc = unstandardize(
            rec, self.encoder.bands_loc, self.encoder.bands_scale, dim=feature_dim
        )
        rec_cyc = rec_cyc.transpose(sample_dim, 1).reshape(-1, *rec_cyc.shape[2:])

        s2_a_cyc = s2_a.unsqueeze(sample_dim).tile(
            [(n_samples if i == sample_dim else 1) for i in range(len(s2_a.size()))]
        )
        s2_a_cyc = s2_a_cyc.transpose(sample_dim, 1).reshape(-1, *s2_a_cyc.shape[2:])
        z_cyc = z.transpose(feature_dim, -1).reshape(-1, z.size(feature_dim))

        cyclical_batch = (rec_cyc, s2_a_cyc, z_cyc)
        cyclical_loss, _ = self.supervised_batch_loss(cyclical_batch, {}, ref_is_lat=True)
        return cyclical_loss

    def get_cyclical_lai_squared_error_from_batch(self, batch, mode="lat_mode", lai_precomputed=False):
        """
        Example cyclical metric for LAI. 
        This function tries to see how consistent LAI is after re-encoding the reconstruction.
        """
        s2_r = batch[0].to(self.device)
        s2_a = batch[1].to(self.device)
        lai_idx = 6  # index of LAI param in your sim space
        input_is_patch = check_is_patch(s2_r)

        if self.spatial_mode:
            assert input_is_patch
        else:
            if input_is_patch:
                s2_r = batchify_batch_latent(s2_r)
                s2_a = batchify_batch_latent(s2_a)

        sample_dim = self.reconstruction_loss.sample_dim
        feature_dim = self.reconstruction_loss.feature_dim

        if not lai_precomputed:
            # 1) compute rec from (s2_r, s2_a)
            _, z, sim, s2_r = self.point_estimate_rec(s2_r, angles=s2_a, mode=mode)
            if self.spatial_mode:
                assert check_is_patch(s2_r)
                s2_a = crop_s2_input(s2_a, self.encoder.nb_enc_cropped_hw)
                sim = crop_s2_input(sim, self.encoder.nb_enc_cropped_hw)

            s2_r = s2_r.transpose(sample_dim, 1).reshape(-1, *s2_r.shape[2:])
            s2_a = s2_a.unsqueeze(sample_dim).transpose(sample_dim, 1).reshape(-1, *s2_a.shape[2:])
        else:
            # if LAI is precomputed
            sim = batch[2].to(self.device)
            if self.spatial_mode:
                sim = crop_s2_input(sim, self.encoder.nb_enc_cropped_hw)
            else:
                sim = batchify_batch_latent(sim)

        # 2) re-encode the reconstruction
        _, _, sim_cyc = self.point_estimate_sim(s2_r, s2_a, mode=mode)

        # 3) compare LAI
        return (sim_cyc.select(feature_dim, lai_idx) - sim.select(feature_dim, lai_idx)).pow(2)

    def get_cyclical_metrics_from_loader(
        self, dataloader, n_samples=1, batch_per_epoch=None, max_samples=None, lai_precomputed=False
    ):
        """
        Example function to gather cyclical metrics (like cyclical_loss, cyclical_rmse)
        from an entire dataloader.
        """
        self.eval()
        n_batches = 0
        cyclical_loss = []
        cyclical_rmse = []
        with torch.no_grad():
            if batch_per_epoch is None:
                batch_per_epoch = len(dataloader)
            for i, batch in zip(range(min(len(dataloader), batch_per_epoch)), dataloader):
                n_batches += batch[0].size(0)
                if max_samples is not None and i == max_samples:
                    break
                # cyclical_loss_val = self.get_cyclical_loss_from_batch(batch, n_samples=n_samples)
                # cyclical_loss.append(cyclical_loss_val.unsqueeze(0))
                cyclical_rmse_val = self.get_cyclical_lai_squared_error_from_batch(
                    batch, mode="lat_mode", lai_precomputed=lai_precomputed
                )
                cyclical_rmse.append(cyclical_rmse_val.unsqueeze(0))

        # cyclical_loss = torch.cat(cyclical_loss).mean()
        cyclical_loss_val = 0  # or implement if you want
        cyclical_rmse_val = torch.cat(cyclical_rmse).mean().sqrt()
        return cyclical_loss_val, cyclical_rmse_val

    def get_cyclical_rmse_from_loader(self, dataloader, batch_per_epoch=None, max_samples=None, lai_precomputed=False):
        """
        Just returns cyclical RMSE from LAI across the entire dataloader.
        """
        self.eval()
        n_batches = 0
        cyclical_rmse = []
        with torch.no_grad():
            if batch_per_epoch is None:
                batch_per_epoch = len(dataloader)
            for i, batch in zip(range(min(len(dataloader), batch_per_epoch)), dataloader):
                n_batches += batch[0].size(0)
                if max_samples is not None and i == max_samples:
                    break
                cyclical_rmse.append(
                    self.get_cyclical_lai_squared_error_from_batch(
                        batch, mode="lat_mode", lai_precomputed=lai_precomputed
                    ).reshape(-1)
                )
        cyclical_rmse_val = torch.cat(cyclical_rmse, 0).mean().sqrt()
        return cyclical_rmse_val

    def pvae_batch_extraction(self, batch: list) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract reflectances + angles from a batch, handling patch or pixel data.
        """
        s2_r = batch[0]
        s2_a = batch[1]
        input_is_patch = check_is_patch(s2_r)
        if self.spatial_mode:
            # If the encoder is a spatial CNN, the input should remain patch-based
            assert input_is_patch
        else:
            # If the encoder is pixel-based (RNN, or Transformer for each pixel),
            # we flatten patches to batch dimension
            if input_is_patch:
                s2_r = batchify_batch_latent(s2_r)
                s2_a = batchify_batch_latent(s2_a)
        return s2_r, s2_a

    def generate_outputs(self, batch: list, n_samples: int = 70):
        """
        Forward pass returning distribution params, latent samples, sim, and rec
        for a given batch.
        """
        s2_r, s2_a = self.pvae_batch_extraction(batch)
        distri_params, z, sim, rec = self.forward(s2_r, n_samples=n_samples, angles=s2_a)
        return s2_r, s2_a, distri_params, z, sim, rec

    def compute_kl_elbo(self, s2_r: torch.Tensor, s2_a: torch.Tensor, distri_params: torch.Tensor) -> torch.Tensor | int:
        """
        Compute KL divergence between encoder latents and prior (uniform or hyper-prior).
        """
        kl_loss = 0
        if self.beta_kl > 0:
            if self.hyper_prior is None:
                # KL Truncated Normal vs Uniform
                kl_loss = (
                    self.beta_kl
                    * self.lat_space.kl(distri_params, lat_idx=self.lat_idx)
                    .sum(1)
                    .mean()
                )
            else:
                # KL Truncated Normal vs a hyper-prior that is also a truncated normal
                s2_r_sup = s2_r
                s2_a_sup = s2_a
                if self.spatial_mode and self.hyper_prior.encoder.get_spatial_encoding():
                    # If both are patch-based? Not fully implemented
                    raise NotImplementedError(
                        "Hyper-prior with patch-based hierarchical approach not implemented"
                    )
                else:
                    # Flatten if needed
                    if self.spatial_mode:
                        s2_r_sup = batchify_batch_latent(s2_r_sup)
                        s2_a_sup = batchify_batch_latent(s2_a_sup)

                with torch.no_grad():
                    params_hyper = self.hyper_prior.encode2lat_params(s2_r_sup, s2_a_sup)
                kl_loss = (
                    self.beta_kl
                    * self.lat_space.kl(distri_params, params_hyper, lat_idx=self.lat_idx)
                    .sum(1)
                    .mean()
                )
        return kl_loss
