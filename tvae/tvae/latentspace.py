#!/usr/bin/env python3
"""
Credit: yoel zerah
"""
import torch
import torch.nn as nn

from prosailvae.dist_utils import truncated_gaussian_nll
from utils.TruncatedNormal import (
    TruncatedNormal,
    kl_truncated_normal_truncated_normal,
    kl_truncated_normal_uniform,
)
from utils.utils import torch_select_unsqueeze


class LatentSpace(nn.Module):
    """
    A class used to represent the latent space of an auto-encoder.
    All latent spaces are to inherit from this class
    ...

    Methods
    -------
    reparametrize(y, n_samples)
        Uses encoding y of time series and outputs n_samples samples from latent
    distribution.
    latent_pdf(params, support_sampling)
        Uses latent distributions parameters derived from encoding of time series to
    output
        a discretized pdf of the latent distribution sampled with step support_sampling.
    loss(z, params)
        computes a loss (likely a NLL) of latent parameters params from sample z.
    """

    def reparametrize(self, y: torch.Tensor, n_samples: int):
        raise NotImplementedError()

    def latent_pdf(self, params: torch.Tensor, support_sampling: float):
        raise NotImplementedError()

    def loss(self, z: torch.Tensor, params: torch.Tensor):
        raise NotImplementedError()


class TruncatedNormalLatent(LatentSpace):
    """A Latent distribution with independent Truncated Normal variables.
    Assumes all variables are in the [0,1] interval.

    Some latent variables can be disabled when sampling. This means
    that, when generating samples, the disabled variables will not be
    random, but deterministic values. To disable a variable, the
    'disabled_latent' field can contain a list of indices corresponding
    to the variables that need to be disabled. These are positional
    indices: index 0 is the first variable. The deterministic values for
    the disabled variables have to be set using the
    'disabled_latenet_values'.

    """

    def __init__(
        self,
        latent_dim: int,
        min_sigma: float = 5e-4,
        max_sigma: float = 1.4,
        device: str = "cpu",
        kl_type: str = "tnu",
        disabled_latent: list[int] = None,
        disabled_latent_values: list[float] = None,
    ):
        super().__init__()
        if disabled_latent is None:
            disabled_latent = []
            disabled_latent_values = []
        else:
            assert len(disabled_latent) == len(disabled_latent_values)
        self.device = device
        self.latent_dim = latent_dim
        self.max_sigma = torch.tensor(max_sigma).to(device)
        self.log_max_sigma = torch.log(self.max_sigma).to(self.device)
        self.min_sigma = torch.tensor(min_sigma).to(device)
        self.log_min_sigma = torch.log(self.min_sigma).to(device)
        self.kl_type = kl_type
        self.disabled_latent = torch.tensor(disabled_latent).to(
            self.device
        )  # Disabling hotspot
        if len(self.disabled_latent):
            print(f"WARNING: disabling latent variable {self.disabled_latent}")
        self.disabled_latent_value = (
            torch.tensor(disabled_latent_values).float().to(self.device)
        )

    def change_device(self, device: str | torch.device):
        """
        A method to change all attributes to desired device
        """
        self.device = device
        self.max_sigma = self.max_sigma.to(device)
        self.log_max_sigma = self.log_max_sigma.to(self.device)
        self.min_sigma = self.min_sigma.to(device)
        self.log_min_sigma = self.log_min_sigma.to(device)
        self.disabled_latent_value.to(device)
        pass

    def get_params_from_encoder(self, encoder_output: torch.Tensor):
        """
        Transforms an encoder's output into distribution parameters
        """
        if len(encoder_output.size()) == 2:
            # input size (B x 2L)
            encoder_output = encoder_output.reshape(
                encoder_output.size(0), 2, self.latent_dim
            )
            mu = torch.sigmoid(encoder_output[:, 0, :].view(-1, self.latent_dim))
            logstd = torch.sigmoid(encoder_output[:, 1, :].view(-1, self.latent_dim))
            logstd = (
                logstd * (self.log_max_sigma - self.log_min_sigma) + self.log_min_sigma
            )
            sigma = torch.exp(logstd)
            params = torch.stack([mu, sigma], dim=2)
            return params
        raise NotImplementedError

    def reparametrize(self, encoder_output: torch.Tensor, n_samples: int = 1):
        """
        Sample the latent variables in a differentiable manner from an encoder's output.
        """
        params, _ = self.get_params_from_encoder(encoder_output)
        z = self.sample_latent_from_params(params, n_samples=n_samples)
        return z

    def sample_latent_from_params(
        self, params: torch.Tensor, n_samples: int = 1, deterministic=False
    ):
        """Sample the latent variables in a differentiable manner from
        distribution parameters.

        """
        mu = params[..., 0]
        if deterministic:
            return mu.unsqueeze(-1)
        sigma = params[..., 1]
        tn_dist = TruncatedNormal(
            loc=mu, scale=sigma, low=torch.zeros_like(mu), high=torch.ones_like(mu)
        )
        z = tn_dist.rsample(torch.Size([n_samples])).permute(1, 2, 0)  # Batch x Latent x Sample
        if len(self.disabled_latent):
            z[..., self.disabled_latent, :] = self.disabled_latent_value
        return z

    def latent_pdf(self, params, support_sampling: float = 0.001):
        """
        Get the latent distribution pdf and its support.
        """
        mu = params[:, :, 0]
        sigma = params[:, :, 1]
        tn_dist = TruncatedNormal(
            loc=mu, scale=sigma, low=torch.zeros_like(mu), high=torch.ones_like(mu)
        )
        supports = torch.arange(0, 1, support_sampling).to(self.device)
        supports = torch_select_unsqueeze(
            supports, select_dim=0, nb_dim=len(mu.size()) + 1
        )
        pdfs = tn_dist.pdf(supports).permute(1, 2, 0)
        supports = supports.repeat(mu.unsqueeze(0).size()).permute(1, 2, 0)
        return pdfs, supports

    def mode(self, params: torch.Tensor):
        """
        Computes the distribution mode (from marginals)
        """
        mu = params[:, :, 0]
        sigma = params[:, :, 1]
        tn_dist = TruncatedNormal(
            loc=mu, scale=sigma, low=torch.zeros_like(mu), high=torch.ones_like(mu)
        )
        return tn_dist.loc

    def median(self, params: torch.Tensor):
        """
        Computes the distribution median
        """
        mu = params[:, :, 0]
        sigma = params[:, :, 1]
        tn_dist = TruncatedNormal(
            loc=mu, scale=sigma, low=torch.zeros_like(mu), high=torch.ones_like(mu)
        )
        return tn_dist.icdf(torch.tensor(0.5))

    def expectation(self, params: torch.Tensor):
        """
        Computes the distribution expectation
        """
        mu = params[:, :, 0]
        sigma = params[:, :, 1]
        tn_dist = TruncatedNormal(
            loc=mu, scale=sigma, low=torch.zeros_like(mu), high=torch.ones_like(mu)
        )
        return tn_dist.mean

    def supervised_loss(self, z, params, reduction="mean", reduction_nll="sum"):
        """
        Computes a supervised loss to train the encoder.
        """
        mu = params[:, :, 0]
        sigma = params[:, :, 1]
        if len(z.size()) == 2:
            z = z.unsqueeze(2)
        nll = truncated_gaussian_nll(
            z, mu.unsqueeze(2), sigma.unsqueeze(2), reduction=reduction_nll
        )
        if reduction == "mean":
            nll = nll.mean()
        return nll

    def kl(
        self,
        params: torch.Tensor,
        params2: torch.Tensor | None = None,
        #    lai_only=False,
        lat_idx: torch.Tensor | None = None,
    ):
        """
        Computes the Kullback-Leibler divergence
        """
        if lat_idx is None:
            lat_idx = torch.tensor([])
        sigma = params[:, :, 1].squeeze()
        mu = params[:, :, 0].squeeze()
        if lat_idx.size(0) > 0:
            sigma = sigma[:, lat_idx].unsqueeze(1)
            mu = mu[:, lat_idx].unsqueeze(1)
        p_tn_dist = TruncatedNormal(
            loc=mu, scale=sigma, low=torch.zeros_like(mu), high=torch.ones_like(mu)
        )
        if self.kl_type == "tnu":
            return kl_truncated_normal_uniform(p=p_tn_dist, q=None)
        if self.kl_type == "tntn":
            assert params is not None
            sigma2 = params2[:, :, 1].squeeze()
            mu2 = params2[:, :, 0].squeeze()
            if lat_idx.size(0) > 0:
                sigma2 = sigma2[:, lat_idx].unsqueeze(1)
                mu2 = mu2[:, lat_idx].unsqueeze(1)
            q_tn_dist = TruncatedNormal(
                loc=mu2,
                scale=sigma2,
                low=torch.zeros_like(mu2),
                high=torch.ones_like(mu2),
            )
            return kl_truncated_normal_truncated_normal(p_tn_dist, q_tn_dist)
        raise NotImplementedError
