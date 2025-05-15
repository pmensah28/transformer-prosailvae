#!/usr/bin/env python3
"""
Created on Tue Nov  8 14:45:12 2022

Updated to support a transformer-based encoder.
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch

from .decoders import ProsailSimulatorDecoder
from .encoders import EncoderConfig, get_encoder
from prosailvae.latentspace import TruncatedNormalLatent
from prosailvae.loss import LossConfig, NLLLoss
from prosailvae.ProsailSimus import ProsailSimulator, SensorSimulator, PROSAILVARS
from prosailvae.simspaces import LinearVarSpace
from prosailvae.simvae import SimVAE, SimVAEConfig
from utils.utils import load_dict


def load_params(config_dir, config_file, parser=None):
    """
    Load parameter dict for prosail vae and training, with default options.
    Now includes defaults for transformer hyperparameters if needed.
    """
    params = load_dict(config_dir + config_file)
    if params["supervised"]:
        params["simulated_dataset"] = True

    if "load_model" not in params:
        params["load_model"] = False
    if "vae_load_dir_path" not in params:
        params["vae_load_dir_path"] = None
    else:
        params["vae_load_dir_path"] = (
            f"{Path(__file__).parent.parent}"
            f"/{params['vae_load_dir_path']}"
        )
    if "lr_recompute_mode" not in params:
        params["lr_recompute_mode"] = False
    if "init_model" not in params:
        params["init_model"] = False
    if parser is not None:
        params["k_fold"] = parser.n_xp
        params["n_fold"] = parser.n_fold if params["k_fold"] > 1 else None

    # === Default layer/conv parameters (RNN/CNN) ===
    if "layer_sizes" not in params:
        params["layer_sizes"] = [512, 512]
    if "kernel_sizes" not in params:
        params["kernel_sizes"] = [3, 3]
    if "first_layer_kernel" not in params:
        params["first_layer_kernel"] = 3
    if "first_layer_size" not in params:
        params["first_layer_size"] = 128
    if "block_layer_sizes" not in params:
        params["block_layer_sizes"] = [128, 128]
    if "block_layer_depths" not in params:
        params["block_layer_depths"] = [2, 2]
    if "block_kernel_sizes" not in params:
        params["block_kernel_sizes"] = [3, 1]
    if "block_n" not in params:
        params["block_n"] = [1, 3]

    if "supervised_kl" not in params:
        params["supervised_kl"] = False
    params["vae_save_file_path"] = None
    if "supervised_config_file" not in params:
        params["supervised_config_file"] = None
    if "supervised_weight_file" not in params:
        params["supervised_weight_file"] = None
    if "disabled_latent" not in params:
        params["disabled_latent"] = []
    if "disabled_latent_values" not in params:
        params["disabled_latent_values"] = []
    if "cycle_training" not in params:
        params["cycle_training"] = False
    if "R_down" not in params:
        params["R_down"] = 1
    if "n_init_models" not in params:
        params["n_init_models"] = 10
    if "n_init_epochs" not in params:
        params["n_init_epochs"] = 10
    if "init_lr" not in params:
        params["init_lr"] = 5e-4
    if "break_init_at_rec_loss" not in params:
        params["break_init_at_rec_loss"] = None
    if "rec_bands_loss_coeffs" not in params:
        params["rec_bands_loss_coeffs"] = None
    if "deterministic" not in params:
        params["deterministic"] = False
    if "accum_iter" not in params:
        params["accum_iter"] = 1
    if "beta_cyclical" not in params:
        params["beta_cyclical"] = 0
    if "lat_loss_type" not in params:
        params["lat_loss_type"] = ""
    if "lrs_threshold" not in params:
        params["lrs_threshold"] = 5e-3
    if "validation_at_every_epoch" not in params:
        params["validation_at_every_epoch"] = None
    if "prosail_vars_dist_type" not in params:
        params["prosail_vars_dist_type"] = "legacy"
    if "lat_idx" not in params:
        params["lat_idx"] = []
    if "prospect_version" not in params:
        params["prospect_version"] = "5"
    if "frm4veg_data_dir" not in params:
        params["frm4veg_data_dir"] = "/work/scratch/zerahy/prosailvae/data/frm4veg_validation"
    if "frm4veg_2021_data_dir" not in params:
        params["frm4veg_2021_data_dir"] = "/work/scratch/zerahy/prosailvae/data/frm4veg_2021_validation"
    if "belsar_dir" not in params:
        params["belsar_dir"] = "/work/scratch/zerahy/prosailvae/data/belSAR_validation"
    if "cyclical_data_dir" not in params:
        params["cyclical_data_dir"] = "/work/scratch/zerahy/prosailvae/data/projected_data"

    # === Defaults for Transformer-based encoders ===
    # Only relevant if encoder_type = "transformer"
    if "n_transformer_layers" not in params:
        params["n_transformer_layers"] = 2
    if "d_model" not in params:
        params["d_model"] = 64
    if "n_heads" not in params:
        params["n_heads"] = 4
    if "dropout" not in params:
        params["dropout"] = 0.1

    return params


@dataclass
class ProsailVAEConfig:
    """
    Dataclass to hold all of PROSAIL_VAE configurations
    """

    encoder_config: EncoderConfig
    loss_config: LossConfig
    rsr_dir: Path | str
    vae_load_file_path: str
    vae_save_file_path: str
    spatial_mode: bool = False
    load_vae: bool = False
    apply_norm_rec: bool = True
    inference_mode: bool = False
    prosail_bands: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
    )
    disabled_latent: list[int] = field(default_factory=lambda: [])
    disabled_latent_values: list[int] = field(default_factory=lambda: [])
    R_down: int = 1
    deterministic: bool = False
    prosail_vars_dist_type: str = "legacy"
    prospect_version: str = "5"
    sensor_type: str = "sentinel2"  # Can be "sentinel2" or "enmap"

    def __post_init__(self):
        self.rsr_dir = Path(self.rsr_dir)


def get_prosail_vae_config(
    params, bands, io_coeffs, inference_mode, prosail_bands, rsr_dir, lai_ccc_mode=False
):
    """
    Get ProsailVAEConfig from params dict.
    Incorporates the newly added transformer fields if encoder_type = "transformer".
    """
    n_idx = io_coeffs.idx.loc.size(0) if io_coeffs.idx.loc is not None else 0

    encoder_config = EncoderConfig(
        encoder_type=params["encoder_type"],
        input_size=len(bands) + 3 + n_idx,  # reflectances + angles + optional idx
        output_size=len(PROSAILVARS) if not lai_ccc_mode else 2,
        io_coeffs=io_coeffs,
        bands=bands,
        last_activation=None,
        n_latent_params=2,
        layer_sizes=params["layer_sizes"],
        kernel_sizes=params["kernel_sizes"],
        padding="valid",
        first_layer_kernel=params["first_layer_kernel"],
        first_layer_size=params["first_layer_size"],
        block_layer_sizes=params["block_layer_sizes"],
        block_layer_depths=params["block_layer_depths"],
        block_kernel_sizes=params["block_kernel_sizes"],
        block_n=params["block_n"],
        disable_s2_r_idx=(n_idx == 0),
        # === Transformer fields ===
        n_transformer_layers=params["n_transformer_layers"],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        dropout=params["dropout"],
    )

    # Check if the chosen encoder is "spatial" (CNN-like)
    spatial_encoder = get_encoder(encoder_config).get_spatial_encoding()
    if spatial_encoder:
        params["loss_type"] = "spatial_nll"

    # Reconstruction band coefficients
    if params["rec_bands_loss_coeffs"] is not None:
        assert len(bands) >= len(params["rec_bands_loss_coeffs"])
        reconstruction_bands_coeffs = params["rec_bands_loss_coeffs"]
    else:
        reconstruction_bands_coeffs = None

    loss_config = LossConfig(
        supervised=params["supervised"],
        beta_index=params["beta_index"],
        beta_kl=params["beta_kl"],
        beta_cyclical=params["beta_cyclical"],
        loss_type=params["loss_type"],
        lat_loss_type=params["lat_loss_type"],
        reconstruction_bands_coeffs=reconstruction_bands_coeffs,
        lat_idx=torch.tensor(params["lat_idx"]).int(),
    )

    # Get sensor type, default to sentinel2 if not specified
    sensor_type = params.get("sensor_type", "sentinel2")

    return ProsailVAEConfig(
        encoder_config=encoder_config,
        loss_config=loss_config,
        rsr_dir=rsr_dir,
        vae_load_file_path=params["vae_load_file_path"],
        vae_save_file_path=params["vae_save_file_path"],
        load_vae=params["load_model"],
        apply_norm_rec=params["apply_norm_rec"],
        inference_mode=inference_mode,
        prosail_bands=prosail_bands,
        disabled_latent=params["disabled_latent"],
        disabled_latent_values=params["disabled_latent_values"],
        R_down=params["R_down"],
        deterministic=params["deterministic"],
        prosail_vars_dist_type=params["prosail_vars_dist_type"],
        prospect_version=params["prospect_version"],
        sensor_type=sensor_type,
    )


def get_prosail_vae(
    pv_config: ProsailVAEConfig,
    device: torch.device | str = "cpu",
    logger_name: str = "",
    hyper_prior: SimVAE | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    load_simulator=True,
    freeze_weights=False,
):
    """
    Initializes an instance of prosail_vae.
    """
    # Instantiate the encoder
    encoder = get_encoder(pv_config.encoder_config, device="cpu")

    # Decide KL type depending on hyper_prior usage
    if hyper_prior is not None:
        kl_type = "tntn"
    else:
        kl_type = "tnu"

    lat_space = TruncatedNormalLatent(
        device="cpu",
        latent_dim=pv_config.encoder_config.output_size,
        kl_type=kl_type,
        disabled_latent=pv_config.disabled_latent,
        disabled_latent_values=pv_config.disabled_latent_values,
    )

    reconstruction_loss = NLLLoss(
        loss_type=pv_config.loss_config.loss_type,
        feature_indexes=pv_config.loss_config.reconstruction_bands_coeffs,
    )

    prosail_var_space = LinearVarSpace(
        latent_dim=pv_config.encoder_config.output_size,
        device="cpu",
        var_bounds_type=pv_config.prosail_vars_dist_type,
    )

    psimulator = ProsailSimulator(
        device="cpu",
        R_down=pv_config.R_down,
        prospect_version=pv_config.prospect_version,
    )

    if load_simulator:
        # Check if we should use the EnMAP sensor simulator
        if hasattr(pv_config, "sensor_type") and pv_config.sensor_type == "enmap":
            from prosailvae.ProsailSimus import EnMapSensorSimulator

            ssimulator = EnMapSensorSimulator(
                rsr_dir=pv_config.rsr_dir,
                rsr_file="enmap.rsr",
                device="cpu",
                bands_loc=pv_config.encoder_config.io_coeffs.bands.loc,
                bands_scale=pv_config.encoder_config.io_coeffs.bands.scale,
                idx_loc=pv_config.encoder_config.io_coeffs.idx.loc,
                idx_scale=pv_config.encoder_config.io_coeffs.idx.scale,
                apply_norm=pv_config.apply_norm_rec,
                R_down=pv_config.R_down,
            )
        else:
            # Use standard SensorSimulator for Sentinel-2
            ssimulator = SensorSimulator(
                pv_config.rsr_dir / "sentinel2.rsr",
                device="cpu",
                bands_loc=pv_config.encoder_config.io_coeffs.bands.loc,
                bands_scale=pv_config.encoder_config.io_coeffs.bands.scale,
                idx_loc=pv_config.encoder_config.io_coeffs.idx.loc,
                idx_scale=pv_config.encoder_config.io_coeffs.idx.scale,
                apply_norm=pv_config.apply_norm_rec,
                bands=pv_config.prosail_bands,
                R_down=pv_config.R_down,
            )
    else:
        # Minimal sensor simulator if not loading
        ssimulator = SensorSimulator(
            pv_config.rsr_dir / "sentinel2.rsr",
            device="cpu",
            bands_loc=None,
            bands_scale=None,
            apply_norm=pv_config.apply_norm_rec,
            bands=pv_config.prosail_bands,
            R_down=pv_config.R_down,
        )

    # Physically informed decoder: PROSAIL + sensor simulator
    decoder = ProsailSimulatorDecoder(
        prosailsimulator=psimulator,
        ssimulator=ssimulator,
        loss_type=pv_config.loss_config.loss_type,
    )

    # Build the full SimVAE
    prosail_vae = SimVAE(
        SimVAEConfig(
            encoder=encoder,
            decoder=decoder,
            lat_space=lat_space,
            sim_space=prosail_var_space,
            deterministic=pv_config.deterministic,
            reconstruction_loss=reconstruction_loss,
            supervised=pv_config.loss_config.supervised,
            device="cpu",
            beta_kl=pv_config.loss_config.beta_kl,
            beta_index=pv_config.loss_config.beta_index,
            beta_cyclical=pv_config.loss_config.beta_cyclical,
            logger_name=logger_name,
            inference_mode=pv_config.inference_mode,
            lat_nll=pv_config.loss_config.lat_loss_type,
            lat_idx=pv_config.loss_config.lat_idx,
        )
    )

    prosail_vae.set_hyper_prior(hyper_prior)

    # Optionally load existing VAE weights
    if pv_config.load_vae and pv_config.vae_load_file_path is not None:
        _, _ = prosail_vae.load_ae(pv_config.vae_load_file_path, optimizer)

    # Move to desired device (e.g. GPU)
    prosail_vae.change_device(device)

    if freeze_weights:
        prosail_vae.freeze_weigths()

    return prosail_vae


def load_prosail_vae_with_hyperprior(
    logger_name: str,
    pv_config: ProsailVAEConfig,
    pv_config_hyper: ProsailVAEConfig | None = None,
):
    """
    Loads prosail vae with or without initializing weight, with optional hyperprior.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyper_prior = None
    if pv_config_hyper is not None:
        hyper_prior = get_prosail_vae(
            pv_config_hyper,
            device=device,
            logger_name=logger_name,
            load_simulator=False,
            freeze_weights=True,
        )

    prosail_vae = get_prosail_vae(
        pv_config, device=device, logger_name=logger_name, hyper_prior=hyper_prior
    )
    return prosail_vae
