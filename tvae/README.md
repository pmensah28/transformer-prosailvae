# Physics Informed Transformer-VAE for Biophysical Parameter Estimation: PROSAIL model inversion in Sentinel‑2 imagery

Transformer-VAE combines the power of transformer architecture with physics-based radiative transfer modeling for accurate biophysical parameter retrieval from Sentinel-2 satellite imagery. The model leverages self-attention mechanisms to effectively capture complex relationships in spectral data while maintaining physical consistency through the PROSAIL model to significantly improve feature extraction and parameter estimation.

## Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Workflow](#workflow)
  - [Data Generation](#data-generation)
  - [Model Training](#model-training)
  - [Validation and Evaluation](#validation-and-evaluation)
- [Results](#results)
- [References](#references)
- [Acknowledgments](#acknowledgments)

## Architecture

<p align="center">
<img src="assets/architecture/training.png" width="80%" />
<br>
<em>Figure 1: Transformer-PROSAILVAE training architecture. The transformer encoder maps spectral reflectance to a latent space while the PROSAIL model serves as a fixed physics-based decoder.</em>
</p>


<p align="center">
<img src="assets/architecture/inference.png" width="80%" />
<br>
<em>Figure 2: Transformer-PROSAILVAE inference architecture. During inference, the model takes Sentinel-2 spectral reflectance as input and generates predictions of biophysical parameters (LAI, CCC) with associated uncertainty estimates.</em>
</p>

This architecture enables the model to discover intricate patterns in spectral signatures that correspond to vegetation properties, significantly outperforming traditional inversion methods and simpler neural network approaches.

## Installation


### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/instadeepai/InstaGeo-Hyperspectral.git
   cd tvae
   ```

2. Create and activate the Conda environment:
   ```bash
   conda create -n tvae python=3.11 -y
   conda activate tvae
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install specific packages:
   ```bash
   pip install git+https://src.koda.cnrs.fr/mmdc/prosailpython.git
   pip install git+https://src.koda.cnrs.fr/mmdc/sensorsio.git
   pip install git+https://src.koda.cnrs.fr/mmdc/torchutils.git
   pip install git+https://src.koda.cnrs.fr/mmdc/mmdc-singledate.git
   ```

5. Install the project in editable mode:
   ```bash
   pip install -e .
   ```

## Workflow

### Data Generation

The first step in the Transformer-VAE workflow involves synthetic data generation using the PROSAIL radiative transfer model, which simulates realistic spectral signatures based on biophysical parameters:

```bash
bash scripts/generate_prosail_dataset.sh
```

By default, this script executes the following sequence:

1. Generates a test dataset (10,000 samples) with diverse vegetation parameters
2. Creates a validation dataset (10,000 samples) with appropriate parameter distributions
3. Produces a training dataset in batches (500,000 samples total) covering the full parameter space
4. Combines batches into a single dataset with computed normalization factors for stable training

The generated datasets include spectral reflectance values across Sentinel-2 bands along with corresponding vegetation parameters such as Leaf Area Index (LAI), chlorophyll content, and canopy structure variables.

### Model Training

To train the Transformer-PROSAILVAE model with default parameters:

```bash
bash scripts/run_transformer_training.sh
```

During training, the transformer encoder progressively learns to map spectral reflectance patterns to a latent space that encodes biophysical parameters, while the PROSAIL model serves as a fixed physics-based decoder that enforces physical consistency. This hybrid approach combines the flexibility of deep learning with the interpretability of process-based models.

### Validation and Evaluation

Before running the validation scripts, you must first process the Sentinel-2 tiles for each validation dataset:

#### 1. Preprocessing Validation Data

##### FRM4VEG Data Processing
```bash
python validation/frm4veg_validation.py --process_all
```

##### BelSAR Data Processing
```bash
python validation/belsar_validation.py --data_dir data/belsar_validation --plot
```

The validation datasets use the following Sentinel-2 products:

- **FRM4VEG S2 Products**:
  - SENTINEL2A_20180613-110957-425_L2A_T30SWJ_D_V1-8 (Barrax 2018)
  - SENTINEL2B_20210722-111020-007_L2A_T30SWJ_C_V3-0 (Barrax 2021)
  - SENTINEL2A_20180629-112645-306_L2A_T30UXC_C_V4-0 (Wytham 2018)

- **BelSAR S2 Products**:
  - SENTINEL2A_20180518-104024-461_L2A_T31UFS_C_V2-2
  - SENTINEL2A_20180528-104613-414_L2A_T31UFS_C_V2-2
  - SENTINEL2A_20180620-105211-086_L2A_T31UFS_C_V2-2
  - SENTINEL2A_20180727-104023-458_L2A_T31UFS_C_V2-2
  - SENTINEL2B_20180715-105300-591_L2A_T31UFS_D_V1-8
  - SENTINEL2B_20180804-105022-459_L2A_T31UFS_C_V2-2

#### 2. In-situ Data Sources

- Sentinel-2 products can be downloaded from [GEODES Portal](https://geodes-portal.cnes.fr/)
- FRM4VEG in-situ field measurements are available for download at [FRM4VEG Website](https://frm4veg.org/)
- BelSAR dataset is available upon request at [ESA's BelSAR Campaign Page](https://earth.esa.int/eogateway/campaigns/belsar-campaign-2018)

#### 3. Running Validation

The project includes a comprehensive validation framework supporting multiple datasets and evaluation metrics:

```bash
bash scripts/run_validation.sh [options]
```

Key validation options include:

- `--model-path`: Path to the model checkpoint for evaluation
- `--data-dir`: Base directory containing validation data
- `--output-dir`: Directory to save validation results and visualizations
- `--device`: Computing device to use (cuda or cpu)
- `--method`: Interpolation method for prediction (closest, linear, cubic)
- `--mode`: Prediction mode for ensemble outputs (sim_tg_mean, lat_mode)
- `--reconstruction`: Flag to save reconstruction errors for model diagnostics

Example validation command:

```bash
# Run comprehensive validation with detailed results
bash scripts/run_validation.sh --datasets all --combine-results --export-plots --export-detailed --comparison-plot --reconstruction
```

The validation process evaluates the model against ground truth measurements from multiple field campaigns, computing various performance metrics and generating visualizations for analysis.

## Results

<p align="center">
<img src="assets/plots/combined_lai_validation_20250427_202455.png" width="48%" /> <img src="assets/plots/combined_ccc_validation_20250427_202455.png" width="48%" />
<br>
<em>Figure 3: Validation results showing model performance across different field sites. Left: Leaf Area Index (LAI) prediction accuracy. Right: Canopy Chlorophyll Content (CCC) prediction accuracy.</em>
</p>

### Quantitative Performance Evaluation

The Transformer-VAE model demonstrates superior performance across diverse vegetation types and environmental conditions, as evidenced by comprehensive validation against ground truth measurements from multiple field campaigns.

#### Transformer-VAE Performance on In-situ Data

| Metric | BelSAR | Barrax (2018) | Barrax (2021) | Wytham | All sites |
|--------|--------|---------------|---------------|--------|-----------|
| **LAI RMSE** | 1.03 | 0.72 | 0.63 | 1.53 | 0.99 |
| **LAI MPIW** | 6.04 | 1.52 | 0.72 | 0.96 | 5.15 |
| **LAI PICP** | 0.92 | 0.67 | 0.37 | 0.10 | 0.95 |
| **LAI R²** | 0.84 | 0.91 | 0.89 | 0.76 | 0.85 |
| **CCC RMSE** | - | 63.43 | 41.56 | 134.75 | 76.56 |
| **CCC MPIW** | - | 56.09 | 30.11 | 77.89 | 310.55 |
| **CCC PICP** | - | 0.21 | 0.19 | 0.16 | 0.88 |

*RMSE: Root Mean Square Error (lower is better), MPIW: Mean Prediction Interval Width (narrower indicates higher confidence), PICP: Prediction Interval Coverage Probability (closer to nominal coverage is better), R²: Coefficient of determination (higher is better)*

These results demonstrate the Transformer-VAE's ability to accurately retrieve biophysical parameters across diverse ecosystems and vegetation types. The model achieves particularly strong performance for agricultural sites (BelSAR, Barrax) while maintaining reasonable accuracy in more complex forest environments (Wytham).

## References

This work builds upon several important research contributions:

1. Zérah, Yoël, Silvia Valero, and Jordi Inglada. "Physics-constrained deep learning for biophysical parameter retrieval from Sentinel-2 images: Inversion of the PROSAIL model." Remote Sensing of Environment 312 (2024): 114309.

2. Jacquemoud, S., Verhoef, W., Baret, F., Bacour, C., Zarco-Tejada, P. J., Asner, G. P., François, C., & Ustin, S. L. (2009). PROSPECT+SAIL models: A review of use for vegetation characterization. Remote Sensing of Environment, 113, S56-S66.

3. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.

4. Berger, K., Verrelst, J., Féret, J. B., Wang, Z., Wocher, M., Strathmann, M., Danner, M., Mauser, W., & Hank, T. (2020). Crop nitrogen monitoring: Recent progress and principal developments in the context of imaging spectroscopy missions. Remote Sensing of Environment, 242, 111758.

## Acknowledgments

This project takes inspiration from the work of Zérah, Yoël, Silvia Valero, and Jordi Inglada. "Physics-constrained deep learning for biophysical parameter retrieval from Sentinel-2 images: Inversion of the PROSAIL model." We built upon their foundation, with significant extensions to incorporate transformer architectures.

Special thanks to the developers of the libraries used in this project:
- `prosailpython`: PROSAIL radiative transfer model implementation in pytorch
- `sensorsio`: Processing of field data and sensor spectral response functions
- `torchutils`: Utilities for PyTorch model implementation
- `mmdc-singledate`: Tools for single-date Earth observation data processing

These libraries were instrumental in facilitating the efficient processing of field validation data.