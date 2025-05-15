import torch
import numpy as np


def NDVI(s2_r, eps=torch.tensor(1e-7), bands_dim=1):
    B4 = s2_r.select(bands_dim, 2).unsqueeze(bands_dim)
    B8 = s2_r.select(bands_dim, 6).unsqueeze(bands_dim)
    num = B8 - B4
    denom = B8 + B4
    non_zero_denom_idx = denom.abs() > eps
    ndvi = -torch.ones_like(B4)
    ndvi[non_zero_denom_idx] = num[non_zero_denom_idx] / denom[non_zero_denom_idx]
    return torch.clamp(ndvi, min=-1, max=1)


def mNDVI750(s2_r, eps=torch.tensor(1e-7), bands_dim=1):
    B2 = s2_r.select(bands_dim, 0).unsqueeze(bands_dim)
    B5 = s2_r.select(bands_dim, 3).unsqueeze(bands_dim)
    B6 = s2_r.select(bands_dim, 4).unsqueeze(bands_dim)
    denom = B6 + B5 - 2 * B2
    num = B6 - B5
    mndvi750 = -torch.ones_like(B6)
    non_zero_denom_idx = denom.abs() > eps
    mndvi750[non_zero_denom_idx] = num[non_zero_denom_idx] / denom[non_zero_denom_idx]
    return torch.clamp(mndvi750, min=-1, max=1)


def CRI2(s2_r, eps=torch.tensor(1e-7), bands_dim=1):
    B2 = s2_r.select(bands_dim, 0).unsqueeze(bands_dim)
    B5 = s2_r.select(bands_dim, 3).unsqueeze(bands_dim)
    cri2 = torch.zeros_like(B2)
    b2_and_b5_sup_0_idx = torch.logical_and(B2 > eps, B5 >= B2)
    cri2[b2_and_b5_sup_0_idx] = 1 / (B2[b2_and_b5_sup_0_idx]) - 1 / ( 
        B5[b2_and_b5_sup_0_idx]
    )
    return torch.clamp(cri2, max=20)


def NDII(s2_r, eps=torch.tensor(1e-7), bands_dim=1):
    B8 = s2_r.select(bands_dim, 6).unsqueeze(bands_dim)
    B11 = s2_r.select(bands_dim, 8).unsqueeze(bands_dim)
    num = B8 - B11
    denom = B8 + B11
    non_zero_denom_idx = denom.abs() > eps
    ndii = -torch.ones_like(B8)
    ndii[non_zero_denom_idx] = num[non_zero_denom_idx] / denom[non_zero_denom_idx]
    return torch.clamp(ndii, min=-1, max=1)


def ND_lma(s2_r, eps=torch.tensor(1e-7), bands_dim=1):
    B11 = s2_r.select(bands_dim, 8).unsqueeze(bands_dim)
    B12 = s2_r.select(bands_dim, 9).unsqueeze(bands_dim)
    num = B12 - B11
    denom = B12 + B11
    non_zero_denom_idx = denom.abs() > eps
    nd_lma = -torch.ones_like(B12)
    nd_lma[non_zero_denom_idx] = num[non_zero_denom_idx] / denom[non_zero_denom_idx]
    return torch.clamp(nd_lma, min=-1, max=1)


def LAI_savi(s2_r, eps=torch.tensor(1e-7), bands_dim=1):
    B4 = s2_r.select(bands_dim, 2).unsqueeze(bands_dim)
    B8 = s2_r.select(bands_dim, 6).unsqueeze(bands_dim)
    return -torch.log(
        torch.abs(
            torch.tensor(0.371)
            + torch.tensor(1.5) * (B8 - B4) / (B8 + B4 + torch.tensor(0.5))
        )
        + eps
    ) / torch.tensor(2.4)


INDEX_DICT = {
    "NDVI": NDVI,
    "NDII": NDII,
    "ND_lma": ND_lma,
    "LAI_savi": LAI_savi,
}  # } #"mNDVI750":mNDVI750, "CRI2":CRI2,


def get_spectral_idx(s2_r, eps=torch.tensor(1e-4), bands_dim=1, index_dict=INDEX_DICT):
    spectral_idx = []
    for idx_name, idx_fn in index_dict.items():
        idx = idx_fn(torch.clamp(s2_r, min=0.0, max=1.0), eps=eps, bands_dim=bands_dim)
        if not s2_r.isnan().any() or s2_r.isinf().any():
            if idx.isnan().any() or idx.isinf().any():
                print(
                    s2_r[
                        torch.logical_or(
                            idx.isnan().tile(
                                [
                                    (s2_r.size(bands_dim) if i == bands_dim else 1)
                                    for i in range(len(s2_r.size()))
                                ]
                            ),
                            idx.isinf().tile(
                                [
                                    (s2_r.size(bands_dim) if i == bands_dim else 1)
                                    for i in range(len(s2_r.size()))
                                ]
                            ),
                        )
                    ]
                )
                raise ValueError(
                    f"{idx_name} has NaN {idx.isnan().int().sum()} or infinite {idx.isinf().int().sum()} values!"
                )
        spectral_idx.append(idx)
    return torch.cat(spectral_idx, axis=bands_dim)

# Define wavelength regions for EnMAP
ENMAP_REGIONS = {
    'blue': (440, 510),       # Blue region
    'green': (510, 580),      # Green region
    'red': (630, 690),        # Red region
    'red_edge': (690, 730),   # Red edge
    'nir': (760, 900),        # Near infrared
    'swir1': (1550, 1750),    # Short-wave infrared 1
    'swir2': (2000, 2300),    # Short-wave infrared 2
}

def get_enmap_band_averages(reflectance, wavelengths, bands_dim=1):
    """
    Get average reflectance values for predefined spectral regions.
    
    Args:
        reflectance: Tensor of shape [batch_size, num_bands] or [num_bands]
        wavelengths: Tensor containing the wavelength for each band
        bands_dim: Dimension for bands
        
    Returns:
        Dictionary mapping region names to their average reflectance values
    """
    # Add batch dimension if needed
    if len(reflectance.shape) == 1:
        reflectance = reflectance.unsqueeze(0)
        
    batch_size = reflectance.shape[0]
    band_averages = {}
    device = reflectance.device
    
    # Ensure wavelengths is on the same device as reflectance
    wavelengths = wavelengths.to(device)
    
    for region_name, (start_wl, end_wl) in ENMAP_REGIONS.items():
        # Find bands within this wavelength range
        mask = (wavelengths >= start_wl) & (wavelengths <= end_wl)
        if torch.any(mask):
            # Get indices where mask is True
            indices = torch.nonzero(mask).squeeze()
            
            # Check if indices are within the reflectance dimension
            max_index = reflectance.shape[bands_dim] - 1
            valid_indices = indices[indices <= max_index]
            
            if len(valid_indices) > 0:
                # Calculate average reflectance in this region
                if bands_dim == 1:
                    region_refl = reflectance[:, valid_indices].mean(dim=bands_dim)
                elif bands_dim == 0:
                    region_refl = reflectance[valid_indices, :].mean(dim=bands_dim)
                else:
                    # Handle other dimensions if needed
                    region_refl = torch.index_select(reflectance, bands_dim, valid_indices).mean(dim=bands_dim)
                
                band_averages[region_name] = region_refl
            else:
                # If no valid bands in this region, use a placeholder value
                band_averages[region_name] = torch.zeros(batch_size, device=device)
        else:
            # If no bands in this region, use a placeholder value
            band_averages[region_name] = torch.zeros(batch_size, device=device)
            
    return band_averages

# EnMAP Indices
def enmap_NDVI(refl_dict, eps=torch.tensor(1e-7)):
    """NDVI calculated from EnMAP hyperspectral regions"""
    num = refl_dict['nir'] - refl_dict['red']
    denom = refl_dict['nir'] + refl_dict['red']
    non_zero_denom_idx = denom.abs() > eps
    ndvi = -torch.ones_like(refl_dict['nir'])
    ndvi[non_zero_denom_idx] = num[non_zero_denom_idx] / denom[non_zero_denom_idx]
    return torch.clamp(ndvi, min=-1, max=1)

def enmap_EVI(refl_dict, eps=torch.tensor(1e-7)):
    """Enhanced Vegetation Index calculated from EnMAP hyperspectral regions"""
    gain = 2.5
    C1, C2 = 6.0, 7.5
    L = 1.0
    return gain * (refl_dict['nir'] - refl_dict['red']) / (refl_dict['nir'] + C1 * refl_dict['red'] - C2 * refl_dict['blue'] + L)

def enmap_NDWI(refl_dict, eps=torch.tensor(1e-7)):
    """Normalized Difference Water Index calculated from EnMAP hyperspectral regions"""
    num = refl_dict['nir'] - refl_dict['swir1']
    denom = refl_dict['nir'] + refl_dict['swir1']
    non_zero_denom_idx = denom.abs() > eps
    ndwi = -torch.ones_like(refl_dict['nir'])
    ndwi[non_zero_denom_idx] = num[non_zero_denom_idx] / denom[non_zero_denom_idx]
    return torch.clamp(ndwi, min=-1, max=1)

def enmap_RED_EDGE_NDVI(refl_dict, eps=torch.tensor(1e-7)):
    """Red Edge NDVI calculated from EnMAP hyperspectral regions"""
    num = refl_dict['nir'] - refl_dict['red_edge']
    denom = refl_dict['nir'] + refl_dict['red_edge']
    non_zero_denom_idx = denom.abs() > eps
    red_edge_ndvi = -torch.ones_like(refl_dict['nir'])
    red_edge_ndvi[non_zero_denom_idx] = num[non_zero_denom_idx] / denom[non_zero_denom_idx]
    return torch.clamp(red_edge_ndvi, min=-1, max=1)

def enmap_MSI(refl_dict, eps=torch.tensor(1e-7)):
    """Moisture Stress Index calculated from EnMAP hyperspectral regions"""
    non_zero_idx = refl_dict['nir'] > eps
    msi = torch.zeros_like(refl_dict['nir'])
    msi[non_zero_idx] = refl_dict['swir1'][non_zero_idx] / refl_dict['nir'][non_zero_idx]
    return torch.clamp(msi, min=0, max=30)

def enmap_NBR(refl_dict, eps=torch.tensor(1e-7)):
    """Normalized Burn Ratio calculated from EnMAP hyperspectral regions"""
    num = refl_dict['nir'] - refl_dict['swir2']
    denom = refl_dict['nir'] + refl_dict['swir2']
    non_zero_denom_idx = denom.abs() > eps
    nbr = -torch.ones_like(refl_dict['nir'])
    nbr[non_zero_denom_idx] = num[non_zero_denom_idx] / denom[non_zero_denom_idx]
    return torch.clamp(nbr, min=-1, max=1)

def enmap_CRI1(refl_dict, eps=torch.tensor(1e-7)):
    """Carotenoid Reflectance Index 1 calculated from EnMAP hyperspectral regions"""
    blue_nz = refl_dict['blue'] > eps
    green_nz = refl_dict['green'] > eps
    
    # Get masks where both are non-zero
    mask = blue_nz & green_nz
    
    cri1 = torch.zeros_like(refl_dict['blue'])
    cri1[mask] = (1 / refl_dict['blue'][mask]) - (1 / refl_dict['green'][mask])
    
    return torch.clamp(cri1, min=0, max=20)

def enmap_CARI(refl_dict, eps=torch.tensor(1e-7)):
    """Chlorophyll Absorption Ratio Index calculated from EnMAP hyperspectral regions"""
    return (refl_dict['red_edge'] / refl_dict['red']) * torch.sqrt((refl_dict['red_edge'] - refl_dict['green'])**2)

# Dictionary of EnMAP indices
ENMAP_INDEX_DICT = {
    "NDVI": enmap_NDVI,
    "EVI": enmap_EVI,
    "NDWI": enmap_NDWI,
    "RED_EDGE_NDVI": enmap_RED_EDGE_NDVI,
    "MSI": enmap_MSI,
    "NBR": enmap_NBR,
    "CRI1": enmap_CRI1,
    "CARI": enmap_CARI,
}

def get_enmap_spectral_idx(reflectance, wavelengths, eps=torch.tensor(1e-4), bands_dim=1):
    """
    Calculate hyperspectral indices for EnMAP data.
    
    Args:
        reflectance: Tensor of shape [batch_size, num_bands] or [num_bands]
        wavelengths: Tensor containing the wavelength for each band
        eps: Small epsilon value to avoid division by zero
        bands_dim: Dimension for bands
        
    Returns:
        Tensor of shape [batch_size, num_indices] containing the calculated indices
    """
    # Add batch dimension if needed
    if len(reflectance.shape) == 1:
        reflectance = reflectance.unsqueeze(0)
    
    # Calculate regional averages
    band_averages = get_enmap_band_averages(
        torch.clamp(reflectance, min=0.0, max=1.0), 
        wavelengths, 
        bands_dim=bands_dim
    )
    
    # Calculate each index
    indices_list = []
    for idx_name, idx_fn in ENMAP_INDEX_DICT.items():
        try:
            idx = idx_fn(band_averages, eps=eps)
            if idx.isnan().any() or idx.isinf().any():
                # Replace NaN and Inf values with zeros
                idx = torch.nan_to_num(idx, nan=0.0, posinf=0.0, neginf=0.0)
            indices_list.append(idx.unsqueeze(bands_dim))
        except Exception as e:
            # Use zeros if calculation fails
            batch_size = reflectance.shape[0]
            indices_list.append(torch.zeros(batch_size, 1, device=reflectance.device))
    
    # Stack all indices
    return torch.cat(indices_list, dim=bands_dim)
