import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

def create_enmap_rsr_from_excel(excel_path, sentinel2_rsr_path, output_dir='data'):
    """Create EnMAP spectral response function from the Excel file and
    save it in the same format as the Sentinel-2 RSR file"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read both sheets from the EnMAP Excel file
    df_vnir = pd.read_excel(excel_path, sheet_name='VNIR')
    df_swir = pd.read_excel(excel_path, sheet_name='SWIR')
    
    print(f"VNIR sheet has {len(df_vnir)} rows and {len(df_vnir.columns)} columns")
    print(f"SWIR sheet has {len(df_swir)} rows and {len(df_swir.columns)} columns")
    
    # Print column headers to help identify the right columns
    print("VNIR columns:", df_vnir.columns.tolist())
    print("SWIR columns:", df_swir.columns.tolist())
    
    # Identify the center wavelength and FWHM columns
    # Based on the screenshot, these are "CW (nm)" and "FWHM (nm)"
    cw_col_vnir = "CW (nm)"
    fwhm_col_vnir = "FWHM (nm)"
    
    cw_col_swir = "CW (nm)"
    fwhm_col_swir = "FWHM (nm)"
    
    # Get the center wavelengths and FWHMs
    vnir_wavelengths = df_vnir[cw_col_vnir].values
    vnir_fwhm = df_vnir[fwhm_col_vnir].values
    
    swir_wavelengths = df_swir[cw_col_swir].values
    swir_fwhm = df_swir[fwhm_col_swir].values
    
    # Combine the wavelengths and FWHMs
    all_wavelengths = np.concatenate([vnir_wavelengths, swir_wavelengths])
    all_fwhm = np.concatenate([vnir_fwhm, swir_fwhm])
    
    # Sort them by wavelength
    sort_idx = np.argsort(all_wavelengths)
    all_wavelengths = all_wavelengths[sort_idx]
    all_fwhm = all_fwhm[sort_idx]
    
    print(f"Total number of EnMAP bands: {len(all_wavelengths)}")
    
    # Define the wavelength range for the RSR file (same as Sentinel-2)
    # Usually from 350nm to 2500nm with 1nm resolution
    wavelength_range_nm = np.arange(350, 2501, 1)  # in nm
    wavelength_range_um = wavelength_range_nm / 1000.0  # convert to µm
    
    # Load the Sentinel-2 RSR file to get the solar spectrum
    s2_rsr = np.loadtxt(sentinel2_rsr_path)
    solar_spectrum = s2_rsr[:, 1]  # Second column contains solar spectrum
    
    # Create the EnMAP RSR matrix
    # First column: wavelength in µm
    # Second column: solar spectrum
    # Remaining columns: EnMAP band responses
    n_bands = len(all_wavelengths)
    n_wavelengths = len(wavelength_range_nm)
    
    # Initialize the RSR matrix with zeros
    enmap_rsr = np.zeros((n_wavelengths, n_bands + 2))
    
    # Set the first column to the wavelength range in µm
    enmap_rsr[:, 0] = wavelength_range_um
    
    # Check for shape mismatch between solar spectrum and wavelength range
    if len(solar_spectrum) < n_wavelengths:
        print(f"Extending solar spectrum from {len(solar_spectrum)} values (350-2400nm) to {n_wavelengths} values (350-2500nm)")
        
        # Create extended solar spectrum array
        extended_solar = np.zeros(n_wavelengths)
        
        # Copy the original values
        extended_solar[:len(solar_spectrum)] = solar_spectrum
        
        # Option 2: Gradual decrease for the extended wavelength range (more realistic)
        # The solar intensity typically decreases in the SWIR region
        decay_factor = np.linspace(1.0, 0.7, n_wavelengths - len(solar_spectrum))
        extended_solar[len(solar_spectrum):] = solar_spectrum[-1] * decay_factor
        
        # Use the extended spectrum
        enmap_rsr[:, 1] = extended_solar
    else:
        enmap_rsr[:, 1] = solar_spectrum
    
    # Create response functions for each band
    for i, (center_wavelength, fwhm) in enumerate(zip(all_wavelengths, all_fwhm)):
        # Convert FWHM to sigma for Gaussian
        sigma = fwhm / 2.355
        
        # Create Gaussian response function
        response = np.exp(-0.5 * ((wavelength_range_nm - center_wavelength) / sigma) ** 2)
        
        # Normalize
        response = response / np.max(response)
        
        # Add to RSR matrix (column index is i+2 because first two columns are wavelength and solar spectrum)
        enmap_rsr[:, i + 2] = response
    
    # Save RSR file
    srf_path = os.path.join(output_dir, "enmap.rsr")
    np.savetxt(srf_path, enmap_rsr, delimiter=' ', fmt='%.8f')
    
    print(f"EnMAP SRF created and saved to {srf_path}")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # VNIR bands (roughly up to 1000nm)
    plt.subplot(1, 2, 1)
    for i, (center_wavelength, _) in enumerate(zip(all_wavelengths, all_fwhm)):
        if center_wavelength <= 1000:  # VNIR range
            band_response = enmap_rsr[:, i + 2]
            plt.plot(wavelength_range_nm, band_response, alpha=0.5, linewidth=0.8)
    plt.title(f'EnMAP VNIR Spectral Response Functions')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Response')
    plt.grid(True, alpha=0.3)
    plt.xlim(400, 1000)
    
    # SWIR bands
    plt.subplot(1, 2, 2)
    for i, (center_wavelength, _) in enumerate(zip(all_wavelengths, all_fwhm)):
        if center_wavelength > 1000:  # SWIR range
            band_response = enmap_rsr[:, i + 2]
            plt.plot(wavelength_range_nm, band_response, alpha=0.5, linewidth=0.8)
    plt.title(f'EnMAP SWIR Spectral Response Functions')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Response')
    plt.grid(True, alpha=0.3)
    plt.xlim(1000, 2500)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "enmap_srf_plot.png")
    plt.savefig(plot_path, dpi=150)
    print(f"SRF visualization saved to {plot_path}")
    
    # Plot the solar spectrum to visualize the extension
    plt.figure(figsize=(10, 5))
    plt.plot(wavelength_range_nm, enmap_rsr[:, 1], 'b-', label='Solar Spectrum')
    if len(solar_spectrum) < n_wavelengths:
        # Highlight the extended region
        plt.axvspan(wavelength_range_nm[len(solar_spectrum)], wavelength_range_nm[-1], 
                   alpha=0.2, color='red', label='Extended Region')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Solar Irradiance')
    plt.title('Solar Spectrum (with Extension for EnMAP)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    solar_plot_path = os.path.join(output_dir, "enmap_solar_spectrum.png")
    plt.savefig(solar_plot_path, dpi=150)
    print(f"Solar spectrum visualization saved to {solar_plot_path}")
    
    return srf_path

def check_enmap_rsr(rsr_path):
    """Check the created EnMAP RSR file to make sure it looks right"""
    rsr = np.loadtxt(rsr_path)
    
    print(f"RSR shape: {rsr.shape}")
    print(f"First few wavelengths: {rsr[:5, 0]}")
    
    # Count non-zero bands at a specific wavelength
    wavelength_idx = np.where(np.abs(rsr[:, 0] - 0.5) < 0.001)[0][0]  # Find index for ~500nm
    non_zero_bands = np.sum(rsr[wavelength_idx, 2:] > 0.01)
    print(f"Number of bands with response > 0.01 at ~500nm: {non_zero_bands}")
    
    # Plot a sample of the response
    plt.figure(figsize=(10, 6))
    
    # Plot VNIR region (400-1000nm)
    for i in range(10):  # Plot first 10 bands
        wavelength_idx = np.where(np.abs(rsr[:, 0] - (0.4 + i*0.05)) < 0.001)[0][0]
        if np.max(rsr[wavelength_idx, 2:]) > 0.01:
            plt.plot(rsr[:, 0] * 1000, rsr[:, i+2], alpha=0.7, label=f"Band {i}")
    
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Response")
    plt.title("Sample of EnMAP bands in the VNIR region")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(rsr_path), "enmap_sample_check.png"))
    
    print("Check plot saved to enmap_sample_check.png")

if __name__ == "__main__":
    excel_path = "/Users/princemensah/Desktop/InstaDeep/prosailvae/EnMAP_Spectral_Bands_update.xlsx"
    sentinel2_rsr_path = "/Users/princemensah/Desktop/InstaDeep/prosailvae/data/simulated_dataset/sentinel2.rsr"
    output_dir = "/Users/princemensah/Desktop/InstaDeep/prosailvae/data/simulated_dataset"
    
    rsr_path = create_enmap_rsr_from_excel(excel_path, sentinel2_rsr_path, output_dir)
    check_enmap_rsr(rsr_path) 