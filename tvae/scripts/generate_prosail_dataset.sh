#!/bin/bash
#PBS -N genprosail
#PBS -l select=1:ncpus=4:mem=92G
#PBS -l walltime=08:00:00

# Constants
DATA_DIR="data/simulated_dataset"
RSR_DIR="rsr_data"
NOISE=0.01
TOTAL_TRAIN_SAMPLES=500000
BATCH_SIZE=100000
NUM_BATCHES=5

# Create directories if they don't exist
mkdir -p ${DATA_DIR}
mkdir -p ${DATA_DIR}/temp_batches

echo "Starting PROSAIL dataset generation for Sentinel-2 only..."
echo "Total training samples: ${TOTAL_TRAIN_SAMPLES} in ${NUM_BATCHES} batches"
echo "Data directory: ${DATA_DIR}"
echo "RSR directory: ${RSR_DIR}"

# Create a temporary script with the fix for the missing lai_corr parameter
TEMP_DIR=$(mktemp -d)
TEMP_SCRIPT="${TEMP_DIR}/generate_sentinel2_data.py"

cat > ${TEMP_SCRIPT} << 'EOF'
#!/usr/bin/env python3
import os
import sys
import argparse
import torch
from dataset.dataset_utils import min_max_to_loc_scale
from prosailvae.prosail_var_dists import VariableDistribution, get_prosail_var_dist
from prosailvae.ProsailSimus import ProsailSimulator, SensorSimulator

# Import the necessary functions from generate_dataset.py
sys.path.append('.')
from dataset.generate_dataset import (
    sample_prosail_vars,
    simulate_reflectances,
    get_bands_norm_factors
)

def save_sentinel2_dataset(data_dir, data_file_prefix, rsr_dir, nb_simus, noise=0, 
                         uniform_mode=False, lai_corr=True, 
                         prosail_var_dist_type="legacy", 
                         lai_var_dist=None, lai_corr_mode="v2", 
                         lai_thresh=None, prospect_version="5"):
    """Save a Sentinel-2 dataset with the specified parameters."""
    # Create output directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Initialize simulators
    psimulator = ProsailSimulator(prospect_version=prospect_version)
    bands = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]  # Sentinel-2 bands
    ssimulator = SensorSimulator(os.path.join(rsr_dir, "sentinel2.rsr"), bands=bands)
    
    # Sample PROSAIL variables
    prosail_vars = sample_prosail_vars(
        nb_simus=nb_simus,
        prosail_var_dist_type=prosail_var_dist_type,
        uniform_mode=uniform_mode,
        lai_corr=lai_corr,
        lai_var_dist=lai_var_dist,
        lai_corr_mode=lai_corr_mode,
        lai_thresh=lai_thresh,
    )
    
    # Simulate reflectances
    prosail_s2_sim = simulate_reflectances(
        prosail_vars,
        noise=noise,
        psimulator=psimulator,
        ssimulator=ssimulator,
        n_samples_per_batch=1024,
    )
    
    # Calculate normalization factors
    (
        norm_mean,
        norm_std,
        cos_angles_loc,
        cos_angles_scale,
        idx_loc,
        idx_scale,
    ) = get_bands_norm_factors(
        torch.from_numpy(prosail_s2_sim).float().transpose(1, 0), mode="quantile"
    )
    
    # Save all data and normalization factors
    torch.save(
        torch.from_numpy(prosail_vars),
        os.path.join(data_dir, f"{data_file_prefix}prosail_sim_vars.pt"),
    )
    torch.save(
        torch.from_numpy(prosail_s2_sim),
        os.path.join(data_dir, f"{data_file_prefix}prosail_s2_sim_refl.pt"),
    )
    torch.save(norm_mean, os.path.join(data_dir, f"{data_file_prefix}norm_mean.pt"))
    torch.save(norm_std, os.path.join(data_dir, f"{data_file_prefix}norm_std.pt"))
    torch.save(
        cos_angles_loc, os.path.join(data_dir, f"{data_file_prefix}angles_loc.pt")
    )
    torch.save(
        cos_angles_scale, os.path.join(data_dir, f"{data_file_prefix}angles_scale.pt")
    )
    torch.save(idx_loc, os.path.join(data_dir, f"{data_file_prefix}idx_loc.pt"))
    torch.save(idx_scale, os.path.join(data_dir, f"{data_file_prefix}idx_scale.pt"))
    
    print(f"Sentinel-2 dataset with {nb_simus} samples saved to {data_dir}")
    return prosail_vars, prosail_s2_sim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                      help="Directory where data is saved")
    parser.add_argument('--file_prefix', type=str, default='',
                      help="Prefix for the data files")
    parser.add_argument('--rsr_dir', type=str, required=True,
                      help="Directory where the RSR data is stored")
    parser.add_argument('--n_samples', type=int, required=True,
                      help="Number of samples to generate")
    parser.add_argument('--noise', type=float, default=0.0,
                      help="Noise to add to the simulated data")
    parser.add_argument('--dist_type', type=str, default='legacy',
                      help="Distribution type for the PROSAIL variables")
    parser.add_argument('--lai_corr_mode', type=str, default='v2',
                      help="Mode for the LAI correlation (v1 or v2)")
    parser.add_argument('--lai_thresh', type=float, default=None,
                      help="Threshold for LAI correlation")
    
    args = parser.parse_args()
    
    print(f"Generating Sentinel-2 dataset with {args.n_samples} samples")
    print(f"Data directory: {args.data_dir}")
    print(f"RSR directory: {args.rsr_dir}")
    print(f"File prefix: {args.file_prefix}")
    print(f"Distribution type: {args.dist_type}")
    print(f"LAI correlation mode: {args.lai_corr_mode}")
    
    save_sentinel2_dataset(
        args.data_dir,
        args.file_prefix,
        args.rsr_dir,
        args.n_samples,
        args.noise,
        uniform_mode=False,
        lai_corr=True,  # Always use LAI correlation
        prosail_var_dist_type=args.dist_type,
        lai_corr_mode=args.lai_corr_mode,
        lai_thresh=args.lai_thresh,
        prospect_version="PRO",
    )

if __name__ == "__main__":
    main()
EOF

chmod +x ${TEMP_SCRIPT}

# Generate test dataset (10,000 samples)
echo "Generating test dataset..."
python ${TEMP_SCRIPT} \
  --data_dir ${DATA_DIR} \
  --n_samples 10000 \
  --file_prefix "test_" \
  --noise ${NOISE} \
  --rsr_dir ${RSR_DIR} \
  --dist_type "new_v2" \
  --lai_corr_mode "v2"

# Generate validation dataset (10,000 samples)
echo "Generating validation dataset..."
python ${TEMP_SCRIPT} \
  --data_dir ${DATA_DIR} \
  --n_samples 10000 \
  --file_prefix "valid_" \
  --noise ${NOISE} \
  --rsr_dir ${RSR_DIR} \
  --dist_type "new_v2" \
  --lai_corr_mode "v2"

# Generate training dataset in batches
echo "Generating ${NUM_BATCHES} training batches of ${BATCH_SIZE} samples each..."

for i in $(seq 1 ${NUM_BATCHES}); do
    echo "Generating batch ${i} of ${NUM_BATCHES}..."
    
    # Create a directory for this batch
    BATCH_DIR="${DATA_DIR}/temp_batches/batch_${i}"
    mkdir -p "${BATCH_DIR}"
    
    python ${TEMP_SCRIPT} \
      --data_dir "${BATCH_DIR}" \
      --n_samples "${BATCH_SIZE}" \
      --file_prefix "batch_${i}_" \
      --noise "${NOISE}" \
      --rsr_dir "${RSR_DIR}" \
      --dist_type "new_v2" \
      --lai_corr_mode "v2"
done

# Combine the batches
echo "Combining batches into a single training dataset..."
python - << EOF
import os
import torch
import glob
import shutil

# Paths
data_dir = "${DATA_DIR}"
batch_dir = "${DATA_DIR}/temp_batches"
output_vars_path = os.path.join(data_dir, "train_prosail_sim_vars.pt")
output_refl_path = os.path.join(data_dir, "train_prosail_s2_sim_refl.pt")

# Find all batch files
vars_files = sorted(glob.glob(os.path.join(batch_dir, "*/batch_*_prosail_sim_vars.pt")))
refl_files = sorted(glob.glob(os.path.join(batch_dir, "*/batch_*_prosail_s2_sim_refl.pt")))

if len(vars_files) == 0 or len(refl_files) == 0:
    print("No batch files found!")
    exit(1)

print(f"Found {len(vars_files)} variable batches and {len(refl_files)} reflectance batches")

# Load and combine all batches
combined_vars = []
combined_refl = []
total_samples = 0

for vars_path, refl_path in zip(vars_files, refl_files):
    print(f"Loading batch from {os.path.basename(os.path.dirname(vars_path))}")
    
    try:
        batch_vars = torch.load(vars_path)
        batch_refl = torch.load(refl_path)
        
        batch_samples = batch_vars.shape[0]
        total_samples += batch_samples
        
        combined_vars.append(batch_vars)
        combined_refl.append(batch_refl)
        
        print(f"  Added {batch_samples} samples")
    except Exception as e:
        print(f"  Error loading batch: {e}")

# Concatenate all batches
if combined_vars and combined_refl:
    all_vars = torch.cat(combined_vars, dim=0)
    all_refl = torch.cat(combined_refl, dim=0)
    
    print(f"Combined dataset has {all_vars.shape[0]} samples")
    
    # Save the combined dataset
    torch.save(all_vars, output_vars_path)
    torch.save(all_refl, output_refl_path)
    
    print(f"Saved combined training dataset to:")
    print(f"  {output_vars_path}")
    print(f"  {output_refl_path}")
    
    # Copy normalization files from one of the batches
    norm_files = ["norm_mean.pt", "norm_std.pt", "angles_loc.pt", "angles_scale.pt", "idx_loc.pt", "idx_scale.pt"]
    batch_dir_first = os.path.dirname(vars_files[0])
    
    for norm_file in norm_files:
        batch_norm_path = os.path.join(batch_dir_first, f"batch_1_{norm_file}")
        if os.path.exists(batch_norm_path):
            train_norm_path = os.path.join(data_dir, f"train_{norm_file}")
            shutil.copy(batch_norm_path, train_norm_path)
            print(f"Copied normalization file: {train_norm_path}")
else:
    print("No data to combine!")
    exit(1)
EOF

# Clean up
echo "Cleaning up temporary files..."
rm -rf "${TEMP_DIR}"
rm -rf "${DATA_DIR}/temp_batches"

echo "All done! Sentinel-2 dataset generation complete."
echo "Dataset files created in ${DATA_DIR}:"
ls -lh ${DATA_DIR}/*.pt
