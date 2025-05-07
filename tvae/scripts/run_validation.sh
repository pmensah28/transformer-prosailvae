#!/bin/bash

# Name: run_validation.sh
# Description: Shell script to run all site validation script for t-VAE models

# Define directories relative to the current working directory
CURRENT_DIR="$(pwd)"
# Default model path (can be overridden using command-line arguments)
DEFAULT_MODEL_PATH="$CURRENT_DIR/results/1"
MODEL_DIR="models/checkpoints"

# Default values
MODEL_PATH=$DEFAULT_MODEL_PATH
OUTPUT_DIR="validation_results/"
DATA_DIR="data"
DEVICE="cpu"
METHOD="closest"
RSR_DIR="$CURRENT_DIR/rsr_data"
RECONSTRUCTION=""
EXPORT_PLOTS=""
DATASETS="all"
CROP_TYPE="all"
COMBINE_RESULTS=""
EXPORT_DETAILED=""
COMPARISON_PLOT=""
# Mode will be set after parsing arguments based on dataset

# Function to display help message
function display_help {
    echo "Usage: $0 [options]"
    echo "All site validation script for transformer-VAE models on multiple datasets"
    echo ""
    echo "Options:"
    echo "  --model-path PATH      Path to the model checkpoint (default: $DEFAULT_MODEL_PATH)"
    echo "  --data-dir DIR         Base directory for validation data (default: data)"
    echo "  --output-dir DIR       Directory to save results (default: validation_results/unified)"
    echo "  --device DEVICE        Device to use (cuda or cpu, default: cpu)"
    echo "  --method METHOD        Interpolation method (default: closest)"
    echo "  --mode MODE            Mode for predictions (automatically selected based on dataset:"
    echo "                         'lat_mode' for BelSAR, 'sim_tg_mean' for FRM4VEG)"
    echo "  --rsr-dir DIR          Directory with RSR data (default: rsr_data)"
    echo "  --datasets DATASETS    Datasets to validate (all, belsar, frm4veg, frm4veg_barrax2018,"
    echo "                         frm4veg_barrax2021, frm4veg_wytham2018) (default: all)"
    echo "                         Can specify multiple with: --datasets 'belsar frm4veg_barrax2018'"
    echo "  --crop-type TYPE       Crop type filter for BelSAR (all, wheat, maize) (default: all)"
    echo "  --combine-results      Generate combined validation plots across all datasets"
    echo "  --reconstruction       Save reconstruction errors" 
    echo "  --export-plots         Export plots as PDF for publication"
    echo "  --export-detailed      Export detailed sample-by-sample CSV files"
    echo "  --comparison-plot      Generate side-by-side comparison plots for multiple variables"
    echo "  --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --model-path results/model1"
    echo "  $0 --datasets belsar --crop-type wheat"
    echo "  $0 --datasets 'frm4veg_barrax2018 frm4veg_wytham2018' --combine-results"
    echo "  $0 --datasets all --combine-results --export-plots --comparison-plot"
    echo "  $0 --datasets belsar --mode sim_tg_mean  # Override automatic mode selection"
    exit 0
}

# Flag to track if mode was explicitly set
MODE_EXPLICITLY_SET=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --method)
      METHOD="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      MODE_EXPLICITLY_SET=true
      shift 2
      ;;
    --rsr-dir)
      RSR_DIR="$2"
      shift 2
      ;;
    --datasets)
      DATASETS="$2"
      shift 2
      ;;
    --crop-type)
      CROP_TYPE="$2"
      shift 2
      ;;
    --combine-results)
      COMBINE_RESULTS="--combine_results"
      shift
      ;;
    --reconstruction)
      RECONSTRUCTION="--reconstruction"
      shift
      ;;
    --export-plots)
      EXPORT_PLOTS="--export_plots"
      shift
      ;;
    --export-detailed)
      EXPORT_DETAILED="--export_detailed"
      shift
      ;;
    --comparison-plot)
      COMPARISON_PLOT="--comparison_plot"
      shift
      ;;
    --help)
      display_help
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Choose appropriate mode based on dataset if not explicitly set
if [ "$MODE_EXPLICITLY_SET" = false ]; then
  # Check if only BelSAR
  if [[ "$DATASETS" == "belsar" ]]; then
    MODE="lat_mode"
    echo "Using BelSAR-optimized default mode: lat_mode"
  # Check if only FRM4VEG or FRM4VEG site
  elif [[ "$DATASETS" == "frm4veg"* ]]; then
    MODE="sim_tg_mean"
    echo "Using FRM4VEG-optimized default mode: sim_tg_mean"
  # For all or combined datasets
  else
    MODE="sim_tg_mean"
    echo "Using default mode for combined/multiple datasets: sim_tg_mean"
  fi
fi

# Validate crop type
if [ "$CROP_TYPE" != "all" ] && [ "$CROP_TYPE" != "wheat" ] && [ "$CROP_TYPE" != "maize" ]; then
    echo "Error: Crop type must be 'wheat', 'maize', or 'all'"
    echo "Use --help for usage information"
    exit 1
fi

# Check if default model path exists, if not try to find latest checkpoint
if [ ! -f "$MODEL_PATH" ] && [ ! -d "$MODEL_PATH" ]; then
  echo "Default model path not found, looking for latest checkpoint..."
  LATEST_MODEL=$(ls -t $MODEL_DIR/*.pth 2>/dev/null | head -n 1)
  
  if [ -z "$LATEST_MODEL" ]; then
    echo "No model checkpoints found. Please specify a valid model path with --model-path."
    exit 1
  else
    MODEL_PATH=$LATEST_MODEL
    echo "Using latest checkpoint: $MODEL_PATH"
  fi
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Display settings
echo "========================================"
echo "All Site Validation"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Interpolation method: $METHOD"
echo "Prediction mode: $MODE"
echo "RSR directory: $RSR_DIR"
echo "Datasets: $DATASETS"
echo "Crop type: $CROP_TYPE"

if [ -n "$COMBINE_RESULTS" ]; then
  echo "Combined results: Yes"
else
  echo "Combined results: No"
fi

if [ -n "$RECONSTRUCTION" ]; then
  echo "Saving reconstruction errors: Yes"
else
  echo "Saving reconstruction errors: No"
fi

if [ -n "$EXPORT_PLOTS" ]; then
  echo "Export plots: Yes"
else
  echo "Export plots: No"
fi

if [ -n "$EXPORT_DETAILED" ]; then
  echo "Export detailed CSV: Yes"
else
  echo "Export detailed CSV: No"
fi

if [ -n "$COMPARISON_PLOT" ]; then
  echo "Comparison plots: Yes"
else
  echo "Comparison plots: No"
fi
echo "========================================"

# Add current directory to PYTHONPATH
export PYTHONPATH="$CURRENT_DIR:$PYTHONPATH"

# Run the validation script
python validation.py \
  --model_path "$MODEL_PATH" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --method "$METHOD" \
  --mode "$MODE" \
  --rsr_dir "$RSR_DIR" \
  --datasets $DATASETS \
  --crop_type "$CROP_TYPE" \
  $COMBINE_RESULTS \
  $RECONSTRUCTION \
  $EXPORT_PLOTS \
  $EXPORT_DETAILED \
  $COMPARISON_PLOT

# Check execution status
if [ $? -eq 0 ]; then
  echo "Validation completed successfully!"
  echo "Results saved to: $OUTPUT_DIR"
else
  echo "Validation failed. Check unified_validation.log for details."
  exit 1
fi