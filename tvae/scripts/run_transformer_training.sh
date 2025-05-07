#!/bin/bash

# Name: run_transformer.sh
# Description: Shell script to train a t-VAE model
#              using the Transformer-based encoder approach.

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate tvae

# Define directories relative to the current working directory
CURRENT_DIR="$(pwd)"
CONFIG_DIR="$CURRENT_DIR/config/"
RESULTS_DIR="$CURRENT_DIR/results"
MODELS_DIR="$CURRENT_DIR/trained_models/transformer"
RSR_DIR="$CURRENT_DIR/data/simulated_dataset"

# Create results and models directories if they don't exist
mkdir -p "$RESULTS_DIR"
mkdir -p "$MODELS_DIR"

# Add current directory to PYTHONPATH
export PYTHONPATH="$CURRENT_DIR:$PYTHONPATH"

# Run the training script with the Transformer configuration
python tvae/train.py \
  -c transformer_config.json \
  -cd "$CONFIG_DIR" \
  -r "$RESULTS_DIR" \
  -rsr "$RSR_DIR" \
  -p true
