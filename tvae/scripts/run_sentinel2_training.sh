#!/bin/bash

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate prosailvae

# Run the training script with the modified configuration
CURRENT_DIR="$(pwd)"
CONFIG_DIR="$CURRENT_DIR/config/"
RESULTS_DIR="$CURRENT_DIR/results"
RSR_DIR="$CURRENT_DIR/data/simulated_dataset"

# Create results directory if it doesn't exist
mkdir -p $RESULTS_DIR

# Add current directory to PYTHONPATH
export PYTHONPATH=$CURRENT_DIR:$PYTHONPATH

# Run the training with the modified configuration
python prosailvae/train.py \
  -c config.json \
  -cd $CONFIG_DIR \
  -r $RESULTS_DIR \
  -rsr $RSR_DIR \
  -p true 