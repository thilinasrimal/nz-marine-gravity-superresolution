#!/bin/bash

# NZ Marine Gravity Super-Resolution Pipeline
echo "=========================================="
echo "NZ EEZ Marine Gravity Super-Resolution"
echo "=========================================="

# Create directory structure
mkdir -p data/{raw,processed,splits}
mkdir -p outputs/{models,predictions,figures}
mkdir -p notebooks

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run data preprocessing
echo "Step 1: Loading and preprocessing data..."
python -c "from src.data_loader import NZGravityDataLoader; \
           import yaml; \
           with open('config/config.yaml') as f: config = yaml.safe_load(f); \
           loader = NZGravityDataLoader(config); \
           features, targets = loader.prepare_training_data(); \
           patches = loader.create_patches(features, targets); \
           print(f'Created {len(patches)} training patches')"

# Train model
echo "Step 2: Training CNN super-resolution model..."
python src/train.py

# Generate visualizations
echo "Step 3: Generating visualizations and analysis..."
jupyter nbconvert --to notebook --execute notebooks/03_visualization.ipynb --output 03_visualization_executed.ipynb

echo "=========================================="
echo "Pipeline completed successfully!"
echo "Results saved in outputs/"
echo "=========================================="