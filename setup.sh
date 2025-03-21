#!/bin/bash

# Setup script for Small Language Model Project

echo "Setting up Small Language Model Project..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install --upgrade setuptools wheel
echo "Installing core dependencies..."
pip install -r requirements.txt

# Create necessary directories if they don't exist
echo "Creating project directories..."
mkdir -p data/processed
mkdir -p models/optimized
mkdir -p app/static

echo "Setup complete! You can now activate the virtual environment with:"
echo "source venv/bin/activate"

echo ""
echo "Quick start:"
echo "1. Download a small model: python scripts/download_model.py --model distilgpt2"
echo "2. Optimize the model: python scripts/optimize_model.py --model-path models/distilgpt2 --output-path models/optimized/distilgpt2 --model-type causal-lm"
echo "3. Run the web app: python app/app.py --model-path models/optimized/distilgpt2/quantized --model-type causal-lm"
echo ""
echo "For more information, see the README.md file."
