#!/bin/bash
# Render build script to train models before starting the app

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Training models and generating pickle files..."
python train_models.py

echo "Build complete!"
