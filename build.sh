#!/usr/bin/env bash

# Exit on error
set -o errexit

echo "Starting Render build process..."

# Upgrade pip and setuptools first
python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
pip install -r requirements.txt

# Clear pip cache to reduce build size
pip cache purge

echo "Build completed successfully for Render!"