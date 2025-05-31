#!/usr/bin/env bash
set -o errexit

echo " Starting Render build process..."

# Upgrade pip and install build tools
python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
pip install -r requirements.txt

# Clear pip cache to reduce slug size
pip cache purge

# Check model artifacts
echo " Verifying model files..."
if [ ! -f "model/model.h5" ]; then
  echo " ERROR: model/model.h5 not found!"
  echo " Contents of model directory:"
  ls -l model/
  exit 1
fi

if [ ! -f "model/labels.json" ]; then
  echo " ERROR: model/labels.json not found!"
  exit 1
fi

echo " Model files found:"
ls -l model/

echo " Build completed successfully for Render!"
