#!/usr/bin/env bash
set -o errexit

echo "=== Starting Render Build ==="

# System setup
python -m pip install --upgrade pip setuptools wheel --no-cache-dir

# Install ONLY production dependencies (no dev packages)
pip install -r requirements.txt --no-cache-dir

# Verify critical model files
required_files=(
  "model/plant_disease_model.h5"
  "model/class_names.json"
  "model/deploy_config.json"
)

echo "--- Verifying Model Files ---"
for file in "${required_files[@]}"; do
  if [ ! -f "$file" ]; then
    echo "ERROR: Missing required file: $file"
    echo "Contents of model/:"
    ls -l model/
    exit 1
  fi
done

# Cleanup to reduce slug size (ignore read-only errors)
echo "--- Cleaning up build files ---"
pip cache purge || true
find /usr/local/lib/python*/ -type d -name "tests" -exec rm -rf {} + || true
find /usr/local/lib/python*/ -type d -name "__pycache__" -exec rm -rf {} + || true

echo "✓ Build completed successfully"