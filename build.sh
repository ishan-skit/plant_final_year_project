#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

python -m pip install --upgrade pip
python -m pip install --upgrade setuptools