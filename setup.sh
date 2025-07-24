#!/bin/bash

# Minimal installation to run the GTFS Railways package

source /opt/miniconda3/etc/profile.d/conda.sh
conda create -n test python=3.8 -y
sleep 2
conda activate test

pip install ../gtfs_railways
pip install ../gtfs_railways/external_packages/osmread
pip install ../gtfs_railways/external_packages/gtfspy
