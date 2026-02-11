#!/bin/bash
# Script to fix corrupted virtual environment

cd "/Users/doyungu/Documents/02-EEE/Year 3/dynamic_phasors/Circuit-Simulations/04-Framework"

echo "Removing corrupted virtual environment..."
rm -rf venv

echo "Creating fresh virtual environment..."
python3.13 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing required packages..."
pip install --upgrade pip
pip install numpy scipy matplotlib "pandas<3.0" jupyter ipykernel

echo "Installing framework in editable mode..."
pip install -e .

echo ""
echo "✓ Environment fixed!"
echo ""
echo "To use:"
echo "  source venv/bin/activate"
echo "  jupyter notebook"
