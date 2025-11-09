#!/bin/bash

# Setup script for Research Semantic POC

set -e

echo "==========================================="
echo "Research Semantic POC - Setup"
echo "==========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model
echo ""
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm || echo "⚠️  spaCy model download failed (optional)"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data
mkdir -p output
mkdir -p notebooks
mkdir -p tests

echo "✅ Directories created"

# Clone aiscientist data if not exists
echo ""
echo "Checking for dataset..."
if [ -d "data/aiscientist" ]; then
    echo "Dataset already exists. Skipping..."
else
    echo "Cloning aiscientist repository..."
    cd data
    git clone https://github.com/sergeicu/aiscientist
    cd ..
    echo "✅ Dataset downloaded"
fi

# Verify dataset
if [ -f "data/aiscientist/pubmed_data_2000.csv" ]; then
    echo "✅ Dataset verified: pubmed_data_2000.csv found"
else
    echo "⚠️  Warning: pubmed_data_2000.csv not found"
fi

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x scripts/*.py
echo "✅ Scripts are now executable"

# Summary
echo ""
echo "==========================================="
echo "Setup Complete!"
echo "==========================================="
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run the pipeline:"
echo "     python scripts/run_full_pipeline.py"
echo ""
echo "  3. Or start Claude Code:"
echo "     claude"
echo ""
echo "For more info, see QUICKSTART.md"
echo ""
