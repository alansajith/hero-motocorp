#!/bin/bash

# Script to create a ZIP file for Google Colab transfer
# This excludes virtual environments, cache files, and large data directories

echo "📦 Creating Colab-ready ZIP file..."

cd ~/Desktop

zip -r hero-code.zip hero \
  -x "hero/venv/*" \
  -x "hero/__pycache__/*" \
  -x "hero/.git/*" \
  -x "hero/data/train/*" \
  -x "hero/data/val/*" \
  -x "hero/data/test/*" \
  -x "hero/models/*" \
  -x "hero/outputs/*" \
  -x "hero/runs/*" \
  -x "*.pyc" \
  -x "*/.DS_Store"

echo "✅ Created hero-code.zip (~5-10 MB)"
echo "📍 Location: ~/Desktop/hero-code.zip"
echo ""
echo "Next steps:"
echo "1. Upload hero-code.zip to Google Drive"
echo "2. Upload your Roboflow dataset ZIP to Google Drive"
echo "3. Open colab_training.ipynb in Google Colab"
