# Vehicle Damage Detection - Google Colab Setup

## 🚀 Quick Transfer to Colab

### Option 1: Upload via GitHub (Recommended)

**Step 1: Create GitHub Repository**
```bash
cd ~/Desktop/hero

# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit - Vehicle damage detection system"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/vehicle-damage-detection.git
git branch -M main
git push -u origin main
```

**Step 2: In Google Colab**
```python
# Clone your repository
!git clone https://github.com/YOUR_USERNAME/vehicle-damage-detection.git
%cd vehicle-damage-detection

# Install dependencies
!pip install -r requirements.txt

# Install project in editable mode
!pip install -e .
```

**Step 3: Upload Dataset to Google Drive**
1. Upload your Roboflow dataset ZIP to Google Drive
2. Mount Drive in Colab and extract

---

### Option 2: Direct ZIP Upload (For Quick Testing)

**Step 1: Create ZIP Archive on Mac**
```bash
cd ~/Desktop
zip -r hero.zip hero -x "*.git*" "*.pyc" "__pycache__/*" "venv/*" "*.DS_Store"
```

**Step 2: Upload to Colab**
- Upload `hero.zip` using Colab's file upload
- Extract and setup

---

### Option 3: Google Drive Sync (Best for Large Datasets)

**Step 1: Upload Project to Google Drive**
```bash
# Compress project (excluding venv and cache)
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
  -x "hero/runs/*"
```

Upload `hero-code.zip` and your dataset ZIP to Google Drive.

**Step 2: Use the Colab Notebook**
- Open `colab_training.ipynb` (I'll create this below)
- Follow the cells to mount Drive, extract, and train

---

## 📊 What to Upload

**Essential Files (~5-10 MB):**
- All `src/` code
- `scripts/` training scripts  
- `config.yaml`
- `requirements.txt`
- `setup.py`
- `data.yaml`

**Dataset (will be large):**
- Upload Roboflow dataset separately to Google Drive
- Extract in Colab

**Don't Upload:**
- `venv/` - Recreate in Colab
- `models/` - Will be created during training
- `outputs/` - Results directory
- `runs/` - Training logs
- `.git/` - Not needed for training

---

## ⚡ GPU Selection in Colab

1. **Runtime** → **Change runtime type**
2. Select **T4 GPU** (free tier)
3. Or **V100/A100** (Colab Pro)

---

## 🎯 Expected Training Time

- **T4 GPU**: ~2-3 hours for 100 epochs with 7000 images
- **V100 GPU**: ~1-2 hours
- **A100 GPU**: ~30-60 minutes

Much faster than M4 Pro! 🚀

---

## 📝 Next Steps

1. Choose your transfer method (GitHub recommended)
2. Open the Colab notebook I'm creating
3. Upload your Roboflow dataset to Drive
4. Run training cells
5. Download trained model back to your Mac

See `colab_training.ipynb` for the complete notebook!
