# Quick Start Guide - Vehicle Damage Detection

## ✅ What You Just Fixed

The installation error was due to:
- Duplicate `pytorch-grad-cam` package in requirements (doesn't exist)
- Python 3.14 compatibility issues

**Fixed by:**
- Removed duplicate `pytorch-grad-cam` line
- Updated numpy version constraint

---

## 📦 Next Steps After Installation Completes

### 1. Download Your Roboflow Dataset

From Roboflow, click "Download Dataset" and select **YOLOv8** format.

### 2. Extract Dataset

```bash
cd ~/Desktop/hero

# Extract the downloaded zip
unzip ~/Downloads/2024-09-29-3-11am.v2-yolov9.yolov9pytorch.zip -d ./dataset_temp

# Move to correct locations
cp -r dataset_temp/train/* data/train/
cp -r dataset_temp/valid/* data/val/
cp -r dataset_temp/test/* data/test/

# Copy data.yaml
cp dataset_temp/data.yaml ./data.yaml

# Clean up
rm -rf dataset_temp
```

### 3. Update data.yaml Paths

Edit `data.yaml`:

```yaml
path: .  # Current directory
train: data/train/images
val: data/val/images
test: data/test/images

nc: 5  # or whatever your dataset has
names: ['scratch', 'dent', 'crack', 'paint_damage', 'broken_part']  # adjust to your classes
```

### 4. Verify Dataset

```bash
ls data/train/images | wc -l   # Should show ~6890
ls data/val/images | wc -l     # Should show ~175
ls data/test/images | wc -l    # Should show ~84
```

### 5. Train YOLOv8 Detection Model

```bash
# Activate virtual environment (if not already)
source venv/bin/activate

# Start training (will use M4 Pro GPU via MPS)
python scripts/train_detection.py --data data.yaml --epochs 100 --batch 16 --imgsz 640
```

**Training Info:**
- With ~7000 images, training will take several hours on M4 Pro
- Model will save to `models/detection/damage_detection/weights/best.pt`
- You'll see real-time metrics and loss curves
- Training will auto-use MPS (Metal Performance Shaders) for GPU acceleration

### 6. Test Inference (After Training)

```bash
python examples/inference_example.py \
  --image data/test/images/any_test_image.jpg \
  --detection-model models/detection/damage_detection/weights/best.pt \
  --output outputs/test_results
```

### 7. Start API Server

```bash
# Make sure to update config to point to trained model
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

Then test with:
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@data/test/images/test_image.jpg"
```

---

## 🎯 Optional: Train Other Models

### U-Net Segmentation (For Better Masks)

```bash
# Organize segmentation data first
python scripts/train_segmentation.py --data data/segmentation --epochs 50
```

### Damage Classification

```bash
# Organize classification data first (images grouped by damage type)
python scripts/train_classification.py --data data/classification --epochs 30
```

---

## 🔍 Monitor Training

YOLOv8 creates real-time training plots in:
- `models/detection/damage_detection/`

Look for:
- `results.png` - Loss and metrics curves
- `confusion_matrix.png` - Classification performance
- tensorboard logs (optional)

---

## ⚡ Performance Tips for M4 Pro

1. **Batch Size**: Start with 16, increase to 32 if memory allows
2. **Image Size**: 640x640 is optimal for speed/accuracy balance
3. **Workers**: Use 4-8 for data loading (`num_workers` in config)
4. **Mixed Precision**: YOLOv8 automatically uses it

---

## 🐛 Troubleshooting

**CUDA errors on M4?**
- The code auto-detects MPS, ignore CUDA warnings

**Out of memory?**
- Reduce batch size: `--batch 8`
- Reduce image size: `--imgsz 512`

**Slow training?**
- Check Activity Monitor - Python should use GPU
- Verify MPS is working: Check logs for "Using MPS"

---

Ready to train! 🚀
