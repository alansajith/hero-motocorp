# AI-Driven Vehicle Damage Detection & Intelligent Assessment

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive computer vision system for automatic vehicle damage detection, classification, and intelligent assessment using state-of-the-art deep learning models.

## 🎯 Features

- **Multi-Stage AI Pipeline**
  - YOLOv8 for damage detection & localization
  - U-Net for fine segmentation
  - EfficientNet for damage classification
  - Rule-based severity estimation
  - Cosmetic vs functional damage assessment

- **Intelligent Analysis**
  - Automatic severity scoring (low/medium/high)
  - Vehicle part identification
  - Damage area and length measurement
  - Critical component detection

- **Explainability**
  - Grad-CAM heatmaps
  - Segmentation overlays
  - Confidence scores
  - Visual reports with annotations

- **REST API**
  - Single image processing
  - Batch processing
  - JSON output format
  - CORS enabled

- **M4 Pro Optimized**
  - Metal Performance Shaders (MPS) support
  - Apple Silicon ARM64 compatible
  - Efficient GPU acceleration

## 🏗️ Architecture

```
Input Image
    ↓
YOLOv8 Detection → Bounding Boxes + Masks
    ↓
U-Net Segmentation → Refined Masks (optional)
    ↓
EfficientNet Classification → Damage Type
    ↓
Part Identification → Vehicle Part
    ↓
Severity Estimation → Low/Medium/High
    ↓
Cosmetic/Functional → Final Assessment
    ↓
JSON Output + Visualizations
```

## 📋 Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4) or any system with Python 3.9+
- 16GB+ RAM recommended
- Annotated vehicle damage dataset

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository_url>
cd hero

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

Organize your dataset:

```
data/
├── train/
│   ├── images/
│   └── labels/  # YOLO format annotations
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Create `data.yaml`:

```yaml
path: ./data
train: train/images
val: val/images
test: test/images

nc: 5  # number of classes
names: ['scratch', 'dent', 'crack', 'paint_damage', 'broken_part']
```

### 3. Train Models

**YOLOv8 Detection:**

```bash
python scripts/train_detection.py --data data.yaml --epochs 100 --batch 16
```

**U-Net Segmentation (optional):**

```bash
python scripts/train_segmentation.py --data data/segmentation --epochs 50
```

**Classification (optional):**

```bash
python scripts/train_classification.py --data data/classification --epochs 30
```

### 4. Run Inference

```bash
python examples/inference_example.py --image path/to/vehicle.jpg --output outputs/
```

### 5. Start API Server

```bash
# Development
uvicorn src.api.app:app --reload

# Production
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📝 API Usage

### Detect Damage

```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@vehicle_image.jpg" \
  -F "generate_report=true"
```

### Response

```json
{
  "damage_detected": true,
  "num_damages": 2,
  "damages": [
    {
      "damage_id": 0,
      "damage_type": "scratch",
      "severity": "medium",
      "vehicle_part": "left_door",
      "is_cosmetic": true,
      "is_functional": false,
      "detection_confidence": 0.92,
      "explanation": "Medium severity scratch - primarily cosmetic"
    }
  ],
  "overall_assessment": {
    "max_severity": "medium",
    "has_functional_damage": false,
    "requires_immediate_attention": false
  }
}
```

### Batch Processing

```bash
curl -X POST "http://localhost:8000/batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

### Health Check

```bash
curl "http://localhost:8000/health"
```

## 📂 Project Structure

```
hero/
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
├── src/
│   ├── data/              # Data loading & augmentation
│   ├── models/            # Model implementations
│   ├── analysis/          # Severity & classification logic
│   ├── explainability/    # Grad-CAM
│   ├── visualization/     # Overlays & reports
│   ├── pipeline/          # Inference pipeline
│   ├── api/              # FastAPI server
│   └── utils/            # Config & metrics
│
├── scripts/
│   ├── train_detection.py
│   ├── train_segmentation.py
│   └── train_classification.py
│
├── examples/
│   └── inference_example.py
│
├── data/                  # Dataset directory
├── models/                # Trained model weights
├── outputs/              # Inference results
└── logs/                 # Application logs
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

- Model architectures
- Training hyperparameters
- Severity thresholds
- Damage classes
- API settings
- Device selection (mps/cpu)

## 🎨 Damage Types

- **Scratch**: Surface-level linear damage
- **Dent**: Deformation without breaking
- **Crack**: Structural fracture
- **Paint Damage**: Discoloration or peeling
- **Broken Part**: Complete component failure

## 📊 Severity Levels

- **Low**: Minor cosmetic issues, < 5% area
- **Medium**: Moderate damage, 5-15% area
- **High**: Severe damage, > 15% area or critical parts

## 🔧 Advanced Usage

### Custom Dataset Format

```python
from src.data import DamageDetectionDataset

dataset = DamageDetectionDataset(
    data_dir="path/to/data",
    split="train",
    image_size=640,
    annotation_format="yolo"  # or "coco"
)
```

### Custom Pipeline

```python
from src.pipeline import DamageDetectionPipeline

pipeline = DamageDetectionPipeline(
    detection_model_path="models/detection/best.pt",
    segmentation_model_path="models/segmentation/best.pt",
    classification_model_path="models/classification/best.pt"
)

results = pipeline.process_image("vehicle.jpg")
```

### Generate Reports

```python
from src.visualization import ReportGenerator

report_gen = ReportGenerator()
files = report_gen.generate_report(
    image="vehicle.jpg",
    results=results,
    output_dir="reports"
)
```

## 🐳 Docker Deployment

```bash
# Build image
docker build -t vehicle-damage-detection .

# Run container
docker run -p 8000:8000 vehicle-damage-detection
```

## 📈 Performance

- **Detection mAP**: Varies by dataset
- **Segmentation IoU**: Varies by dataset
- **Classification Accuracy**: Varies by dataset
- **Inference Speed**: ~200-500ms per image (M4 Pro with MPS)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Ultralytics YOLOv8
- PyTorch Team
- segmentation_models_pytorch
- EfficientNet authors

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This system requires annotated training data. Model performance depends on dataset quality and size. For production use, ensure thorough testing and validation.
