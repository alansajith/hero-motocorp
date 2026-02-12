# Dataset directory placeholder
# Place your training data here

## Directory Structure

```
data/
├── train/
│   ├── images/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   └── labels/
│       ├── image001.txt
│       ├── image002.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## YOLO Annotation Format

Each `.txt` file should contain annotations in YOLO format:

```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `class_id`: Integer class ID (0-4)
- `x_center, y_center`: Center coordinates (normalized 0-1)
- `width, height`: Box dimensions (normalized 0-1)

Example:
```
0 0.5 0.5 0.3 0.2
1 0.7 0.3 0.15 0.15
```

## Classes

0. scratch
1. dent
2. crack
3. paint_damage
4. broken_part
