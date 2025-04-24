# YOLOv8 Fruit and Vegetable Recognition

Real-time fruit and vegetable detection with price calculation using YOLOv8.

## Features

- Real-time webcam detection of fruits and vegetables
- Price calculation for the shopping cart
- Adjustable confidence threshold
- Screenshot capture
- Custom training with Roboflow datasets

## Dataset

Location: `C:\Yolo\dataset fruits and vegetables`

Format: YOLOv8 (train/valid folders with images and labels)

## Requirements

```
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.20.0
PyYAML>=6.0
matplotlib>=3.5.0
```

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/yolov8-fruit-vegetable-recognition.git
cd yolov8-fruit-vegetable-recognition

# Install dependencies
pip install -r requirements.txt
```

## Training

```bash
# Train model
python src/train_model.py --data "C:\Yolo\dataset fruits and vegetables\data.yaml" --epochs 100 --model-size s

# Parameters:
# --data: Path to data.yaml file
# --epochs: Number of training epochs (default: 100)
# --img-size: Training image size (default: 640)
# --batch: Batch size (default: 16)
# --model-size: YOLOv8 model size: n, s, m, l, x (default: s)
# --pretrained: Use pretrained weights (flag)
# --project: Project directory (default: runs/train)
# --name: Experiment name (default: fruits_veg_YYYYMMDD)
```

## Detection

```bash
# Run detector
python src/fruit_veg_detector.py

# Controls:
# q - Quit
# s - Save screenshot
# +/- - Adjust confidence threshold
# c - Clear cart
```
![image](https://github.com/user-attachments/assets/e997b7d1-dbd6-44fc-a57a-8588753dec5e)

## Dataset Analysis

```bash
# Analyze dataset
python src/dataset_stats.py --data "C:\Yolo\dataset fruits and vegetables"
```

## Customization

Edit the price dictionary in `src/fruit_veg_detector.py`:

```python
price_dict = {
    "apple": 1.20,
    "banana": 0.50,
    "orange": 0.80,
    "tomato": 0.75,
    # Add more items as needed
}
```

## Project Structure

```
yolov8-fruit-vegetable-recognition/
├── src/
│   ├── fruit_veg_detector.py  # Main detection script
│   ├── train_model.py         # Training script
│   └── dataset_stats.py       # Dataset analysis tool
├── models/                    # Store trained models
├── dataset/                   # Dataset instructions
├── screenshots/               # Detector screenshots
├── utils/
│   └── dataset_utils.py       # Dataset utility functions
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## License

MIT License
