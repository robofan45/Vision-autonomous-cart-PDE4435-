# YOLOv8 Fruit and Vegetable Recognition

Real-time fruit and vegetable detection with price calculation using YOLOv8.

## Features

- Real-time webcam detection of fruits and vegetables (using standard USB/built-in cameras)
- **NEW**: Raspberry Pi Camera Module 3 support with autofocus (`yolo_picamera2_detector.py`)
- Price calculation for the shopping cart
- Adjustable confidence threshold
- Screenshot capture
- Audio feedback (Pi camera script)
- Headless operation support (Pi camera script)
- Custom training with Roboflow datasets

## Dataset

Location: `C:\Yolo\dataset fruits and vegetables`

Format: YOLOv8 (train/valid folders with images and labels)
link of the dataset
https://www.kaggle.com/datasets/kvnpatel/fruits-vegetable-detection-for-yolov4
## Requirements

For standard webcam detection:
```
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.20.0
PyYAML>=6.0
matplotlib>=3.5.0
```

For Raspberry Pi Camera Module 3 (additional requirements):
```bash
# Install on Raspberry Pi OS
sudo apt install python3-picamera2 espeak-ng
```

## Script Comparison

| Feature | Webcam Script | Pi Camera Script |
|---------|---------------|------------------|
| **Target Platform** | Any system with webcam | Raspberry Pi with Camera Module |
| **Camera Support** | USB/Built-in webcams | RPi Camera Module 3 with autofocus |
| **Price Calculation** | ✅ Shopping cart with prices | ❌ Object detection only |
| **Audio Feedback** | ❌ | ✅ espeak-ng announcements |
| **Autofocus Control** | ❌ | ✅ Continuous + manual trigger |
| **Headless Operation** | ❌ | ✅ Automatic fallback |
| **Video Recording** | ❌ | ✅ MP4 output |
| **Model Support** | Custom fruit/veg models | YOLO11n/YOLOv8n with fallback |

Choose the **webcam script** for shopping cart applications with price calculation.
Choose the **Pi camera script** for advanced computer vision applications on Raspberry Pi.

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

### Standard Webcam Detection

```bash
# Run webcam detector
python "# Fruit and Vegetable Recognition Webcam.py"

# Controls:
# q - Quit
# s - Save screenshot
# +/- - Adjust confidence threshold
# c - Clear cart
```

### Raspberry Pi Camera Module 3 Detection

```bash
# Run Pi camera detector with autofocus support
python yolo_picamera2_detector.py

# Command line options:
python yolo_picamera2_detector.py --model yolo11n.pt --conf 0.35 --preview cv2
python yolo_picamera2_detector.py --preview none --duration 60 --save  # headless mode

# Controls (when preview enabled):
# q - Quit
# s - Save snapshot
# f - Trigger autofocus
# m - Toggle macro/normal focus mode

# Features:
# - Autofocus support for RPi Camera Module 3
# - Audio feedback with object announcements
# - Headless operation fallback
# - Video recording support
# - Automatic model fallback (yolo11n.pt -> yolov8n.pt)
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
Vision-autonomous-cart-PDE4435-/
├── # Fruit and Vegetable Recognition Webcam.py  # Webcam detection script
├── yolo_picamera2_detector.py                    # RPi Camera Module 3 script with autofocus
├── requirements.txt                               # Project dependencies
└── README.md                                      # This file

# Optional directories (created during runtime):
├── fruit_veg_screenshots/    # Screenshots from webcam detector
├── snapshots/               # Snapshots from Pi camera detector  
└── out.mp4                 # Video output from Pi camera detector
```

## Troubleshooting

### Raspberry Pi Camera Issues

1. **Camera not detected**:
   ```bash
   # Enable camera interface
   sudo raspi-config
   # Navigate to Interface Options > Camera > Enable
   
   # Check camera connection
   libcamera-hello --list-cameras
   ```

2. **Permission errors**:
   ```bash
   # Add user to video group
   sudo usermod -a -G video $USER
   # Logout and login again
   ```

3. **Autofocus not working**:
   - Ensure you have RPi Camera Module 3 (not v1 or v2)
   - Check that libcamera is up to date:
     ```bash
     sudo apt update && sudo apt upgrade
     ```

4. **Audio feedback not working**:
   ```bash
   # Install espeak-ng
   sudo apt install espeak-ng
   
   # Test audio
   espeak-ng "Hello world"
   ```

### General Issues

1. **ModuleNotFoundError**:
   ```bash
   # Install missing dependencies
   pip install -r requirements.txt
   
   # For Pi-specific modules
   sudo apt install python3-picamera2
   ```

2. **Model not found**:
   - The script will automatically download yolo11n.pt or yolov8n.pt
   - For custom models, use: `--model /path/to/your/model.pt`

## License

MIT License
