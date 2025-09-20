#!/usr/bin/env python3
"""
Example usage script for the YOLO + Picamera2 detector.
This script shows different ways to run the detector with various configurations.
"""

import subprocess
import sys
import time

def run_command(cmd, description):
    """Run a command and print its description."""
    print(f"\n{'='*60}")
    print(f"Example: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    # Ask user if they want to run this example
    response = input("Run this example? (y/N): ").strip().lower()
    if response == 'y':
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}")
        except KeyboardInterrupt:
            print("\nExample interrupted by user")
    else:
        print("Skipping example...")

def main():
    print("YOLO + Picamera2 Detector - Usage Examples")
    print("==========================================")
    print("\nNote: These examples require a Raspberry Pi with Camera Module 3")
    print("Make sure picamera2 and libcamera are installed:")
    print("  sudo apt install python3-picamera2 espeak-ng")
    
    examples = [
        {
            "cmd": ["python3", "yolo_picamera2_detector.py", "--help"],
            "desc": "Show all available command line options"
        },
        {
            "cmd": ["python3", "yolo_picamera2_detector.py"],
            "desc": "Basic detection with default settings (yolo11n.pt model, cv2 preview)"
        },
        {
            "cmd": ["python3", "yolo_picamera2_detector.py", "--model", "yolov8n.pt", "--conf", "0.5"],
            "desc": "Use YOLOv8n model with higher confidence threshold"
        },
        {
            "cmd": ["python3", "yolo_picamera2_detector.py", "--preview", "none", "--duration", "30"],
            "desc": "Headless mode - run for 30 seconds without display"
        },
        {
            "cmd": ["python3", "yolo_picamera2_detector.py", "--save", "--duration", "60"],
            "desc": "Record video while detecting for 1 minute"
        },
        {
            "cmd": ["python3", "yolo_picamera2_detector.py", "--imgsz", "416", "--conf", "0.25"],
            "desc": "Fast detection with smaller image size and lower confidence"
        }
    ]
    
    for example in examples:
        run_command(example["cmd"], example["desc"])
    
    print(f"\n{'='*60}")
    print("Interactive Controls (when preview is enabled):")
    print("  q - Quit the application")
    print("  s - Save snapshot to disk")
    print("  f - Trigger autofocus")
    print("  m - Toggle between macro and normal focus modes")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()