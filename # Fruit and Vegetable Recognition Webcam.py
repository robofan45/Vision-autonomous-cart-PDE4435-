# Fruit and Vegetable Recognition Webcam Test with YOLOv8
from ultralytics import YOLO
import cv2
import time
import os
import glob

print("YOLOv8 Fruit and Vegetable Recognition Test")
print("=========================================")

# Find the model
model_path = None

# Check if model_path.txt exists
if os.path.exists(r"C:\Yolo\runs\model_path.txt"):
    with open(r"C:\Yolo\runs\model_path.txt", 'r') as f:
        model_path = f.read().strip()
        if os.path.exists(model_path):
            print(f"Using model from model_path.txt: {model_path}")
        else:
            model_path = None

# If not found, search common locations
if not model_path:
    possible_paths = [
        r"C:\Yolo\runs\detect\train\weights\best.pt",  # YOLOv8 default path
        r"C:\Yolo\runs\detect\train\weights\last.pt",
        r"C:\Yolo\runs\fruits_veg_detection\weights\best.pt",
        r"C:\Yolo\model\fruits_veg\weights\best.pt",
        r"C:\Yolo\runs\detect\fruits_vegetables_yolo\weights\best.pt",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            print(f"Found model at: {model_path}")
            break
    
    # If still not found, search for any .pt files
    if not model_path:
        pt_files = glob.glob(r"C:\Yolo\\**\*.pt", recursive=True)
        
        for path in pt_files:
            if os.path.getsize(path) > 1000000:  # Files larger than 1MB
                model_path = path
                print(f"Found potential model at: {model_path}")
                break

# If still not found, ask user
if not model_path:
    print("Model not found. Please enter the full path to your model file:")
    model_path = input("> ")
    if not os.path.exists(model_path):
        print(f"Error: File not found at {model_path}")
        exit()

# Load the model
print(f"\nLoading model from: {model_path}")
try:
    model = YOLO(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Open webcam
print("\nInitializing webcam...")
cap = cv2.VideoCapture(0)  # Try camera 0 first

# If can't open camera 0, try camera 1
if not cap.isOpened():
    print("Could not open default camera (index 0), trying camera 1...")
    cap = cv2.VideoCapture(1)

# If still can't open, show error
if not cap.isOpened():
    print("Error: Could not open any webcam.")
    print("Make sure your webcam is connected and not being used by another application.")
    exit()

# Set lower resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Webcam initialized successfully!")
print("Press 'q' to quit, 's' to save a screenshot")
print("Press '+' or '-' to adjust confidence threshold")

# Create directory for screenshots
os.makedirs("fruit_veg_screenshots", exist_ok=True)

# Configure model settings
conf_threshold = 0.4  # Confidence threshold

# Counter for total items and prices
total_items = 0
total_price = 0.0

# Sample price dictionary (replace with your actual fruits and vegetables)
price_dict = {
    "apple": 1.20,
    "banana": 0.50,
    "orange": 0.80,
    "tomato": 0.75,
    "carrot": 0.60,
    "broccoli": 1.50,
    "potato": 0.30,
    "cucumber": 0.90,
    "lemon": 0.70,
    "strawberry": 2.50,
    # Add more items as needed
}

# Main detection loop
try:
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from webcam")
            break
        
        # Run detection with YOLOv8
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Calculate inference time
        inference_time = results[0].speed['inference']
        
        # Draw results on frame (YOLOv8 way)
        annotated_frame = results[0].plot()
        
        # Add FPS information
        fps = 1000 / inference_time if inference_time > 0 else 0
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Get detection information
        boxes = results[0].boxes
        detection_count = len(boxes)
        
        # Add detection count
        cv2.putText(annotated_frame, f"Items: {detection_count}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Process detections and calculate prices
        if detection_count > 0:
            class_counts = {}
            cart_price = 0.0
            
            for box in boxes:
                cls_id = int(box.cls[0].item())  # Get class ID as a Python integer
                conf = float(box.conf[0].item())  # Get confidence score
                cls_name = model.names[cls_id]
                
                if cls_name in class_counts:
                    class_counts[cls_name] += 1
                else:
                    class_counts[cls_name] = 1
                
                # Add item price to cart if available
                if cls_name.lower() in price_dict:
                    cart_price += price_dict[cls_name.lower()]
            
            # Display class counts and calculate prices
            y_pos = 120
            for cls_name, count in class_counts.items():
                item_price = 0.0
                if cls_name.lower() in price_dict:
                    item_price = price_dict[cls_name.lower()] * count
                    price_text = f"{cls_name}: {count} (${item_price:.2f})"
                else:
                    price_text = f"{cls_name}: {count}"
                
                cv2.putText(annotated_frame, price_text, (20, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_pos += 40
            
            # Display cart total
            cv2.putText(annotated_frame, f"CART TOTAL: ${cart_price:.2f}", 
                        (frame.shape[1] - 300, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the confidence threshold
        cv2.putText(annotated_frame, f"Conf: {conf_threshold:.2f}", (20, annotated_frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Show the result
        cv2.imshow("YOLOv8 Fruit and Vegetable Recognition", annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'q' to quit
        if key == ord('q'):
            break
        # Press 's' to save a screenshot
        elif key == ord('s'):
            screenshot_path = os.path.join("fruit_veg_screenshots", f"cart_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(screenshot_path, annotated_frame)
            print(f"Cart screenshot saved to {screenshot_path}")
        # Press '+' to increase confidence threshold
        elif key == ord('+') or key == ord('='):
            conf_threshold = min(0.95, conf_threshold + 0.05)
            print(f"Confidence threshold increased to {conf_threshold:.2f}")
        # Press '-' to decrease confidence threshold
        elif key == ord('-'):
            conf_threshold = max(0.05, conf_threshold - 0.05)
            print(f"Confidence threshold decreased to {conf_threshold:.2f}")
        # Press 'c' to clear cart
        elif key == ord('c'):
            total_items = 0
            total_price = 0.0
            print("Cart cleared")

except KeyboardInterrupt:
    print("Test interrupted by user")
except Exception as e:
    print(f"Error during testing: {e}")
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("\nFruit and Vegetable recognition test completed")