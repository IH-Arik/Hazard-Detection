from ultralytics import YOLO

# Load the model
model = YOLO("https://huggingface.co/yihong1120/Construction-Hazard-Detection/resolve/main/models/yolo11/pt/yolo11n.pt")

# Test image
source = 'http://images.cocodataset.org/val2017/000000039769.jpg'

# Run detection
results = model(source, conf=0.25)

# Check all detections
print("All detections found:")
for result in results:
    for box in result.boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        confidence = float(box.conf)
        print(f"  Class {class_id}: {class_name} (confidence: {confidence:.2f})")

# Test filtering
ALLOWED_CLASSES = {0, 1, 2, 3, 4, 5, 8}  # Current filter
print(f"\nAllowed classes: {ALLOWED_CLASSES}")
print("\nFiltered detections:")
for result in results:
    for box in result.boxes:
        class_id = int(box.cls)
        if class_id in ALLOWED_CLASSES:
            class_name = model.names[class_id]
            confidence = float(box.conf)
            print(f"  Class {class_id}: {class_name} (confidence: {confidence:.2f})")
        else:
            class_name = model.names[class_id]
            print(f"  FILTERED OUT Class {class_id}: {class_name}")
