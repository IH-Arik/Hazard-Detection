# API Documentation

## Overview

The Hazard Detection System provides a comprehensive API for construction hazard and PPE detection using YOLO models.

## Core Components

### 1. Detection Models

#### ConstructionHazardDetector
Main class for construction hazard detection using YOLO11.

```python
from src.models.construction_detector import ConstructionHazardDetector

detector = ConstructionHazardDetector()
results = detector.detect(image_path, confidence=0.25)
```

#### PPEDetector
Class for Personal Protective Equipment detection.

```python
from src.models.ppe_detector import PPEDetector

ppe_detector = PPEDetector()
results = ppe_detector.detect(image_path, confidence=0.25)
```

### 2. Utility Functions

#### Detection Utils
Located in `src/utils/detection_utils.py`

- `draw_custom_bboxes(image, boxes, model)`: Draw color-coded bounding boxes
- `filter_detections(results, model)`: Filter detections by allowed classes
- `calculate_detection_summary(detections)`: Calculate detection statistics
- `validate_file_format(filename, file_type)`: Validate file formats
- `resize_image(image, max_size)`: Resize image maintaining aspect ratio

### 3. Configuration

#### Config Class
Located in `src/config/config.py`

```python
from src.config.config import Config

# Access configuration settings
model_url = Config.CONSTRUCTION_HAZARD_MODEL
confidence = Config.DEFAULT_CONFIDENCE_THRESHOLD
colors = Config.COLORS
```

## Model Classes

### Construction Hazard Detection Classes

| Class ID | Class Name | Description | Color |
|----------|------------|-------------|-------|
| 0 | Hardhat | Workers wearing hard hats | Green |
| 1 | Mask | Workers wearing masks | Green |
| 2 | NO-Hardhat | Workers without hard hats | Red |
| 3 | NO-Mask | Workers without masks | Red |
| 4 | NO-Safety Vest | Workers without safety vests | Red |
| 5 | Person | People in construction areas | Light Blue |
| 7 | Safety Vest | Workers wearing safety vests | Green |

### PPE Detection Classes

The PPE detection model includes additional classes for enhanced safety equipment detection.

## Usage Examples

### Basic Image Detection

```python
import cv2
from src.models.construction_detector import ConstructionHazardDetector
from src.utils.detection_utils import draw_custom_bboxes

# Load detector
detector = ConstructionHazardDetector()

# Load image
image = cv2.imread("path/to/image.jpg")

# Run detection
results = detector.detect(image, confidence=0.25)

# Draw results
annotated_image = draw_custom_bboxes(image, results['boxes'], detector.model)

# Save result
cv2.imwrite("result.jpg", annotated_image)
```

### Video Processing

```python
import cv2
from src.models.construction_detector import ConstructionHazardDetector

detector = ConstructionHazardDetector()
cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = detector.detect(frame, confidence=0.25)
    annotated_frame = draw_custom_bboxes(frame, results['boxes'], detector.model)
    
    cv2.imshow("Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Webcam Detection

```python
import cv2
from src.models.construction_detector import ConstructionHazardDetector

detector = ConstructionHazardDetector()
cap = cv2.VideoCapture(0)  # Use webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = detector.detect(frame, confidence=0.25)
    annotated_frame = draw_custom_bboxes(frame, results['boxes'], detector.model)
    
    cv2.imshow("Live Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Streamlit Application

### Main Application

The main application is located in `src/app.py` and provides a web interface for:

- Image upload and processing
- Video file processing
- Real-time webcam detection
- Confidence threshold adjustment
- Detection statistics and summaries

### Running the Application

```bash
streamlit run src/app.py
```

## Error Handling

### Common Exceptions

- `ModelLoadError`: Failed to load YOLO model
- `InvalidImageError`: Invalid or corrupted image file
- `UnsupportedFormatError`: Unsupported file format
- `DetectionError`: Error during detection process

### Error Handling Example

```python
try:
    detector = ConstructionHazardDetector()
    results = detector.detect(image_path)
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
except InvalidImageError as e:
    print(f"Invalid image: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Considerations

### Optimization Tips

1. **Image Size**: Resize large images before processing
2. **Confidence Threshold**: Adjust based on accuracy requirements
3. **Batch Processing**: Process multiple images in batches when possible
4. **GPU Acceleration**: Use CUDA-enabled PyTorch for faster processing

### Memory Management

- Clear detection results after processing
- Use appropriate image resolutions
- Monitor memory usage during video processing

## Configuration

### Environment Variables

- `HAZARD_DETECTION_MODEL_DIR`: Directory to store downloaded models
- `HAZARD_DETECTION_CONFIDENCE`: Default confidence threshold
- `HAZARD_DETECTION_MAX_SIZE`: Maximum image size

### Custom Configuration

```python
from src.config.config import Config

# Override default settings
Config.DEFAULT_CONFIDENCE_THRESHOLD = 0.3
Config.MAX_IMAGE_SIZE = 1280
```

## Testing

### Unit Tests

Run unit tests using pytest:

```bash
pytest tests/
```

### Test Coverage

Generate test coverage report:

```bash
pytest --cov=src tests/
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
EXPOSE 8501

CMD ["streamlit", "run", "src/app.py"]
```

### Cloud Deployment

The application can be deployed to:
- Streamlit Cloud
- Heroku
- AWS
- Google Cloud Platform
- Azure

## Contributing

### Code Style

Follow PEP 8 guidelines. Use black for code formatting:

```bash
black src/
```

### Testing

Ensure all tests pass before submitting pull requests:

```bash
pytest tests/
```

### Documentation

Update documentation for any new features or API changes.
