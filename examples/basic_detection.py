"""
Basic example of hazard detection usage
"""

import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from src.utils.detection_utils import draw_custom_bboxes, filter_detections, calculate_detection_summary
from src.config.config import Config


def main():
    """Main function for basic detection example"""
    
    # Load construction hazard detection model
    print("Loading construction hazard detection model...")
    model = YOLO(Config.CONSTRUCTION_HAZARD_MODEL)
    
    # Load PPE detection model
    print("Loading PPE detection model...")
    ppe_model_path = hf_hub_download(
        repo_id=Config.PPE_DETECTION_MODEL_REPO, 
        filename=Config.PPE_DETECTION_MODEL_FILE
    )
    ppe_model = YOLO(ppe_model_path)
    
    # Example image path (replace with your image path)
    image_path = "data/images/sample.jpg"
    
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return
        
        print(f"Processing image: {image_path}")
        
        # Run construction hazard detection
        print("Running construction hazard detection...")
        hazard_results = model(image, conf=Config.DEFAULT_CONFIDENCE_THRESHOLD)
        filtered_boxes, detection_info = filter_detections(hazard_results, model)
        
        # Draw annotations
        if filtered_boxes:
            annotated_image = draw_custom_bboxes(image, filtered_boxes, model)
        else:
            annotated_image = image.copy()
        
        # Calculate and display summary
        summary = calculate_detection_summary(detection_info)
        print("\n=== Construction Hazard Detection Results ===")
        print(f"Total detections: {summary['total_detections']}")
        
        for class_name, count in summary['class_counts'].items():
            avg_conf = summary['class_confidences'][class_name]
            print(f"{class_name}: {count} detections (avg confidence: {avg_conf:.2f})")
        
        # Save annotated image
        output_path = "data/images/hazard_detection_result.jpg"
        cv2.imwrite(output_path, annotated_image)
        print(f"\nAnnotated image saved to: {output_path}")
        
        # Run PPE detection
        print("\nRunning PPE detection...")
        ppe_results = ppe_model(image, conf=Config.DEFAULT_CONFIDENCE_THRESHOLD)
        
        print(f"PPE detections: {len(ppe_results[0].boxes) if ppe_results else 0}")
        
        # Save PPE detection result
        if ppe_results and len(ppe_results[0].boxes) > 0:
            ppe_annotated = ppe_results[0].plot()
            ppe_output_path = "data/images/ppe_detection_result.jpg"
            cv2.imwrite(ppe_output_path, ppe_annotated)
            print(f"PPE annotated image saved to: {ppe_output_path}")
        
    except Exception as e:
        print(f"Error during detection: {e}")


if __name__ == "__main__":
    main()
