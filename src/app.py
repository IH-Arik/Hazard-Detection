import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time

# Page title and description
st.set_page_config(
    page_title="Construction Hazard Detection",
    page_icon="🚧",
    layout="wide"
)

st.title("🚧 Construction Hazard Detection")
st.markdown("Upload an image or video to detect construction hazards using YOLO11 model.")

# Load model
@st.cache_resource
def load_model():
    model = YOLO("https://huggingface.co/yihong1120/Construction-Hazard-Detection/resolve/main/models/yolo11/pt/yolo11n.pt")
    return model

# Filter classes - only keep classes 0, 1, 2, 3, 4, 5, 7
ALLOWED_CLASSES = {0, 1, 2, 3, 4, 5, 7}  # Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Vest

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Sidebar for options
st.sidebar.header("Detection Options")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
video_processing = st.sidebar.selectbox("Processing Mode", ["Frame by Frame", "Summary Only"])

# Input options
st.header("Input Source")
input_option = st.radio(
    "Choose input source:",
    ["Upload File", "Webcam", "Image URL"]
)

if input_option == "Upload File":
    uploaded_file = st.file_uploader(
        "Choose an image or video...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp', 'mp4', 'avi', 'mov', 'mkv']
    )
    image_url = None
elif input_option == "Webcam":
    uploaded_file = None
    image_url = None
elif input_option == "Image URL":
    uploaded_file = None
    image_url = st.text_input("Enter image URL:", "http://images.cocodataset.org/val2017/000000039769.jpg")

def process_image(image_path, confidence_threshold):
    """Process single image and return results"""
    results = model(image_path, conf=confidence_threshold)
    
    for result in results:
        # Filter detections to only keep allowed classes
        filtered_boxes = []
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id in ALLOWED_CLASSES:
                filtered_boxes.append(box)
        
        # Create custom annotated image with color-coded bounding boxes
        if filtered_boxes:
            annotated_image = draw_custom_bboxes(result.orig_img.copy(), filtered_boxes)
        else:
            annotated_image = result.orig_img.copy()
        
        # Convert BGR to RGB for display
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        return annotated_image_rgb, filtered_boxes
    
    return None, []

def draw_custom_bboxes(image, boxes):
    """Draw bounding boxes with custom colors"""
    # Define colors: green for proper safety equipment, red for violations, light blue for person
    green_color = (0, 255, 0)    # Green in BGR
    red_color = (0, 0, 255)      # Red in BGR
    lightblue_color = (255, 255, 0)  # Light blue in BGR (Yellow in BGR = Light blue in RGB)
    
    # Classes with color mapping
    green_classes = {0, 1, 7}  # Hardhat, Mask, Safety Vest
    person_class = {5}             # Person
    
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls)
        class_name = model.names[class_id]
        confidence = float(box.conf)
        
        # Choose color based on class
        if class_id in green_classes:
            color = green_color
            label_color = green_color
        elif class_id in person_class:
            color = lightblue_color
            label_color = lightblue_color
        else:
            color = red_color
            label_color = red_color
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"{class_name}: {confidence:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), label_color, -1)
        
        # Draw label text
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return image

def process_webcam(confidence_threshold):
    """Process webcam feed in real-time"""
    st.subheader("📹 Live Webcam Detection")
    
    # Instructions
    st.info("📹 Click 'Start Webcam' below to begin live detection. Click 'Stop' to end.")
    
    # Create placeholders
    frame_placeholder = st.empty()
    summary_placeholder = st.empty()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Unable to access webcam. Please check your camera permissions.")
        return
    
    # Control buttons
    start_button = st.button("📹 Start Webcam", type="primary")
    stop_button = st.button("⏹ Stop Webcam")
    
    # Storage for recent detections
    recent_detections = []
    webcam_active = False
    
    if start_button:
        webcam_active = True
        st.success("📹 Webcam started! Close this tab or click 'Stop' to end.")
    
    if stop_button:
        webcam_active = False
        st.warning("📹 Webcam stopped.")
        cap.release()
        return
    
    # Process frames if webcam is active
    if webcam_active:
        try:
            for _ in range(100):  # Process 100 frames max to avoid infinite loop
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                results = model(frame, conf=confidence_threshold)
                
                # Filter detections
                filtered_boxes = []
                frame_detections = []
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        if class_id in ALLOWED_CLASSES:
                            filtered_boxes.append(box)
                            frame_detections.append({
                                'class': model.names[class_id],
                                'confidence': float(box.conf)
                            })
                
                # Annotate frame with custom colors
                if filtered_boxes:
                    annotated_frame = draw_custom_bboxes(frame.copy(), filtered_boxes)
                else:
                    annotated_frame = frame.copy()
                
                # Convert BGR to RGB for display
                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                frame_placeholder.image(annotated_rgb, channels="RGB", width=800)
                
                # Update recent detections
                recent_detections.extend(frame_detections)
                if len(recent_detections) > 100:  # Keep only recent detections
                    recent_detections = recent_detections[-100:]
                
                # Update summary every 10 frames
                if len(recent_detections) % 10 == 0 and len(recent_detections) > 0:
                    update_webcam_summary(recent_detections, summary_placeholder)
                
                # Small delay to control frame rate
                time.sleep(0.1)
                
            st.info("📹 Webcam session completed. Click 'Start Webcam' to begin again.")
                
        except Exception as e:
            st.error(f"Error in webcam processing: {e}")
        finally:
            cap.release()
    else:
        # Show initial frame
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", width=800)
            st.info("📹 Click 'Start Webcam' to begin detection.")
        cap.release()

def update_webcam_summary(detections, placeholder):
    """Update webcam detection summary"""
    if not detections:
        return
    
    # Count detections by class
    class_counts = {}
    class_confidences = {}
    
    for det in detections:
        class_name = det['class']
        confidence = det['confidence']
        
        if class_name not in class_counts:
            class_counts[class_name] = 0
            class_confidences[class_name] = []
        
        class_counts[class_name] += 1
        class_confidences[class_name].append(confidence)
    
    # Display summary
    with placeholder.container():
        st.subheader("📊 Recent Detections (Last 100 frames)")
        for class_name, count in class_counts.items():
            avg_conf = np.mean(class_confidences[class_name])
            st.write(f"**{class_name}**: {count} detections (avg confidence: {avg_conf:.2f})")
        st.write(f"**Total Recent Detections**: {len(detections)}")

def process_video(video_path, confidence_threshold, processing_mode):
    """Process video file and return results"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    st.info(f"Video Info: {total_frames} frames, {fps} FPS, {width}x{height} resolution")
    
    # Create progress bar
    progress_bar = st.progress(0)
    frame_placeholder = st.empty()
    summary_placeholder = st.empty()
    
    # Storage for all detections
    all_detections = []
    processed_frames = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            results = model(frame, conf=confidence_threshold)
            
            frame_detections = []
            for result in results:
                # Filter detections
                filtered_boxes = []
                for box in result.boxes:
                    class_id = int(box.cls)
                    if class_id in ALLOWED_CLASSES:
                        filtered_boxes.append(box)
                        frame_detections.append({
                            'class': model.names[class_id],
                            'confidence': float(box.conf),
                            'frame': processed_frames
                        })
                
                # Annotate frame with custom colors
                if filtered_boxes:
                    annotated_frame = draw_custom_bboxes(frame.copy(), filtered_boxes)
                else:
                    annotated_frame = frame.copy()
            
            # Update progress
            progress = (processed_frames / total_frames)
            progress_bar.progress(progress)
            
            # Display frame if in frame-by-frame mode
            if processing_mode == "Frame by Frame":
                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_rgb, caption=f"Frame {processed_frames}/{total_frames}", width=800)
                time.sleep(1/fps)  # Play at normal speed
            
            all_detections.extend(frame_detections)
            processed_frames += 1
            
            # Update summary periodically
            if processed_frames % 10 == 0:
                update_detection_summary(all_detections, summary_placeholder)
    
    finally:
        cap.release()
    
    # Final summary
    update_detection_summary(all_detections, summary_placeholder, final=True)
    
    return all_detections

def update_detection_summary(detections, placeholder, final=False):
    """Update detection summary display"""
    if not detections:
        placeholder.info("No hazards detected yet.")
        return
    
    # Count detections by class
    class_counts = {}
    class_confidences = {}
    
    for det in detections:
        class_name = det['class']
        confidence = det['confidence']
        
        if class_name not in class_counts:
            class_counts[class_name] = 0
            class_confidences[class_name] = []
        
        class_counts[class_name] += 1
        class_confidences[class_name].append(confidence)
    
    # Display summary
    with placeholder.container():
        if final:
            st.subheader("📊 Final Detection Summary")
        else:
            st.subheader("📊 Current Detection Summary")
        
        for class_name, count in class_counts.items():
            avg_conf = np.mean(class_confidences[class_name])
            st.write(f"**{class_name}**: {count} detections (avg confidence: {avg_conf:.2f})")
        
        st.write(f"**Total Detections**: {len(detections)}")

# Handle different input sources
if input_option == "Webcam":
    if st.button("📹 Start Webcam", type="primary"):
        process_webcam(confidence_threshold)

elif input_option == "Upload File" and uploaded_file is not None:
    # Check if it's a video file
    is_video = uploaded_file.type.startswith('video/')
    
    if is_video:
        st.header("🎥 Video Processing")
        
        if st.button("🎬 Process Video", type="primary"):
            # Save uploaded video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            try:
                with st.spinner("Processing video..."):
                    detections = process_video(video_path, confidence_threshold, video_processing)
                st.success(f"✅ Video processing complete! Found {len(detections)} total detections.")
            except Exception as e:
                st.error(f"Error processing video: {e}")
            finally:
                os.unlink(video_path)
    
    else:
        # Image processing
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=600)
            
            # Save uploaded image to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name)
                image_path = tmp_file.name
        
        with col2:
            st.subheader("Detection Results")
            
            if st.button("🔍 Detect Hazards", type="primary"):
                with st.spinner("Detecting hazards..."):
                    try:
                        annotated_image, detections = process_image(image_path, confidence_threshold)
                        
                        if annotated_image is not None:
                            st.image(annotated_image, caption="Detection Results", width=600)
                        
                        # Display detection information
                        st.subheader("📊 Detection Summary")
                        
                        if len(detections) > 0:
                            # Count detections by class
                            class_counts = {}
                            for box in detections:
                                class_name = model.names[int(box.cls)]
                                confidence = float(box.conf)
                                
                                if class_name not in class_counts:
                                    class_counts[class_name] = []
                                class_counts[class_name].append(confidence)
                            
                            # Display results
                            for class_name, confidences in class_counts.items():
                                avg_conf = np.mean(confidences)
                                count = len(confidences)
                                st.write(f"**{class_name}**: {count} detected (avg confidence: {avg_conf:.2f})")
                            
                            # Total detections
                            st.write(f"**Total Hazards Detected**: {len(detections)}")
                        else:
                            st.info("No hazards detected with current confidence threshold.")
                        
                        # Clean up temporary file
                        os.unlink(image_path)
                            
                    except Exception as e:
                        st.error(f"Error during detection: {e}")

elif input_option == "Image URL" and image_url:
    # Image processing from URL
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        try:
            st.image(image_url, caption="Image from URL", width=600)
            image_path = image_url
        except Exception as e:
            st.error(f"Error loading image from URL: {e}")
            st.stop()
    
    with col2:
        st.subheader("Detection Results")
        
        if st.button("🔍 Detect Hazards", type="primary"):
            with st.spinner("Detecting hazards..."):
                try:
                    annotated_image, detections = process_image(image_path, confidence_threshold)
                    
                    if annotated_image is not None:
                        st.image(annotated_image, caption="Detection Results", width=600)
                    
                    # Display detection information
                    st.subheader("📊 Detection Summary")
                    
                    if len(detections) > 0:
                        # Count detections by class
                        class_counts = {}
                        for box in detections:
                            class_name = model.names[int(box.cls)]
                            confidence = float(box.conf)
                            
                            if class_name not in class_counts:
                                class_counts[class_name] = []
                            class_counts[class_name].append(confidence)
                        
                        # Display results
                        for class_name, confidences in class_counts.items():
                            avg_conf = np.mean(confidences)
                            count = len(confidences)
                            st.write(f"**{class_name}**: {count} detected (avg confidence: {avg_conf:.2f})")
                        
                        # Total detections
                        st.write(f"**Total Hazards Detected**: {len(detections)}")
                    else:
                        st.info("No hazards detected with current confidence threshold.")
                        
                except Exception as e:
                    st.error(f"Error during detection: {e}")

# Instructions
st.markdown("---")
st.markdown("### 📋 Instructions")
st.markdown("""
1. **Choose input source**: Upload file, use webcam, or provide image URL
2. **Adjust confidence threshold** using the slider (lower = more detections, higher = more accurate)
3. **For videos**: Choose processing mode (Frame by Frame or Summary Only)
4. **Click action button**: "Start Webcam", "Detect Hazards", or "Process Video"
5. **View results** with color-coded bounding boxes and labels

### 📹 Webcam Features:
- **Real-time detection**: Live hazard detection from your camera
- **Color-coded boxes**: Green for safety equipment, red for violations, light blue for people
- **Recent summary**: Shows detection statistics from last 100 frames
- **Automatic cleanup**: Manages memory efficiently

### 🎥 Video Features:
- **Frame by Frame**: See each processed frame with detections
- **Summary Only**: Get final statistics without frame-by-frame display
- **Progress tracking**: Real-time progress bar and detection counts
- **Multiple formats**: MP4, AVI, MOV, MKV supported

### 🏗️ Hazard Classes Detected:
- **🟢 Hardhat**: Workers wearing hard hats (Green box)
- **🟢 Mask**: Workers wearing masks (Green box)
- **🔴 NO-Mask**: Workers without masks (Red box)
- **🔴 NO-Hardhat**: Workers without hard hats (Red box)
- **🔴 NO-Safety Vest**: Workers without safety vests (Red box)
- **🔵 Person**: People in the construction area (Light blue box)
- **🟢 Safety Vest**: Workers wearing safety vests (Green box)

### 🎨 Color Coding:
- **🟢 Green Boxes**: Proper safety equipment (Hardhat, Mask, Safety Vest)
- **🔴 Red Boxes**: Safety violations (NO-Hardhat, NO-Mask, NO-Safety Vest)
- **🔵 Light Blue Boxes**: People detected (Person)

*Note: Classes Safety Cone, machinery, and vehicle are filtered out as requested*
""")

# Footer
st.markdown("---")
st.markdown("Powered by [YOLO11](https://github.com/ultralytics/ultralytics) and [Construction Hazard Detection Model](https://huggingface.co/yihong1120/Construction-Hazard-Detection)")
