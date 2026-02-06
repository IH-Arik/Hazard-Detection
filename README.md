# 🚧 Hazard Detection System

A comprehensive AI-powered construction hazard and Personal Protective Equipment (PPE) detection system built with YOLO models and Streamlit.

## 🌟 Features

### 🏗️ Construction Hazard Detection
- Real-time detection of construction hazards using YOLO11
- Multiple input sources: Upload files, webcam, or image URLs
- Support for both images and videos
- Color-coded bounding boxes for different hazard types
- Confidence threshold adjustment
- Frame-by-frame or summary-only video processing

### � PPE Detection
- Personal Protective Equipment detection using specialized models
- Detection of safety equipment: hard hats, masks, safety vests
- Integration with Hugging Face model hub
- High accuracy with configurable confidence thresholds

### � Key Capabilities
- **Real-time Processing**: Live webcam detection with real-time annotations
- **Batch Processing**: Video file processing with progress tracking
- **Multi-format Support**: Images (JPG, PNG, BMP, WebP) and Videos (MP4, AVI, MOV, MKV)
- **Interactive UI**: Modern Streamlit interface with intuitive controls
- **Detection Statistics**: Comprehensive detection summaries and analytics

## � Project Structure

```
hazard-detection/
├── src/                          # Source code
│   ├── app.py                    # Main Streamlit application
│   ├── detetion.py               # PPE detection module
│   ├── models/                   # Model weights and configurations
│   ├── utils/                    # Utility functions
│   └── config/                   # Configuration files
├── scripts/                      # Helper scripts
│   ├── check_classes.py          # Model class inspection
│   └── check_actual_classes.py   # Class verification
├── tests/                        # Test files
├── data/                         # Data directories
│   ├── images/                   # Sample images
│   └── videos/                   # Sample videos
├── docs/                         # Documentation
├── examples/                     # Example usage
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/IH-Arik/Hazard-Detection.git
   cd Hazard-Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run src/app.py
   ```

The application will open in your web browser at `http://localhost:8501`

### 📦 Dependencies

The project requires the following main packages:
- `streamlit` - Web application framework
- `ultralytics` - YOLO model implementation
- `opencv-python` - Computer vision operations
- `Pillow` - Image processing
- `numpy` - Numerical computations
- `huggingface_hub` - Model hub integration

## 🎮 Usage Guide

### 1. **Choose Input Source**
- **Upload File**: Select images or videos from your device
- **Webcam**: Use your camera for real-time detection
- **Image URL**: Provide a URL to an online image

### 2. **Adjust Settings**
- **Confidence Threshold**: Slider to adjust detection sensitivity (0.0-1.0)
- **Processing Mode**: For videos, choose between frame-by-frame or summary-only

### 3. **Run Detection**
- Click the appropriate action button based on your input source
- View results with color-coded bounding boxes and labels

### 4. **Interpret Results**
- 🟢 **Green Boxes**: Proper safety equipment (Hardhat, Mask, Safety Vest)
- 🔴 **Red Boxes**: Safety violations (NO-Hardhat, NO-Mask, NO-Safety Vest)
- 🔵 **Light Blue Boxes**: People detected (Person)

## 🏷️ Detected Classes

### Construction Hazard Detection
- **Hardhat**: Workers wearing hard hats
- **Mask**: Workers wearing masks
- **NO-Hardhat**: Workers without hard hats
- **NO-Mask**: Workers without masks
- **NO-Safety Vest**: Workers without safety vests
- **Person**: People in construction areas
- **Safety Vest**: Workers wearing safety vests

### PPE Detection
- Additional PPE items detected via specialized model
- Enhanced accuracy for safety equipment identification

## 🔧 Configuration

### Model Configuration
Models are automatically downloaded from:
- Construction Hazard: `yihong1120/Construction-Hazard-Detection`
- PPE Detection: `melihuzunoglu/ppe-detection`

### Customization
You can modify:
- Confidence thresholds in the UI
- Allowed classes in `src/app.py`
- Color schemes in the detection functions
- Model URLs in the configuration

## 🧪 Testing

Run the helper scripts to verify model functionality:

```bash
# Check available classes
python scripts/check_classes.py

# Verify actual model classes
python scripts/check_actual_classes.py
```

## 📊 Performance

- **Real-time Processing**: ~30 FPS on modern hardware
- **Accuracy**: High precision for PPE detection
- **Memory Usage**: Optimized for efficient processing
- **Supported Formats**: Multiple image and video formats

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Development Roadmap

- [ ] Add more PPE detection classes
- [ ] Implement model training scripts
- [ ] Add API endpoints
- [ ] Mobile application support
- [ ] Advanced analytics dashboard
- [ ] Multi-language support

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure stable internet connection for model downloads
   - Check if all dependencies are properly installed

2. **Webcam Access Issues**
   - Verify camera permissions
   - Check if other applications are using the camera

3. **Memory Issues**
   - Reduce video resolution for large files
   - Close other memory-intensive applications

### Getting Help

- Open an issue on GitHub for bug reports
- Check the documentation in the `docs/` folder
- Review example scripts in the `examples/` folder

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO implementation
- [Streamlit](https://streamlit.io/) for the web framework
- [Hugging Face](https://huggingface.co/) for model hosting
- The open-source community for various tools and libraries

## 📞 Contact

- **Project Maintainer**: IH Arik
- **GitHub**: [@IH-Arik](https://github.com/IH-Arik)
- **Project URL**: [https://github.com/IH-Arik/Hazard-Detection](https://github.com/IH-Arik/Hazard-Detection)

---

⭐ If this project helps you, please give it a star!
