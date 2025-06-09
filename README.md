# 🤖 AI-Powered Object Detection Chatbot

A modern chatbot application that combines real-time object detection with a chat interface. This project uses state-of-the-art AI models to detect and identify objects in your camera feed while providing a user-friendly chat experience.

## 🌟 Features

- 🎥 Real-time object detection using DETR (DEtection TRansformer)
- 💬 Modern chat interface with dark theme
- 📊 Confidence score display for detections
- 🎯 Bounding box visualization
- 🔄 Continuous object tracking
- 🎨 Beautiful GUI using Tkinter

## 🛠️ Technologies Used

- **DETR Model** (`facebook/detr-resnet-50`): State-of-the-art object detection model
- **OpenCV**: Camera handling and image processing
- **Tkinter**: GUI development
- **Transformers**: Hugging Face's library for AI models
- **PyTorch**: Deep learning framework
- **PIL**: Image processing

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam
- Internet connection (for first-time model download)

## 🚀 Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
```

3. Activate the virtual environment:
- Windows:
```bash
.venv\Scripts\activate
```
- Linux/Mac:
```bash
source .venv/bin/activate
```

4. Install required packages:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Run the chatbot:
```bash
python chatbot.py
```

2. Using the Application:
   - Type messages in the input field at the bottom
   - Click "Start Camera" to begin object detection
   - Watch as objects are detected and displayed in the chat
   - The video feed will show bounding boxes around detected objects

## 🔍 Object Detection

The application can detect 91 different types of objects including:
- 🚗 Vehicles (cars, trucks, buses)
- 🐱 Animals (dogs, cats, birds)
- 🏠 Buildings and structures
- 🍎 Food items
- 👥 People
- And many more!

## ⚙️ Configuration

The main configuration parameters are:
- Confidence threshold: 0.5 (50%)
- Camera resolution: 640x480
- Model: DETR-ResNet-50

## 🤝 Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a new branch
3. Making your changes
4. Submitting a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Facebook AI Research for the DETR model
- Hugging Face for the Transformers library
- OpenCV community for computer vision tools

## 🆘 Support

If you encounter any issues or have questions:
1. Check the existing issues
2. Create a new issue with detailed information
3. Contact the maintainers

## 📈 Future Improvements

- [ ] Add support for multiple languages
- [ ] Implement object tracking
- [ ] Add custom model training
- [ ] Support for multiple cameras
- [ ] Export detection results

---
Made with ❤️ by [Your Name] 