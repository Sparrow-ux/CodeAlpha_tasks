# Handwriting Classification Web App

## Overview

This project provides a simple web application that classifies handwritten characters from an uploaded image. It uses a pre‑trained **CNN** model built with **PyTorch** and leverages **OpenCV** for image preprocessing. The Flask server exposes a single page where users can upload an image, view bounding‑box visualisation, and get the predicted text.

## Features

- Upload an image containing handwritten text.
- Automatic contour detection and character segmentation.
- Real‑time prediction of each character using a neural network.
- Visual feedback with bounding boxes and recognised characters overlaid on the original image.

## Quick Start

1. **Clone the repository** (or copy the folder into your workspace).
2. **Create a virtual environment** (recommended) and install the dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Run the Flask app**:
   ```bash
   python app.py
   ```
4. Open your browser and navigate to `http://127.0.0.1:5000`. Upload an image and watch the predictions appear.

## Project Structure

```
Codealpha_HandWriting_Classification/
│   app.py                # Flask entry point, prediction logic
│   requirements.txt      # Python dependencies
│   README.md             # This document
│   model/
│       handwriting_model.pth   # Pre‑trained model weights
│   static/
│       uploads/               # Uploaded images (runtime)
│   templates/
│       index.html             # HTML front‑end
│   cnn.py               # CNN model definition (imported by app.py)
```

## Dependencies

All required Python packages are listed in `requirements.txt`. The core libraries are:

- **Flask** – lightweight web framework.
- **opencv-python** – image processing and contour detection.
- **torch** – deep learning framework.
- **torchvision** – transforms for tensor conversion.

## Customising the Model

The model architecture lives in `model/cnn.py`. To train your own model, replace `model/handwriting_model.pth` with a new checkpoint and adjust the loading path in `app.py` if necessary.

## License

This project is provided for educational purposes. Feel free to modify and use it in your own applications.
