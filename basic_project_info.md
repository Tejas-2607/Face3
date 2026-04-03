# 🎥 Face Recognition Surveillance System

An intelligent vision-based surveillance system that can automatically detect, recognize, and interact with people in a camera feed based on natural language commands.

## 📋 Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Command Syntax](#command-syntax)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [API Documentation](#api-documentation)

## ✨ Features

- **Natural Language Commands**: Control the system using simple English commands
- **Real-time Face Recognition**: Identify people in live camera feeds
- **Spatial Awareness**: Detect people relative to others (left, right, first, second, etc.)
- **Web Interface**: User-friendly Flask-based web dashboard
- **Dataset Management**: Easy capture, view, and manage face datasets
- **Snapshot Capture**: Save moments with automatic timestamping
- **Flexible Camera Support**: Works with webcams and CCTV (RTSP) streams
- **High Accuracy**: Powered by InsightFace (ArcFace embeddings)

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Flask Web Interface                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Dataset    │  │   Generate   │  │ Recognition  │      │
│  │   Capture    │─▶│  Embeddings  │─▶│   & Query    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│              Command Processing (NLP Layer)                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Parse → Extract → Validate → Execute                  │ │
│  └────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│            Face Detection & Recognition Engine               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │ InsightFace│─▶│  Embedding │─▶│   Cosine   │           │
│  │  Detection │  │ Generation │  │ Similarity │           │
│  └────────────┘  └────────────┘  └────────────┘           │
├─────────────────────────────────────────────────────────────┤
│                  Camera Input Layer                          │
│         Webcam / USB Camera / RTSP CCTV Stream              │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam or CCTV camera
- 4GB+ RAM recommended

### Step 1: Clone or Download

```bash
# Download all project files to a directory
cd face-recognition-system
```

### Step 2: Install Dependencies

```bash
pip install opencv-python insightface onnxruntime numpy scikit-learn flask
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import cv2, insightface, flask; print('✓ All dependencies installed')"
```

## 🚀 Quick Start

### 1. Capture Face Dataset

Run the capture script to collect face images:

```bash
python capture_faces.py
```

**Instructions:**
- Enter the person's name when prompted (e.g., "User1", "Alice")
- Look at the camera and move your head slowly
- Script captures 50 images automatically
- Press 'q' to quit early

**⚠️ IMPORTANT**: Before running, check the camera configuration at the top of `capture_faces.py`:

```python
# Line 13-18 in capture_faces.py
CAMERA_INDEX = 0  # ⚠️ CHANGE THIS ACCORDING TO YOUR DEVICE
```

Common values:
- `0` = Built-in laptop webcam
- `1` = External USB webcam  
- `"rtsp://user:pass@ip:port/stream"` = CCTV camera

### 2. Generate Embeddings

After capturing datasets, generate the face embeddings:

**Option A - Web Interface:**
1. Start the Flask app: `python app.py`
2. Open browser to `http://localhost:5000`
3. Click "Generate Embeddings" button

**Option B - Command Line:**
```bash
python generate_embeddings.py
```

### 3. Start Recognition

#### Web Interface (Recommended)

```bash
python app.py
```

Then open your browser to: `http://localhost:5000`

Navigate to the "Recognition" page and enter commands like:
- `detect person right to User1`
- `find second person on left of Alice`

#### Standalone Script

```bash
python recognize_faces.py
```

**⚠️ IMPORTANT**: Before running, check camera configuration in `app.py`:

```python
# Line 35-40 in app.py
CAMERA_INDEX = 0  # ⚠️ CHANGE THIS ACCORDING TO YOUR DEVICE
```

## 📖 Usage Guide

### Capturing Datasets

**Tips for best results:**

1. **Lighting**: Ensure even, bright lighting on face
2. **Angles**: Slowly rotate head (left, right, up, down)
3. **Expressions**: Include neutral, smiling, serious faces
4. **Accessories**: Capture with/without glasses if worn regularly
5. **Quantity**: Minimum 30-50 images per person

### Command Syntax

**General Format:**
```
[action] [position] person [direction] of [reference_person]
```

**Components:**

- **Action** (optional): `detect`, `find`, `show`, `capture`
- **Position** (optional): `first`, `second`, `third`, `fourth` (default: first)
- **Direction** (required): `left`, `right`
- **Reference Person** (required): Must match dataset name exactly (case-sensitive)

**Examples:**

```bash
# Detect first person on the right of User1
detect person right to User1

# Find the second person on the left of Alice
find second person on left of Alice

# Show third person to the right of Bob
show third person right of Bob

# Capture snapshot of person left of User1
capture person left to User1
```

### Understanding the Display

**Visual Indicators:**

- **Green Box** = Anchor person (reference person from command)
- **Magenta Box** = Target person (person matching command)
- **Blue Box** = Other detected faces
- **Green Text** = Success message
- **Red Text** = Error/not found message

**Status Messages:**

- `✓ [Name] detected` = Target person found successfully
- `❌ [Name] not detected` = Reference person not in frame
- `❌ No person on LEFT/RIGHT` = No one on specified side
- `❌ Only N person(s) on LEFT/RIGHT` = Not enough people

## ⚙️ Configuration

### Camera Configuration

**For Webcams:**
```python
CAMERA_INDEX = 0  # Built-in camera
CAMERA_INDEX = 1  # External USB camera
```

**For CCTV (RTSP):**
```python
CAMERA_INDEX = "rtsp://username:password@192.168.1.100:554/stream"
```

### Recognition Thresholds

Edit in `app.py`:

```python
RECOGNITION_THRESHOLD = 0.65  # Higher = stricter matching (0.0 - 1.0)
MIN_DETECTION_CONFIDENCE = 0.5  # Face detection confidence
```

### File Paths

Default paths (can be customized in `app.py`):

```python
DATASET_PATH = "dataset"                      # Face images
EMBEDDINGS_PATH = "embeddings/face_embeddings.pkl"  # Embeddings DB
SNAPSHOTS_PATH = "snapshots"                  # Captured snapshots
```

## 🔧 Troubleshooting

### Camera Not Opening

**Problem**: `Could not open camera`

**Solutions:**
1. Check `CAMERA_INDEX` value in your script
2. Try different indices: 0, 1, 2
3. Verify camera is not used by another application
4. For Linux: Check camera permissions
5. Test with: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

### No Face Detected

**Problem**: Face not being detected during capture

**Solutions:**
1. Ensure good lighting
2. Face the camera directly
3. Lower `MIN_CONFIDENCE` threshold
4. Check camera is not obscured
5. Move closer to camera

### Low Recognition Accuracy

**Problem**: People not being recognized correctly

**Solutions:**
1. Capture more images (50+ recommended)
2. Include varied angles and expressions
3. Regenerate embeddings after adding images
4. Lower `RECOGNITION_THRESHOLD` (try 0.55-0.60)
5. Ensure good lighting in both capture and recognition

### Command Not Working

**Problem**: Commands not being parsed correctly

**Solutions:**
1. Check reference person name matches dataset exactly (case-sensitive)
2. Include direction keyword (left/right)
3. Follow command format: `[action] [position] person [direction] of [name]`
4. Check examples on web interface

### Port Already in Use

**Problem**: Flask cannot start (port 5000 in use)

**Solution:**
```python
# In app.py, change port number
app.run(debug=True, host='0.0.0.0', port=5001)  # Changed to 5001
```

## 📡 API Documentation

### Endpoints

#### `POST /api/set_command`
Set the current recognition command.

**Request:**
```json
{
  "command": "detect person right to User1"
}
```

**Response:**
```json
{
  "success": true,
  "message": "✓ Detecting first person on right of User1",
  "command": {
    "action": "detect",
    "direction": "right",
    "position": 1,
    "reference_person": "User1",
    "valid": true
  }
}
```

#### `POST /api/clear_command`
Clear the current command.

**Response:**
```json
{
  "success": true,
  "message": "Command cleared"
}
```

#### `POST /api/generate_embeddings`
Generate embeddings from dataset.

**Response:**
```json
{
  "success": true,
  "message": "Generated 150 embeddings (5 failed)",
  "count": 150
}
```

#### `POST /api/capture_snapshot`
Capture current frame as snapshot.

**Response:**
```json
{
  "success": true,
  "message": "Snapshot saved",
  "filename": "snapshot_20240214_153045.jpg",
  "path": "snapshots/snapshot_20240214_153045.jpg"
}
```

#### `GET /api/system_status`
Get current system status.

**Response:**
```json
{
  "embeddings_loaded": true,
  "embeddings_count": 150,
  "dataset_persons": 3,
  "persons": ["User1", "Alice", "Bob"],
  "camera_active": true,
  "current_command": {...}
}
```

#### `DELETE /api/delete_person/<person_name>`
Delete a person's dataset.

**Response:**
```json
{
  "success": true,
  "message": "User1 deleted"
}
```

## 📁 Project Structure

```
face-recognition-system/
│
├── app.py                          # Flask web application
├── capture_faces.py                # Dataset capture script
├── generate_embeddings.py          # Embedding generation script
├── command_parsing_enhanced.py     # Enhanced NLP parser
├── recognize_faces.py              # Standalone recognition (optional)
├── command_parsing.py              # Legacy parser (optional)
│
├── templates/                      # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── capture.html
│   ├── recognize.html
│   └── manage.html
│
├── static/                         # Static assets
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
│
├── dataset/                        # Face images (created automatically)
│   ├── User1/
│   ├── Alice/
│   └── Bob/
│
├── embeddings/                     # Generated embeddings
│   └── face_embeddings.pkl
│
├── snapshots/                      # Captured snapshots
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Voice command integration
- Multi-camera support
- Mobile app interface
- Advanced NLP with intent recognition
- Performance optimizations for real-time processing

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- **InsightFace** - Face detection and recognition
- **OpenCV** - Computer vision operations
- **Flask** - Web framework
- **scikit-learn** - Similarity calculations