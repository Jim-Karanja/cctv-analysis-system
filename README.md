# CCTV Analysis and Notification System

An AI-powered CCTV analysis system that identifies authorized personnel and sends real-time notifications.

## Overview

This system combines computer vision, database management, and notification services to create a comprehensive security monitoring solution. It processes CCTV video feeds in real-time to identify known personnel and trigger appropriate notifications.

## System Architecture

### 1. Video Ingestion & Processing (`video_ingestion/`)
- **Video Source**: CCTV camera feed acquisition
- **Frame Extraction**: Real-time video stream processing
- **Preprocessing**: Frame preparation for AI analysis

### 2. AI & Recognition Engine (`ai_engine/`)
- **Face Detection**: Locate faces within video frames
- **Face Recognition**: Generate unique facial embeddings
- **Person Identification**: Match faces to known personnel database

### 3. Database (`database/`)
- **Personnel Database**: Store registered user profiles and facial embeddings
- **Event Logging**: Record all detection events with timestamps and confidence scores

### 4. Notification Service (`notification_service/`)
- **Event Triggers**: Monitor for specific conditions
- **Mobile Notifications**: Real-time alerts via push notifications

## Technology Stack

- **Computer Vision**: OpenCV, Dlib, face_recognition
- **Database**: PostgreSQL/MongoDB for scalable data storage
- **Notifications**: Firebase Cloud Messaging (FCM)
- **Backend**: Python with asyncio for real-time processing

## Features

- **Dual Mode Operation**: 
  - Basic mode: OpenCV-only face detection (works on any platform)
  - Advanced mode: Full face recognition with dlib (recommended for Ubuntu)
- **Real-time Processing**: Asynchronous video stream processing
- **Web Interface**: FastAPI-based dashboard at http://localhost:8080
- **Multiple Camera Support**: USB cameras, IP cameras, RTSP streams
- **Flexible Configuration**: YAML-based configuration system

## Installation

### Windows (Basic Mode - Detection Only)

1. Clone the repository:
```bash
git clone <repository-url>
cd cctv-analysis-system
```

2. Install basic dependencies:
```bash
pip install opencv-python fastapi uvicorn pyyaml numpy
```

3. Configure the system:
```bash
copy config\config.example.yaml config\config.yaml
# Edit config.yaml with your camera settings
```

### Ubuntu (Advanced Mode - Full Recognition)

1. Install system dependencies:
```bash
sudo apt update
sudo apt install -y git python3-venv build-essential cmake python3-dev
```

2. Clone and setup:
```bash
git clone <repository-url>
cd cctv-analysis-system
python3 -m venv .venv
source .venv/bin/activate
```

3. Install Python dependencies:
```bash
pip install opencv-python fastapi uvicorn pyyaml numpy
# For full face recognition:
pip install dlib face_recognition
```

4. Configure the system:
```bash
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your camera settings
```

## Usage

1. Start the main application:
```bash
python main.py
```

2. Access the web interface at `http://localhost:8080`

## Configuration

Edit `config/config.yaml` to customize:
- Camera sources and settings
- Database connection details
- Notification service credentials
- AI model parameters

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Security Considerations

- All facial data is encrypted at rest
- Network communications use TLS encryption
- Access controls implemented for all system components
- Regular security audits recommended

## License

[Your License Here]

## Contributing

[Contributing Guidelines Here]
