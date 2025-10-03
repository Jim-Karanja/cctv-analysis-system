# CCTV Analysis System - Windows Installation Guide (Conda)

## 🚀 Quick Start for Windows Users

This guide will help you set up and launch the CCTV Analysis System on Windows using Anaconda/Miniconda.

## 📋 Prerequisites

### 1. Install Miniconda (Recommended) or Anaconda
- **Option A - Miniconda (Lightweight, Recommended):**
  - Download from: https://docs.conda.io/en/latest/miniconda.html
  - Select "Miniconda3 Windows 64-bit" installer
  
- **Option B - Full Anaconda:**
  - Download from: https://www.anaconda.com/products/distribution
  - Select Windows installer

### 2. System Requirements
- Windows 10/11 (64-bit)
- 8GB RAM (minimum), 16GB+ recommended
- 2GB free disk space
- USB camera or IP camera for testing

## 🛠️ Installation Steps

### Step 1: Open Anaconda Prompt
1. Press `Windows + R`, type `cmd`, press Enter
2. Or search "Anaconda Prompt" in Start Menu and run as Administrator

### Step 2: Clone the Repository
```bash
# Navigate to your desired directory
cd C:\Users\%USERNAME%\
git clone https://github.com/yourusername/cctv-analysis-system.git
cd cctv-analysis-system
```

### Step 3: Create Conda Environment
```bash
# Create a new conda environment with Python 3.9
conda create -n cctv-system python=3.9 -y

# Activate the environment
conda activate cctv-system
```

### Step 4: Install Dependencies via Conda
```bash
# Install core packages from conda-forge
conda install -c conda-forge opencv numpy pillow pyyaml requests -y

# Install additional packages via pip
pip install fastapi==0.103.1
pip install uvicorn[standard]==0.23.2
pip install asyncio-mqtt==0.11.1
pip install aiofiles==23.2.1
pip install pyfcm==1.5.4
pip install python-dotenv==1.0.0
pip install structlog==23.1.0
pip install prometheus-client==0.17.1
pip install click==8.1.7
pip install tqdm==4.66.1
pip install python-dateutil==2.8.2
pip install pytest==7.4.2
```

### Step 5: Configure the System
```bash
# Copy example configuration
copy config\config.example.yaml config\config.yaml

# Open configuration file in notepad for editing
notepad config\config.yaml
```

**Important Configuration Changes:**
- Set `video_sources` → `url` to `0` for USB camera or your IP camera URL
- Adjust `web_interface` → `host` to `127.0.0.1` for local access only
- Modify `logging` → `file` path to use Windows path format

### Step 6: Create Required Directories
```bash
# Create necessary directories
mkdir data
mkdir logs
mkdir uploads
```

## 🚀 Launch the System

### Option 1: Automated Launcher (Recommended)
```bash
# Make sure conda environment is activated
conda activate cctv-system

# Run the system launcher
python launch_system.py
```

This will open two separate command windows:
- **Web Server Window**: Runs the web interface
- **Video Capture Window**: Handles camera processing

### Option 2: Manual Launch (For Debugging)

**Terminal 1 - Web Server:**
```bash
conda activate cctv-system
python improved_web_server.py
```

**Terminal 2 - Video Capture:**
```bash
conda activate cctv-system
python security_video_capture.py
```

## 🌐 Access the System

1. **Web Interface**: Open browser and go to `http://localhost:8080`
2. **API Documentation**: Visit `http://localhost:8080/docs`

## 🔧 Troubleshooting

### Camera Not Detected
```bash
# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.read()[0] else 'Camera Failed'); cap.release()"
```

### Port Already in Use
```bash
# Kill process using port 8080
netstat -ano | findstr :8080
# Note the PID and kill it
taskkill /PID [PID_NUMBER] /F
```

### Missing Visual C++ Redistributable
If you get errors about missing DLLs:
1. Download Visual C++ Redistributable from Microsoft
2. Install both x86 and x64 versions

### OpenCV Issues
```bash
# Reinstall OpenCV if issues occur
conda uninstall opencv -y
conda install -c conda-forge opencv -y
```

## 🏗️ Environment Management

### Save Current Environment
```bash
# Export environment to YAML file
conda env export > environment.yml
```

### Recreate Environment from YAML
```bash
# Create environment from exported file
conda env create -f environment.yml
```

### Update Environment
```bash
# Update all packages
conda activate cctv-system
conda update --all -y
```

### Remove Environment
```bash
# Deactivate and remove environment
conda deactivate
conda env remove -n cctv-system
```

## 🎯 Testing the Installation

### Test 1: Basic System Check
```bash
conda activate cctv-system
python -c "import cv2, fastapi, uvicorn; print('✅ All core modules imported successfully')"
```

### Test 2: Camera Test
```bash
conda activate cctv-system
python -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print('✅ Camera working - Resolution:', frame.shape)
else:
    print('❌ Camera not detected')
cap.release()
"
```

### Test 3: Web Server Test
```bash
conda activate cctv-system
# Start web server (Ctrl+C to stop)
python -c "
import uvicorn
from improved_web_server import app
print('🌐 Starting test web server on http://localhost:8080')
uvicorn.run(app, host='127.0.0.1', port=8080, log_level='info')
"
```

## 🔐 Security Notes

- **Firewall**: Windows may prompt to allow Python through firewall - click "Allow"
- **Network Access**: By default, system runs on localhost only
- **Camera Privacy**: Green LED indicates when camera is active

## 📁 Project Structure
```
cctv-analysis-system/
├── config/
│   ├── config.yaml           # Main configuration
│   └── config.example.yaml   # Example configuration
├── data/                     # Data storage
├── logs/                     # System logs
├── notification_service/     # Alert system
├── launch_system.py          # Main launcher
├── improved_web_server.py    # Web interface
├── security_video_capture.py # Camera processing
└── requirements.txt          # Python dependencies
```

## 🚪 System Shutdown

### Graceful Shutdown
1. Close web browser
2. Press `Ctrl+C` in both terminal windows
3. Close terminal windows

### Force Shutdown
```bash
# Kill all Python processes (use with caution)
taskkill /f /im python.exe
```

## 📞 Getting Help

1. **Check Logs**: Look in `logs/cctv_system.log` for errors
2. **Verbose Mode**: Edit config.yaml and set `logging.level` to `DEBUG`
3. **Test Mode**: Run with single camera first before adding multiple sources

## 🔄 Updates and Maintenance

### Update System Code
```bash
conda activate cctv-system
git pull origin main
pip install -r requirements.txt --upgrade
```

### Clean Cache and Logs
```bash
# Clear Python cache
for /d /r . %d in (__pycache__) do @if exist "%d" rd /s /q "%d"

# Clear old logs (optional)
del /q logs\*.log.1
del /q logs\*.log.2
```

## 🎉 Success Indicators

When everything is working correctly, you should see:
- ✅ Two command windows open automatically
- ✅ Web interface accessible at http://localhost:8080
- ✅ Camera feed visible in web browser
- ✅ No error messages in terminal windows
- ✅ Logs being written to `logs/` directory

---

**🎯 Pro Tip**: Always activate the conda environment (`conda activate cctv-system`) before running any system commands!

**⚠️ Note**: For face recognition features, you may need to install additional packages like `dlib` which can be challenging on Windows. The system will work with basic face detection using OpenCV only.
