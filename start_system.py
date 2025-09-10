#!/usr/bin/env python3
"""
CCTV Analysis System Startup Script

Quick startup script that checks dependencies and starts the system.
"""

import sys
import subprocess
import yaml
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    core_modules = ['cv2', 'fastapi', 'uvicorn', 'yaml', 'numpy']
    optional_modules = ['face_recognition', 'dlib']
    
    missing_core = []
    missing_optional = []
    
    for module in core_modules:
        try:
            __import__(module)
        except ImportError:
            missing_core.append(module)
    
    for module in optional_modules:
        try:
            __import__(module)
        except ImportError:
            missing_optional.append(module)
    
    if missing_core:
        print(f"❌ Missing core dependencies: {', '.join(missing_core)}")
        print("💡 Install them with: pip install opencv-python fastapi uvicorn pyyaml numpy")
        return False
    
    if missing_optional:
        print(f"⚠️  Missing optional dependencies: {', '.join(missing_optional)}")
        print("💡 Face recognition will be disabled. Install with: pip install face-recognition")
        print("   (Note: face-recognition requires CMake and Visual Studio Build Tools on Windows)")
        print("✅ Core dependencies are installed - running in basic mode")
    else:
        print("✅ All dependencies are installed")
    
    return True

def check_config():
    """Check if configuration is set up correctly."""
    config_path = Path("config/config.yaml")
    
    if not config_path.exists():
        print("❌ Configuration file not found!")
        print("💡 Copy config.example.yaml to config.yaml and customize it")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check for enabled video sources
        video_sources = config.get('video_sources', [])
        enabled_sources = [s for s in video_sources if s.get('enabled', False)]
        
        if not enabled_sources:
            print("❌ No enabled video sources found in config.yaml")
            print("💡 Enable at least one video source")
            return False
        
        print(f"✅ Configuration valid with {len(enabled_sources)} enabled camera(s)")
        return True
        
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")
        return False

def create_directories():
    """Create required directories."""
    dirs = ["data", "logs"]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    print("✅ Required directories created")

def main():
    """Main startup function."""
    print("🚀 CCTV Analysis System Startup")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check configuration
    if not check_config():
        return 1
    
    # Create directories
    create_directories()
    
    print("\n🎯 Starting CCTV Analysis System...")
    print("   - Video processing with laptop camera")
    print("   - Face detection and recognition")
    print("   - Web interface at http://127.0.0.1:8080")
    print("   - Press Ctrl+C to stop")
    print("\n" + "=" * 40)
    
    try:
        # Start the main application
        subprocess.run([sys.executable, "main.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ System error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
