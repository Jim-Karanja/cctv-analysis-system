#!/usr/bin/env python3
"""
Simple launcher to start both the web server and video capture in separate windows.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def main():
    print("🚀 CCTV Analysis System - Window Launcher")
    print("=" * 50)
    
    # Check if files exist and prioritize improved web server
    web_script = None
    if Path("improved_web_server.py").exists():
        web_script = "improved_web_server.py"
    elif Path("web_server.py").exists():
        web_script = "web_server.py"
    else:
        print("❌ No web server script found!")
        return False
    
    print(f"🌐 Using web server script: {web_script}")
    
    video_script = None
    # Prioritize threaded implementations for better performance
    if Path("threaded_security_video.py").exists():
        video_script = "threaded_security_video.py"
    elif Path("basic_security_video.py").exists():
        video_script = "basic_security_video.py"
    elif Path("security_video_capture.py").exists():
        video_script = "security_video_capture.py"
    elif Path("simple_video_capture.py").exists():
        video_script = "simple_video_capture.py"
    else:
        print("❌ No video capture script found!")
        return False
    
    print(f"📹 Using video script: {video_script}")
    print()
    
    # For Windows, start both in separate command windows
    try:
        print("🌐 Starting web server in new window...")
        os.system(f'start "CCTV Web Server" cmd /k "python {web_script}"')
        time.sleep(2)  # Give web server time to start
        
        print("🎥 Starting video capture in new window...")
        os.system(f'start "CCTV Video Capture" cmd /k "python {video_script}"')
        
        print()
        print("🎉 Both systems started in separate windows!")
        print("🔗 Web server: http://localhost:8080")
        print("📊 Check both terminal windows for status")
        print("⏹️  Close the terminal windows to stop the services")
        print()
        print("✅ Setup complete - both processes are now running independently")
        return True
        
    except Exception as e:
        print(f"❌ Error starting services: {e}")
        return False

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
