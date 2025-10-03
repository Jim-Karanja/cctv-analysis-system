#!/usr/bin/env python3
"""
Simple launcher to start both the web server and video capture in separate Konsole windows (KDE).
"""

import subprocess
import time
from pathlib import Path

def main():
    print("🚀 CCTV Analysis System - Konsole Window Launcher")
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
    if Path("threaded_security_video.py").exists():
        video_script = "threaded_security_video.py"
    else:
        print("❌ No video capture script found!")
        return False

    print(f"📹 Using video script: {video_script}")
    print()

    try:
        print("🌐 Starting web server in new Konsole window...")
        subprocess.Popen(["konsole", "-e", "python3", web_script])
        time.sleep(2)  # Give web server time to start

        print("🎥 Starting video capture in new Konsole window...")
        subprocess.Popen(["konsole", "-e", "python3", video_script])

        print()
        print("🎉 Both systems started in separate Konsole windows!")
        print("🔗 Web server: http://localhost:8080")
        print("📊 Check both Konsole windows for status")
        print("⏹️  Close the Konsole windows to stop the services")
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
