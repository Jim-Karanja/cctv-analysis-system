#!/usr/bin/env python3
"""
Setup script for adding authorized personnel to the CCTV Security System.
"""

import cv2
import os
import time
from pathlib import Path

def setup_personnel_directory():
    """Create the personnel directory if it doesn't exist."""
    personnel_dir = Path("data/personnel")
    personnel_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Personnel directory ready: {personnel_dir}")
    return personnel_dir

def capture_personnel_photo(name: str, personnel_dir: Path):
    """Capture a photo of authorized personnel using the camera."""
    print(f"\n📸 Capturing photo for: {name}")
    print("   Position yourself in front of the camera")
    print("   Press SPACE to take photo, ESC to cancel")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return False
    
    # Configure camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            # Show preview
            cv2.putText(frame, f"Capturing: {name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to capture, ESC to cancel", (10, 460), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Personnel Setup - Photo Capture", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to capture
                # Save the photo
                filename = f"{name.lower().replace(' ', '_')}.jpg"
                filepath = personnel_dir / filename
                
                success = cv2.imwrite(str(filepath), frame)
                if success:
                    print(f"✅ Photo saved: {filepath}")
                    cv2.destroyAllWindows()
                    cap.release()
                    return True
                else:
                    print(f"❌ Failed to save photo: {filepath}")
                    
            elif key == 27:  # ESC to cancel
                print("❌ Capture cancelled")
                break
        
    except KeyboardInterrupt:
        print("\n❌ Capture interrupted")
    
    finally:
        cv2.destroyAllWindows()
        cap.release()
    
    return False

def list_personnel(personnel_dir: Path):
    """List current authorized personnel."""
    image_files = list(personnel_dir.glob("*.jpg")) + list(personnel_dir.glob("*.png"))
    
    if not image_files:
        print("📋 No authorized personnel found")
        return
    
    print(f"\n👥 Current Authorized Personnel ({len(image_files)}):")
    for i, filepath in enumerate(image_files, 1):
        name = filepath.stem.replace('_', ' ').title()
        file_size = filepath.stat().st_size
        print(f"   {i}. {name} ({file_size} bytes)")

def main():
    print("🔒 CCTV Security System - Personnel Setup")
    print("=" * 50)
    
    # Setup directory
    personnel_dir = setup_personnel_directory()
    
    # List current personnel
    list_personnel(personnel_dir)
    
    while True:
        print("\n📋 Options:")
        print("1. Add authorized personnel (with camera)")
        print("2. List current personnel")
        print("3. Exit")
        
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                name = input("Enter person's full name: ").strip()
                if not name:
                    print("❌ Name cannot be empty")
                    continue
                
                # Check if person already exists
                filename = f"{name.lower().replace(' ', '_')}.jpg"
                filepath = personnel_dir / filename
                
                if filepath.exists():
                    overwrite = input(f"⚠️  {name} already exists. Overwrite? (y/N): ").strip().lower()
                    if overwrite != 'y':
                        continue
                
                success = capture_personnel_photo(name, personnel_dir)
                if success:
                    print(f"✅ {name} added to authorized personnel")
                else:
                    print(f"❌ Failed to add {name}")
                    
            elif choice == '2':
                list_personnel(personnel_dir)
                
            elif choice == '3':
                break
                
            else:
                print("❌ Invalid choice")
                
        except KeyboardInterrupt:
            print("\n\n👋 Setup cancelled by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n✅ Personnel setup completed")
    print("💡 Next steps:")
    print("   1. Run the security system: python launch_system.py")
    print("   2. Open web interface: http://localhost:8080")
    print("   3. Check that authorized personnel are recognized")

if __name__ == "__main__":
    main()
