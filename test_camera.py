"""
Test script to verify laptop camera integration with the CCTV system.
"""

import cv2
import time
import yaml
from pathlib import Path

def test_camera():
    """Test laptop camera functionality."""
    print("🎥 Testing laptop camera...")
    
    # Test camera access
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open laptop camera")
        print("💡 Try changing the camera index in config.yaml (url: 1 instead of url: 0)")
        return False
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"✅ Camera opened successfully!")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    
    print("\n📸 Press 'q' to quit the camera preview")
    print("🔍 Look for face detection rectangles around detected faces")
    
    # Load a basic face detector for demo
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        frame_count += 1
        
        # Add face detection for demo
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'Face Detected', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add system info overlay
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        
        cv2.putText(frame, f'CCTV Analysis System - Test Mode', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'FPS: {current_fps:.1f} | Faces: {len(faces)}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f'Press Q to quit', (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the frame
        cv2.imshow('CCTV System - Laptop Camera Test', frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Auto-exit after 30 seconds for demo
        if elapsed > 30:
            print("🎬 Demo completed (30 seconds)")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 Test completed:")
    print(f"   Total frames: {frame_count}")
    print(f"   Duration: {elapsed:.2f} seconds")
    print(f"   Average FPS: {frame_count/elapsed:.2f}")
    
    return True

def check_config():
    """Check if config is set up for laptop camera."""
    config_path = Path("config/config.yaml")
    
    if not config_path.exists():
        print("❌ Config file not found!")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    video_sources = config.get('video_sources', [])
    laptop_camera = None
    
    for source in video_sources:
        if source.get('source_id') == 'laptop_camera' and source.get('enabled'):
            laptop_camera = source
            break
    
    if laptop_camera:
        print("✅ Config is set up for laptop camera")
        print(f"   Camera URL: {laptop_camera['url']}")
        print(f"   Resolution: {laptop_camera['resolution']}")
        print(f"   FPS: {laptop_camera['fps']}")
        return True
    else:
        print("❌ Laptop camera not configured or not enabled in config.yaml")
        return False

def main():
    print("🚀 CCTV Analysis System - Laptop Camera Test")
    print("=" * 50)
    
    # Check configuration
    if not check_config():
        print("\n💡 To fix this, ensure your config/config.yaml has:")
        print("""
video_sources:
  - source_id: 'laptop_camera'
    url: 0  # Use 1 if 0 doesn't work
    fps: 30
    resolution: [1280, 720]
    enabled: true
""")
        return
    
    print()
    
    # Test camera
    if test_camera():
        print("\n🎉 Success! Your laptop camera is working with the CCTV system.")
        print("\n🔧 Next steps:")
        print("1. Run the main system: python main.py")
        print("2. Access web interface: http://localhost:8080")
        print("3. Add personnel to database for recognition")
    else:
        print("\n❌ Camera test failed. Check your camera permissions and try a different camera index.")

if __name__ == "__main__":
    main()
