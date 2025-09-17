#!/usr/bin/env python3
"""
Test Face Recognition Scores

Quick test to see what scores we get when matching your face.
"""

import cv2
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG)

def test_face_matching():
    """Test face recognition with your personnel photo."""
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load personnel photo
    personnel_dir = Path("data/personnel")
    personnel_files = list(personnel_dir.glob("*.jpg")) + list(personnel_dir.glob("*.png"))
    
    if not personnel_files:
        print("❌ No personnel photos found!")
        return
    
    personnel_file = personnel_files[0]
    print(f"📸 Loading personnel photo: {personnel_file.name}")
    
    # Load and process personnel photo
    personnel_img = cv2.imread(str(personnel_file))
    personnel_gray = cv2.cvtColor(personnel_img, cv2.COLOR_BGR2GRAY)
    
    # Detect face in personnel photo
    personnel_faces = face_cascade.detectMultiScale(personnel_gray, 1.1, 5, minSize=(50, 50))
    
    if len(personnel_faces) == 0:
        print("❌ No face found in personnel photo!")
        return
    
    # Get the largest face
    if len(personnel_faces) > 1:
        personnel_faces = sorted(personnel_faces, key=lambda x: x[2]*x[3], reverse=True)
    
    px, py, pw, ph = personnel_faces[0]
    personnel_template = cv2.resize(personnel_gray[py:py+ph, px:px+pw], (100, 100))
    
    print(f"✅ Extracted face template from {personnel_file.stem}")
    
    # Test with camera
    print("📹 Testing with camera - show your face to see recognition scores...")
    print("Press 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                # Extract and resize face
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (100, 100))
                
                # Match against personnel template
                result = cv2.matchTemplate(face_resized, personnel_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                # Determine color and label based on threshold
                threshold = 0.3
                if max_val > threshold:
                    color = (0, 255, 0)  # Green
                    label = f"AUTHORIZED: {personnel_file.stem} ({max_val:.3f})"
                else:
                    color = (0, 0, 255)  # Red
                    label = f"UNKNOWN: ({max_val:.3f})"
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Print score to console
                print(f"Match score: {max_val:.3f} ({'AUTHORIZED' if max_val > threshold else 'UNKNOWN'})")
            
            cv2.imshow('Face Recognition Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_face_matching()
