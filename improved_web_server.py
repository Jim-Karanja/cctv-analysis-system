#!/usr/bin/env python3
"""
Improved Web Server for CCTV Analysis System with MJPEG streaming

Features:
- Proper MJPEG streaming instead of static image refreshing
- Frame buffering for consistency
- Adaptive quality control
- Better error handling
"""

import asyncio
import json
import logging
import signal
import sys
import time
import os
import io
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import queue

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CCTV Analysis System", description="Real-time CCTV monitoring dashboard")

# Shared data files
CURRENT_FRAME_FILE = "data/current_frame.jpg"
FRAME_BUFFER_SIZE = 10
TARGET_FPS = 15
MIN_QUALITY = 60
MAX_QUALITY = 90
DEFAULT_QUALITY = 85

# Adaptive quality controller
class QualityController:
    def __init__(self):
        self.current_quality = DEFAULT_QUALITY
        self.frame_times = []
        self.max_frame_time_history = 10
        self.last_adjustment = 0
        self.adjustment_cooldown = 5.0  # seconds
    
    def update_performance(self, frame_time):
        """Update performance metrics and adjust quality if needed."""
        self.frame_times.append(frame_time)
        
        # Keep only recent frame times
        if len(self.frame_times) > self.max_frame_time_history:
            self.frame_times.pop(0)
        
        current_time = time.time()
        if current_time - self.last_adjustment > self.adjustment_cooldown:
            self._adjust_quality()
            self.last_adjustment = current_time
    
    def _adjust_quality(self):
        """Adjust quality based on recent performance."""
        if len(self.frame_times) < 5:
            return
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        target_frame_time = 1.0 / TARGET_FPS
        
        if avg_frame_time > target_frame_time * 1.5:
            # Performance is poor, reduce quality
            if self.current_quality > MIN_QUALITY:
                self.current_quality = max(MIN_QUALITY, self.current_quality - 10)
                logger.info(f"Reduced quality to {self.current_quality}% due to poor performance")
        
        elif avg_frame_time < target_frame_time * 0.8:
            # Performance is good, increase quality
            if self.current_quality < MAX_QUALITY:
                self.current_quality = min(MAX_QUALITY, self.current_quality + 5)
                logger.info(f"Increased quality to {self.current_quality}% due to good performance")
    
    def get_quality(self):
        """Get current quality setting."""
        return self.current_quality

# Global quality controller
quality_controller = QualityController()

# Global frame buffer for consistent streaming
class FrameBuffer:
    def __init__(self, max_size=FRAME_BUFFER_SIZE):
        self.frames = queue.Queue(maxsize=max_size)
        self.lock = threading.Lock()
        self.last_frame = None
        self.last_update = 0
        
    def put_frame(self, frame_data, timestamp=None):
        """Add a frame to the buffer."""
        if timestamp is None:
            timestamp = time.time()
            
        try:
            # Remove old frame if buffer is full
            if self.frames.full():
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
            
            self.frames.put((frame_data, timestamp), block=False)
            
            with self.lock:
                self.last_frame = frame_data
                self.last_update = timestamp
                
        except queue.Full:
            # Buffer is full, skip this frame
            pass
    
    def get_frame(self, timeout=1.0):
        """Get the most recent frame from buffer."""
        try:
            # Get the most recent frame by emptying the queue
            latest_frame = None
            while not self.frames.empty():
                try:
                    latest_frame = self.frames.get_nowait()
                except queue.Empty:
                    break
            
            if latest_frame:
                return latest_frame[0], latest_frame[1]
            
            # If no new frames, return the last known frame
            with self.lock:
                if self.last_frame is not None:
                    return self.last_frame, self.last_update
                    
            return None, 0
            
        except Exception as e:
            logger.error(f"Error getting frame from buffer: {e}")
            return None, 0
    
    def get_latest_frame_sync(self):
        """Get the latest frame synchronously (for static serving)."""
        with self.lock:
            if self.last_frame is not None:
                return self.last_frame, self.last_update
        return None, 0

# Global frame buffer
frame_buffer = FrameBuffer()

class FrameWatcher:
    """Watches for new frames and updates the buffer."""
    
    def __init__(self, frame_buffer):
        self.frame_buffer = frame_buffer
        self.running = False
        self.watcher_thread = None
        
    def start(self):
        """Start the frame watcher thread."""
        if not self.running:
            self.running = True
            self.watcher_thread = threading.Thread(target=self._watch_frames, daemon=True)
            self.watcher_thread.start()
            logger.info("Frame watcher started")
    
    def stop(self):
        """Stop the frame watcher thread."""
        self.running = False
        if self.watcher_thread and self.watcher_thread.is_alive():
            self.watcher_thread.join(timeout=2)
        logger.info("Frame watcher stopped")
    
    def _watch_frames(self):
        """Main frame watching loop."""
        last_modified = 0
        
        while self.running:
            try:
                if os.path.exists(CURRENT_FRAME_FILE):
                    current_modified = os.path.getmtime(CURRENT_FRAME_FILE)
                    
                    if current_modified > last_modified:
                        # New frame available
                        frame = cv2.imread(CURRENT_FRAME_FILE)
                        if frame is not None:
                            # Encode frame as JPEG with adaptive quality
                            current_quality = quality_controller.get_quality()
                            encode_param = [cv2.IMWRITE_JPEG_QUALITY, current_quality]
                            success, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
                            
                            if success:
                                self.frame_buffer.put_frame(encoded_frame.tobytes(), current_modified)
                                
                        last_modified = current_modified
                
                time.sleep(1.0 / TARGET_FPS)  # Control frame rate
                
            except Exception as e:
                logger.error(f"Error in frame watcher: {e}")
                time.sleep(1)

# Global frame watcher
frame_watcher = FrameWatcher(frame_buffer)

# Ensure data directory exists
Path("data").mkdir(exist_ok=True)

def create_placeholder_frame():
    """Create a placeholder frame when no video is available."""
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add gradient background
    for y in range(480):
        placeholder[y, :] = [40 + y//10, 40 + y//10, 40 + y//10]
    
    # Add text
    cv2.putText(placeholder, 'CCTV Security System', (120, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(placeholder, 'No Video Stream Available', (150, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(placeholder, f'Waiting for video...', (200, 320), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    
    return placeholder

async def generate_mjpeg_stream():
    """Generate MJPEG stream for consistent video delivery with adaptive quality."""
    frame_interval = 1.0 / TARGET_FPS
    last_frame_time = 0
    
    while True:
        try:
            stream_start_time = time.time()
            current_time = stream_start_time
            
            # Control frame rate
            if current_time - last_frame_time < frame_interval:
                await asyncio.sleep(0.01)
                continue
            
            # Get frame from buffer
            frame_data, timestamp = frame_buffer.get_frame()
            current_quality = quality_controller.get_quality()
            
            if frame_data is None:
                # No frame available, create placeholder
                placeholder = create_placeholder_frame()
                encode_param = [cv2.IMWRITE_JPEG_QUALITY, max(70, current_quality - 15)]
                success, encoded_frame = cv2.imencode('.jpg', placeholder, encode_param)
                
                if success:
                    frame_data = encoded_frame.tobytes()
                else:
                    continue
            
            # Check if frame is too old (more than 5 seconds)
            elif current_time - timestamp > 5.0:
                placeholder = create_placeholder_frame()
                cv2.putText(placeholder, 'Video stream paused...', (180, 360), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)
                cv2.putText(placeholder, f'Quality: {current_quality}%', (200, 400), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                           
                encode_param = [cv2.IMWRITE_JPEG_QUALITY, max(70, current_quality - 15)]
                success, encoded_frame = cv2.imencode('.jpg', placeholder, encode_param)
                
                if success:
                    frame_data = encoded_frame.tobytes()
                else:
                    continue
            
            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            
            # Update quality controller with performance metrics
            frame_processing_time = time.time() - stream_start_time
            quality_controller.update_performance(frame_processing_time)
            
            last_frame_time = current_time
            
        except Exception as e:
            logger.error(f"Error in MJPEG stream generation: {e}")
            await asyncio.sleep(0.1)

@app.get("/")
async def get_dashboard():
    """Main dashboard page with improved video streaming."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CCTV Security System</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            .header {{ background: #c0392b; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            .status-card {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
            .metric {{ text-align: center; padding: 15px; }}
            .metric-value {{ font-size: 2em; font-weight: bold; }}
            .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
            .events-list {{ max-height: 400px; overflow-y: auto; }}
            .event-item {{ padding: 10px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }}
            .alert-item {{ background-color: #ffebee; border-left: 4px solid #f44336; }}
            .authorized-item {{ background-color: #e8f5e8; border-left: 4px solid #4caf50; }}
            .status-indicator {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 10px; }}
            .status-active {{ background-color: #27ae60; }}
            .status-inactive {{ background-color: #e74c3c; }}
            .status-alert {{ background-color: #f39c12; }}
            .video-container {{ text-align: center; position: relative; }}
            .video-stream {{ max-width: 100%; height: auto; border: 2px solid #ddd; border-radius: 8px; }}
            .video-controls {{ margin-top: 10px; text-align: center; }}
            .control-button {{ padding: 8px 16px; margin: 0 5px; border: none; border-radius: 4px; cursor: pointer; }}
            .control-primary {{ background: #3498db; color: white; }}
            .control-secondary {{ background: #95a5a6; color: white; }}
            .security-alert {{ background: #f44336; color: white; padding: 10px; border-radius: 4px; margin: 10px 0; }}
            .metric-good {{ color: #27ae60; }}
            .metric-warning {{ color: #f39c12; }}
            .metric-danger {{ color: #e74c3c; }}
            .setup-info {{ background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #2196f3; }}
            .stream-info {{ background: #f8f9fa; padding: 10px; border-radius: 4px; margin: 10px 0; font-size: 0.9em; color: #666; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🛡️ CCTV Security System</h1>
                <p>Real-time Security Monitoring & Access Control - Enhanced Streaming</p>
            </div>
            
            <div class="status-card">
                <h2>Security Status</h2>
                <div class="grid">
                    <div class="metric">
                        <div class="metric-value status-indicator" id="video-status">
                            <span class="status-inactive"></span>
                            Checking...
                        </div>
                        <div class="metric-label">Video System</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value metric-good" id="personnel-count">0</div>
                        <div class="metric-label">Authorized Personnel</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value metric-warning" id="total-events">0</div>
                        <div class="metric-label">Total Detections</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value metric-danger" id="alert-count">0</div>
                        <div class="metric-label">Security Alerts</div>
                    </div>
                </div>
                
                <div id="setup-reminder" class="setup-info" style="display: none;">
                    📋 <strong>Setup Instructions:</strong><br>
                    1. Add authorized personnel photos to <code>data/personnel/</code> directory<br>
                    2. Name files with person's name (e.g., <code>john_doe.jpg</code>)<br>
                    3. Run the security video capture system<br>
                </div>
            </div>
            
            <div class="status-card">
                <h2>Live Video Feed with Security Analysis</h2>
                <div class="video-container">
                    <img id="video-stream" 
                         class="video-stream"
                         src="/video_stream" 
                         alt="Security Video Stream">
                    
                    <div class="video-controls">
                        <button class="control-button control-primary" onclick="refreshStream()">
                            🔄 Refresh Stream
                        </button>
                        <button class="control-button control-secondary" onclick="toggleStreamInfo()">
                            📊 Stream Info
                        </button>
                    </div>
                    
                    <div id="stream-info" class="stream-info" style="display: none;">
                        <strong>Stream Information:</strong><br>
                        • Format: MJPEG over HTTP<br>
                        • Target FPS: {TARGET_FPS}<br>
                        • Buffer Size: {FRAME_BUFFER_SIZE} frames<br>
                        • Quality: Adaptive (85% JPEG)<br>
                    </div>
                </div>
                <p style="text-align: center; color: #666; margin-top: 10px;">
                    📹 Green boxes: Authorized personnel | 🚨 Red boxes: Unauthorized access
                </p>
            </div>
            
            <div class="grid">
                <div class="status-card">
                    <h2>🚨 Security Alerts</h2>
                    <div class="events-list" id="alerts-list">
                        <div class="event-item">
                            <span>Loading security alerts...</span>
                            <span>--:--</span>
                        </div>
                    </div>
                </div>
                
                <div class="status-card">
                    <h2>📋 Recent Activity</h2>
                    <div class="events-list" id="events-list">
                        <div class="event-item">
                            <span>Loading activity log...</span>
                            <span>--:--</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            async function refreshSecurityStatus() {{
                try {{
                    const response = await fetch('/api/security/status');
                    const status = await response.json();
                    
                    // Update video status
                    const videoStatus = document.getElementById('video-status');
                    if (status.video_active) {{
                        videoStatus.innerHTML = '<span class="status-indicator status-active"></span>Active';
                    }} else {{
                        videoStatus.innerHTML = '<span class="status-indicator status-inactive"></span>Offline';
                    }}
                    
                    // Update metrics
                    document.getElementById('personnel-count').textContent = status.authorized_personnel;
                    document.getElementById('total-events').textContent = status.total_events;
                    document.getElementById('alert-count').textContent = status.unauthorized_alerts;
                    
                    // Show setup reminder if no personnel
                    const setupReminder = document.getElementById('setup-reminder');
                    if (status.authorized_personnel === 0) {{
                        setupReminder.style.display = 'block';
                    }} else {{
                        setupReminder.style.display = 'none';
                    }}
                    
                }} catch (error) {{
                    console.error('Error refreshing security status:', error);
                    document.getElementById('video-status').innerHTML = 
                        '<span class="status-indicator status-inactive"></span>Error';
                }}
            }}
            
            async function refreshSecurityEvents() {{
                try {{
                    const response = await fetch('/api/security/events');
                    const events = await response.json();
                    
                    const eventsList = document.getElementById('events-list');
                    if (events.length === 0) {{
                        eventsList.innerHTML = '<div class="event-item"><span>No recent activity</span><span>--:--</span></div>';
                        return;
                    }}
                    
                    eventsList.innerHTML = events.map(event => {{
                        const itemClass = event.authorized ? 'authorized-item' : 'alert-item';
                        const statusClass = event.authorized ? 'status-active' : 'status-alert';
                        const personName = event.person_name || 'Unknown';
                        const alertLevel = event.authorized ? '✅' : '🚨';
                        
                        return `
                            <div class="event-item ${{itemClass}}">
                                <span>
                                    <span class="status-indicator ${{statusClass}}"></span>
                                    ${{alertLevel}} ${{personName}}
                                    ${{event.authorized ? '' : ' (UNAUTHORIZED)'}}
                                </span>
                                <span>${{new Date(event.datetime).toLocaleTimeString()}}</span>
                            </div>
                        `;
                    }}).join('');
                }} catch (error) {{
                    console.error('Error refreshing security events:', error);
                }}
            }}
            
            async function refreshSecurityAlerts() {{
                try {{
                    const response = await fetch('/api/security/alerts');
                    const alerts = await response.json();
                    
                    const alertsList = document.getElementById('alerts-list');
                    if (alerts.length === 0) {{
                        alertsList.innerHTML = '<div class="event-item"><span>✅ No security alerts</span><span>All Clear</span></div>';
                        return;
                    }}
                    
                    alertsList.innerHTML = alerts.map(alert => `
                        <div class="event-item alert-item">
                            <span>
                                <span class="status-indicator status-alert"></span>
                                🚨 UNAUTHORIZED: ${{alert.person_name || 'Unknown Person'}}
                                ${{alert.confidence ? `(${{Math.round(alert.confidence * 100)}}%)` : ''}}
                            </span>
                            <span>${{new Date(alert.datetime).toLocaleTimeString()}}</span>
                        </div>
                    `).join('');
                }} catch (error) {{
                    console.error('Error refreshing security alerts:', error);
                }}
            }}
            
            function refreshStream() {{
                const videoImg = document.getElementById('video-stream');
                const currentSrc = videoImg.src;
                videoImg.src = '';
                setTimeout(() => {{
                    videoImg.src = currentSrc;
                }}, 100);
            }}
            
            function toggleStreamInfo() {{
                const streamInfo = document.getElementById('stream-info');
                streamInfo.style.display = streamInfo.style.display === 'none' ? 'block' : 'none';
            }}
            
            // Auto-refresh security data every 3 seconds
            setInterval(() => {{
                refreshSecurityStatus();
                refreshSecurityEvents();
                refreshSecurityAlerts();
            }}, 3000);
            
            // Initial load
            refreshSecurityStatus();
            refreshSecurityEvents();
            refreshSecurityAlerts();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/video_stream")
async def video_stream():
    """Serve MJPEG video stream."""
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/security/events")
async def get_security_events():
    """Get recent security events."""
    try:
        events_file = "data/security_events.json"
        if os.path.exists(events_file):
            with open(events_file, 'r') as f:
                events = json.load(f)
            return events[-20:]  # Last 20 events
        return []
    except Exception as e:
        logger.error(f"Error reading security events: {e}")
        return []

@app.get("/api/security/alerts")
async def get_security_alerts():
    """Get recent security alerts (unauthorized detections)."""
    try:
        events_file = "data/security_events.json"
        if os.path.exists(events_file):
            with open(events_file, 'r') as f:
                events = json.load(f)
            # Filter for unauthorized events only
            alerts = [event for event in events if not event.get('authorized', True)]
            return alerts[-10:]  # Last 10 alerts
        return []
    except Exception as e:
        logger.error(f"Error reading security alerts: {e}")
        return []

@app.get("/api/security/status")
async def get_security_status():
    """Get security system status."""
    try:
        events_file = "data/security_events.json"
        personnel_dir = Path("data/personnel")
        
        # Count authorized personnel
        personnel_count = 0
        if personnel_dir.exists():
            personnel_files = list(personnel_dir.glob("*.jpg")) + list(personnel_dir.glob("*.png"))
            personnel_count = len(personnel_files)
        
        # Get recent events stats
        total_events = 0
        unauthorized_events = 0
        if os.path.exists(events_file):
            with open(events_file, 'r') as f:
                events = json.load(f)
            total_events = len(events)
            unauthorized_events = sum(1 for event in events if not event.get('authorized', True))
        
        # Check if video capture is running (frame age)
        video_active = False
        if os.path.exists(CURRENT_FRAME_FILE):
            file_time = os.path.getmtime(CURRENT_FRAME_FILE)
            age_seconds = time.time() - file_time
            video_active = age_seconds < 60
        
        return {
            "video_active": video_active,
            "authorized_personnel": personnel_count,
            "total_events": total_events,
            "unauthorized_alerts": unauthorized_events,
            "last_update": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting security status: {e}")
        return {
            "video_active": False,
            "authorized_personnel": 0,
            "total_events": 0,
            "unauthorized_alerts": 0,
            "last_update": datetime.now().isoformat()
        }

@app.get("/current_frame")
async def get_current_frame():
    """Serve current frame as static image (fallback)."""
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    
    # Get frame from buffer
    frame_data, timestamp = frame_buffer.get_latest_frame_sync()
    
    if frame_data is not None:
        # Save frame data to temporary file
        temp_path = "data/temp_frame.jpg"
        with open(temp_path, 'wb') as f:
            f.write(frame_data)
        
        return FileResponse(
            path=temp_path,
            media_type="image/jpeg",
            filename="current_frame.jpg"
        )
    
    # Fallback to file-based approach
    if os.path.exists(CURRENT_FRAME_FILE):
        file_time = os.path.getmtime(CURRENT_FRAME_FILE)
        age_seconds = time.time() - file_time
        if age_seconds < 60:
            return FileResponse(
                path=CURRENT_FRAME_FILE,
                media_type="image/jpeg",
                filename="current_frame.jpg"
            )
    
    # Create placeholder image if no current frame
    placeholder = create_placeholder_frame()
    placeholder_path = "data/placeholder.jpg"
    cv2.imwrite(placeholder_path, placeholder)
    
    return FileResponse(
        path=placeholder_path,
        media_type="image/jpeg",
        filename="placeholder.jpg"
    )

@app.on_event("startup")
async def startup_event():
    """Start the frame watcher when the server starts."""
    frame_watcher.start()

@app.on_event("shutdown")
async def shutdown_event():
    """Stop the frame watcher when the server shuts down."""
    frame_watcher.stop()

async def main():
    """Run the improved web server."""
    print("🌐 Starting Enhanced CCTV Security Web Server...")
    print("🎥 Features: MJPEG streaming, frame buffering, adaptive quality")
    print("🔒 Security monitoring dashboard with consistent video streaming")
    print("🔗 Open http://localhost:8080 in your browser")
    print("⏹️  Press Ctrl+C to stop")
    print("=" * 60)
    
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=8080,
        log_level="info"
    )
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        print("✅ Enhanced web server stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Enhanced web server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
