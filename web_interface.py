"""
Simple Web Interface for CCTV Analysis System

Provides a basic web interface to view system status and recent events.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from database import EventLogger, PersonnelManager


app = FastAPI(title="CCTV Analysis System", description="Real-time CCTV monitoring and face recognition")

# Global variables for system state
system_status = {
    "status": "stopped",
    "active_cameras": [],
    "last_update": None,
    "total_detections": 0,
    "recognized_persons": 0
}

# Global event storage for real-time display
recent_events = []

class WebInterface:
    """Simple web interface for the CCTV system."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize database connections
        db_config = config.get('database', {})
        self.event_logger = EventLogger(db_config.get('events', {}))
        self.personnel_manager = PersonnelManager(db_config.get('personnel', {}))
    
    async def get_recent_events(self, limit: int = 50) -> List[Dict]:
        """Get recent detection events."""
        try:
            # Return recent events stored in memory
            return recent_events[-limit:] if recent_events else []
        except Exception as e:
            self.logger.error(f"Error getting recent events: {e}")
            return []
    
    async def get_system_status(self) -> Dict:
        """Get current system status."""
        return system_status


# Create web interface instance (will be initialized when config is available)
web_interface = None


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Main dashboard page."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CCTV Analysis System</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            .status-card { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .metric { text-align: center; padding: 15px; }
            .metric-value { font-size: 2em; font-weight: bold; color: #3498db; }
            .metric-label { color: #7f8c8d; margin-top: 5px; }
            .events-list { max-height: 400px; overflow-y: auto; }
            .event-item { padding: 10px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; }
            .status-indicator { width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 10px; }
            .status-active { background-color: #27ae60; }
            .status-inactive { background-color: #e74c3c; }
            .status-unknown { background-color: #f39c12; }
            .refresh-btn { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
            .refresh-btn:hover { background: #2980b9; }
            .log-container { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 4px; font-family: monospace; height: 200px; overflow-y: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎥 CCTV Analysis System</h1>
                <p>Real-time face detection and recognition monitoring</p>
            </div>
            
            <div class="status-card">
                <h2>System Status</h2>
                <div class="grid">
                    <div class="metric">
                        <div class="metric-value" id="system-status">
                            <span class="status-indicator status-inactive"></span>
                            Stopped
                        </div>
                        <div class="metric-label">System Status</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="active-cameras">0</div>
                        <div class="metric-label">Active Cameras</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="total-detections">0</div>
                        <div class="metric-label">Total Detections</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="recognized-persons">0</div>
                        <div class="metric-label">Recognized Persons</div>
                    </div>
                </div>
                <button class="refresh-btn" onclick="refreshStatus()">Refresh Status</button>
            </div>
            
            <div class="status-card">
                <h2>Recent Events</h2>
                <div class="events-list" id="events-list">
                    <div class="event-item">
                        <span>No recent events</span>
                        <span>--:--</span>
                    </div>
                </div>
            </div>
            
            <div class="status-card">
                <h2>System Logs</h2>
                <div class="log-container" id="logs">
                    System ready. Waiting for events...
                </div>
            </div>
        </div>
        
        <script>
            async function refreshStatus() {
                try {
                    const response = await fetch('/api/status');
                    const status = await response.json();
                    
                    document.getElementById('system-status').innerHTML = 
                        `<span class="status-indicator ${status.status === 'running' ? 'status-active' : 'status-inactive'}"></span>
                         ${status.status.charAt(0).toUpperCase() + status.status.slice(1)}`;
                    
                    document.getElementById('active-cameras').textContent = status.active_cameras.length;
                    document.getElementById('total-detections').textContent = status.total_detections;
                    document.getElementById('recognized-persons').textContent = status.recognized_persons;
                    
                    // Update last refresh time
                    console.log('Status updated:', new Date().toLocaleTimeString());
                } catch (error) {
                    console.error('Error refreshing status:', error);
                }
            }
            
            async function refreshEvents() {
                try {
                    const response = await fetch('/api/events');
                    const events = await response.json();
                    
                    const eventsList = document.getElementById('events-list');
                    if (events.length === 0) {
                        eventsList.innerHTML = '<div class="event-item"><span>No recent events</span><span>--:--</span></div>';
                        return;
                    }
                    
                    eventsList.innerHTML = events.map(event => `
                        <div class="event-item">
                            <span>
                                <span class="status-indicator ${event.status === 'recognized' ? 'status-active' : 'status-unknown'}"></span>
                                ${event.person_name} (${event.camera_id})
                            </span>
                            <span>${new Date(event.timestamp).toLocaleTimeString()}</span>
                        </div>
                    `).join('');
                } catch (error) {
                    console.error('Error refreshing events:', error);
                }
            }
            
            function addLog(message) {
                const logs = document.getElementById('logs');
                const timestamp = new Date().toLocaleTimeString();
                logs.innerHTML += `[${timestamp}] ${message}\\n`;
                logs.scrollTop = logs.scrollHeight;
            }
            
            // Auto-refresh every 5 seconds
            setInterval(() => {
                refreshStatus();
                refreshEvents();
            }, 5000);
            
            // Initial load
            refreshStatus();
            refreshEvents();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/api/status")
async def get_status():
    """Get system status API endpoint."""
    return system_status


@app.get("/api/events")
async def get_events():
    """Get recent events API endpoint."""
    if web_interface:
        return await web_interface.get_recent_events()
    return []


def update_system_status(status: str, active_cameras: List[str] = None, 
                        total_detections: int = None, recognized_persons: int = None):
    """Update system status from main application."""
    global system_status
    
    system_status["status"] = status
    system_status["last_update"] = datetime.now().isoformat()
    
    if active_cameras is not None:
        system_status["active_cameras"] = active_cameras
    if total_detections is not None:
        system_status["total_detections"] = total_detections
    if recognized_persons is not None:
        system_status["recognized_persons"] = recognized_persons


def add_detection_event(camera_id: str, person_detections: List):
    """Add detection events from main application."""
    global recent_events
    
    for detection in person_detections:
        event = {
            "timestamp": datetime.fromtimestamp(detection.timestamp).isoformat(),
            "camera_id": camera_id,
            "person_name": detection.person_name or "Unknown Person",
            "confidence": detection.recognition_confidence,
            "status": "recognized" if detection.is_recognized else "unrecognized"
        }
        recent_events.append(event)
    
    # Keep only last 100 events to prevent memory issues
    if len(recent_events) > 100:
        recent_events = recent_events[-100:]


async def start_web_server(config: dict):
    """Start the web server."""
    global web_interface
    web_interface = WebInterface(config)
    
    web_config = config.get('web_interface', {})
    host = web_config.get('host', '127.0.0.1')
    port = web_config.get('port', 8080)
    debug = web_config.get('debug', False)
    
    logging.info(f"Starting web server on http://{host}:{port}")
    
    config_uvicorn = uvicorn.Config(
        app, 
        host=host, 
        port=port, 
        log_level="info" if debug else "warning"
    )
    server = uvicorn.Server(config_uvicorn)
    await server.serve()


if __name__ == "__main__":
    # For testing the web interface standalone
    import yaml
    
    try:
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        asyncio.run(start_web_server(config))
    except FileNotFoundError:
        print("Configuration file not found. Please ensure config/config.yaml exists.")
    except Exception as e:
        print(f"Error starting web server: {e}")
