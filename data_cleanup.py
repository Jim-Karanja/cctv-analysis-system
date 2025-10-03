#!/usr/bin/env python3
"""
Data Cleanup Manager for CCTV Security System

Automatically cleans up old personnel data events to optimize memory usage and storage.
Configurable retention policies for different types of data.
"""

import json
import os
import time
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import threading

logger = logging.getLogger(__name__)

class DataCleanupManager:
    """Manages automatic cleanup of old data files and events."""
    
    def __init__(self, config_file: str = "data/cleanup_config.json"):
        self.config_file = config_file
        self.config = self._load_default_config()
        self.cleanup_stats = {}
        self.last_cleanup = 0
        self._load_config()
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default cleanup configuration."""
        return {
            "enabled": True,
            "auto_cleanup_interval_hours": 24,  # Run cleanup every 24 hours
            "retention_policies": {
                "security_events": {
                    "max_events": 1000,  # Keep maximum 1000 events
                    "max_age_days": 30,  # Keep events for 30 days
                    "keep_unauthorized_longer": True,  # Keep unauthorized events longer
                    "unauthorized_max_age_days": 90,  # Keep unauthorized for 90 days
                    "compress_old_events": True  # Compress events older than 7 days
                },
                "notification_logs": {
                    "max_events": 500,
                    "max_age_days": 14
                },
                "system_logs": {
                    "max_age_days": 7,
                    "max_file_size_mb": 10
                },
                "frames": {
                    "keep_latest_only": True,  # Only keep current_frame.jpg
                    "cleanup_temp_files": True,  # Remove temp files
                    "max_age_minutes": 60  # Remove old frames after 1 hour
                }
            },
            "backup_before_cleanup": True,  # Create backup before major cleanup
            "backup_retention_days": 7,
            "memory_optimization": {
                "compress_events": True,
                "batch_process_size": 100,  # Process in batches to save memory
                "use_rotation": True  # Rotate files instead of growing indefinitely
            }
        }
    
    def _load_config(self):
        """Load cleanup configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                self._merge_config(self.config, file_config)
                logger.info(f"Loaded cleanup config from {self.config_file}")
        except Exception as e:
            logger.error(f"Error loading cleanup config: {e}")
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved cleanup config to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving cleanup config: {e}")
            return False
    
    def _merge_config(self, default: Dict, loaded: Dict):
        """Recursively merge loaded config with defaults."""
        for key, value in loaded.items():
            if key in default:
                if isinstance(value, dict) and isinstance(default[key], dict):
                    self._merge_config(default[key], value)
                else:
                    default[key] = value
    
    def get_data_size_info(self) -> Dict[str, Any]:
        """Get information about data file sizes."""
        data_dir = Path("data")
        info = {
            "total_size_mb": 0,
            "files": {},
            "directories": {}
        }
        
        try:
            for item in data_dir.rglob("*"):
                if item.is_file():
                    size_mb = item.stat().st_size / (1024 * 1024)
                    info["files"][str(item)] = {
                        "size_mb": round(size_mb, 3),
                        "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    }
                    info["total_size_mb"] += size_mb
            
            info["total_size_mb"] = round(info["total_size_mb"], 3)
            
        except Exception as e:
            logger.error(f"Error getting data size info: {e}")
        
        return info
    
    def cleanup_security_events(self) -> Dict[str, Any]:
        """Clean up old security events based on retention policies."""
        events_file = "data/security_events.json"
        cleanup_result = {
            "processed": False,
            "original_count": 0,
            "final_count": 0,
            "removed_count": 0,
            "compressed": False,
            "file_size_before_mb": 0,
            "file_size_after_mb": 0
        }
        
        try:
            if not os.path.exists(events_file):
                return cleanup_result
            
            # Get file size before
            cleanup_result["file_size_before_mb"] = round(os.path.getsize(events_file) / (1024 * 1024), 3)
            
            with open(events_file, 'r') as f:
                events = json.load(f)
            
            cleanup_result["original_count"] = len(events)
            
            if not events:
                return cleanup_result
            
            policy = self.config["retention_policies"]["security_events"]
            current_time = time.time()
            
            # Filter events based on retention policy
            filtered_events = []
            
            for event in events:
                event_time = event.get('timestamp', 0)
                event_age_days = (current_time - event_time) / (24 * 3600)
                is_unauthorized = not event.get('authorized', True)
                
                # Keep event if:
                # 1. Within max age for authorized events
                # 2. Unauthorized and within extended retention period
                # 3. Within max event count (keep most recent)
                
                max_age = policy["unauthorized_max_age_days"] if (is_unauthorized and policy["keep_unauthorized_longer"]) else policy["max_age_days"]
                
                if event_age_days <= max_age:
                    filtered_events.append(event)
            
            # Sort by timestamp (newest first) and keep only max_events
            filtered_events.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            if len(filtered_events) > policy["max_events"]:
                filtered_events = filtered_events[:policy["max_events"]]
            
            cleanup_result["final_count"] = len(filtered_events)
            cleanup_result["removed_count"] = cleanup_result["original_count"] - cleanup_result["final_count"]
            
            # Create backup if significant cleanup
            if cleanup_result["removed_count"] > 10 and self.config["backup_before_cleanup"]:
                backup_file = f"data/backup/security_events_backup_{int(current_time)}.json"
                os.makedirs("data/backup", exist_ok=True)
                shutil.copy2(events_file, backup_file)
                logger.info(f"Created backup: {backup_file}")
            
            # Save cleaned events
            with open(events_file, 'w') as f:
                json.dump(filtered_events, f, separators=(',', ':'))  # Compact format to save space
            
            # Get file size after
            cleanup_result["file_size_after_mb"] = round(os.path.getsize(events_file) / (1024 * 1024), 3)
            cleanup_result["processed"] = True
            
            if cleanup_result["removed_count"] > 0:
                logger.info(f"Cleaned up {cleanup_result['removed_count']} old security events")
            
        except Exception as e:
            logger.error(f"Error cleaning up security events: {e}")
            cleanup_result["error"] = str(e)
        
        return cleanup_result
    
    def cleanup_old_frames(self) -> Dict[str, Any]:
        """Clean up old frame files."""
        cleanup_result = {
            "processed": False,
            "removed_files": [],
            "total_size_freed_mb": 0
        }
        
        try:
            data_dir = Path("data")
            policy = self.config["retention_policies"]["frames"]
            current_time = time.time()
            max_age_seconds = policy["max_age_minutes"] * 60
            
            # Files to potentially remove
            frame_patterns = ["*frame*.jpg", "*frame*.png", "temp_*.jpg", "placeholder*.jpg"]
            files_to_check = []
            
            for pattern in frame_patterns:
                files_to_check.extend(data_dir.glob(pattern))
            
            for file_path in files_to_check:
                if not file_path.is_file():
                    continue
                
                # Keep current_frame.jpg if policy says so
                if policy["keep_latest_only"] and file_path.name == "current_frame.jpg":
                    continue
                
                # Check age
                file_age_seconds = current_time - file_path.stat().st_mtime
                
                if file_age_seconds > max_age_seconds:
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    try:
                        file_path.unlink()
                        cleanup_result["removed_files"].append(str(file_path))
                        cleanup_result["total_size_freed_mb"] += file_size_mb
                    except Exception as e:
                        logger.warning(f"Failed to remove {file_path}: {e}")
            
            cleanup_result["total_size_freed_mb"] = round(cleanup_result["total_size_freed_mb"], 3)
            cleanup_result["processed"] = True
            
            if cleanup_result["removed_files"]:
                logger.info(f"Cleaned up {len(cleanup_result['removed_files'])} old frame files, freed {cleanup_result['total_size_freed_mb']} MB")
        
        except Exception as e:
            logger.error(f"Error cleaning up frames: {e}")
            cleanup_result["error"] = str(e)
        
        return cleanup_result
    
    def cleanup_old_backups(self) -> Dict[str, Any]:
        """Clean up old backup files."""
        cleanup_result = {
            "processed": False,
            "removed_files": [],
            "total_size_freed_mb": 0
        }
        
        try:
            backup_dir = Path("data/backup")
            if not backup_dir.exists():
                cleanup_result["processed"] = True
                return cleanup_result
            
            current_time = time.time()
            max_age_seconds = self.config["backup_retention_days"] * 24 * 3600
            
            for backup_file in backup_dir.glob("*"):
                if not backup_file.is_file():
                    continue
                
                file_age_seconds = current_time - backup_file.stat().st_mtime
                
                if file_age_seconds > max_age_seconds:
                    file_size_mb = backup_file.stat().st_size / (1024 * 1024)
                    try:
                        backup_file.unlink()
                        cleanup_result["removed_files"].append(str(backup_file))
                        cleanup_result["total_size_freed_mb"] += file_size_mb
                    except Exception as e:
                        logger.warning(f"Failed to remove backup {backup_file}: {e}")
            
            cleanup_result["total_size_freed_mb"] = round(cleanup_result["total_size_freed_mb"], 3)
            cleanup_result["processed"] = True
            
            if cleanup_result["removed_files"]:
                logger.info(f"Cleaned up {len(cleanup_result['removed_files'])} old backup files")
        
        except Exception as e:
            logger.error(f"Error cleaning up backups: {e}")
            cleanup_result["error"] = str(e)
        
        return cleanup_result
    
    def run_full_cleanup(self) -> Dict[str, Any]:
        """Run complete cleanup process."""
        if not self.config["enabled"]:
            return {"skipped": True, "reason": "Cleanup disabled in configuration"}
        
        start_time = time.time()
        logger.info("Starting full data cleanup process...")
        
        results = {
            "start_time": datetime.now().isoformat(),
            "security_events": self.cleanup_security_events(),
            "frames": self.cleanup_old_frames(),
            "backups": self.cleanup_old_backups(),
            "total_time_seconds": 0,
            "total_freed_mb": 0
        }
        
        # Calculate totals
        results["total_time_seconds"] = round(time.time() - start_time, 2)
        
        for category in ["security_events", "frames", "backups"]:
            category_result = results[category]
            if "file_size_before_mb" in category_result and "file_size_after_mb" in category_result:
                results["total_freed_mb"] += category_result["file_size_before_mb"] - category_result["file_size_after_mb"]
            elif "total_size_freed_mb" in category_result:
                results["total_freed_mb"] += category_result["total_size_freed_mb"]
        
        results["total_freed_mb"] = round(results["total_freed_mb"], 3)
        
        # Update last cleanup time
        self.last_cleanup = time.time()
        
        logger.info(f"Cleanup completed in {results['total_time_seconds']}s, freed {results['total_freed_mb']} MB")
        
        # Save cleanup stats
        self.cleanup_stats = results
        
        return results
    
    def should_run_cleanup(self) -> bool:
        """Check if it's time to run automatic cleanup."""
        if not self.config["enabled"]:
            return False
        
        if self.last_cleanup == 0:
            return True  # Never run before
        
        interval_seconds = self.config["auto_cleanup_interval_hours"] * 3600
        return (time.time() - self.last_cleanup) >= interval_seconds
    
    def get_cleanup_stats(self) -> Dict[str, Any]:
        """Get statistics about last cleanup run."""
        return self.cleanup_stats
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage by reorganizing data structures."""
        results = {
            "processed": False,
            "optimizations": []
        }
        
        try:
            # Compact security events file
            events_file = "data/security_events.json"
            if os.path.exists(events_file):
                with open(events_file, 'r') as f:
                    events = json.load(f)
                
                # Remove unnecessary fields, normalize data
                optimized_events = []
                for event in events:
                    # Keep only essential fields
                    essential_event = {
                        'person_name': event.get('person_name', 'Unknown'),
                        'authorized': bool(event.get('authorized', False)),
                        'confidence': round(float(event.get('confidence', 0)), 2),
                        'timestamp': float(event.get('timestamp', 0)),
                        'alert_level': event.get('alert_level', 'HIGH')
                    }
                    
                    # Only include bbox if it's meaningful
                    bbox = event.get('bbox', [])
                    if bbox and len(bbox) == 4 and any(x > 0 for x in bbox):
                        essential_event['bbox'] = bbox
                    
                    optimized_events.append(essential_event)
                
                # Save optimized version
                with open(events_file, 'w') as f:
                    json.dump(optimized_events, f, separators=(',', ':'))
                
                results["optimizations"].append("Optimized security events structure")
            
            results["processed"] = True
            
        except Exception as e:
            logger.error(f"Error optimizing memory usage: {e}")
            results["error"] = str(e)
        
        return results
    
    def configure_retention_policy(self, category: str, **settings) -> bool:
        """Update retention policy for a specific data category."""
        try:
            if category not in self.config["retention_policies"]:
                logger.error(f"Unknown category: {category}")
                return False
            
            for key, value in settings.items():
                if key in self.config["retention_policies"][category]:
                    self.config["retention_policies"][category][key] = value
            
            return self.save_config()
        
        except Exception as e:
            logger.error(f"Error configuring retention policy: {e}")
            return False

class AutoCleanupService:
    """Background service for automatic data cleanup."""
    
    def __init__(self, cleanup_manager: DataCleanupManager):
        self.cleanup_manager = cleanup_manager
        self.running = False
        self.cleanup_thread = None
    
    def start(self):
        """Start the automatic cleanup service."""
        if not self.running:
            self.running = True
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()
            logger.info("Auto cleanup service started")
    
    def stop(self):
        """Stop the automatic cleanup service."""
        self.running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
        logger.info("Auto cleanup service stopped")
    
    def _cleanup_loop(self):
        """Main cleanup loop running in background thread."""
        while self.running:
            try:
                if self.cleanup_manager.should_run_cleanup():
                    logger.info("Running scheduled data cleanup...")
                    self.cleanup_manager.run_full_cleanup()
                
                # Sleep for 1 hour before checking again
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                time.sleep(3600)  # Wait before retrying

def main():
    """Test/demo the cleanup functionality."""
    logging.basicConfig(level=logging.INFO)
    
    cleanup_manager = DataCleanupManager()
    
    print("🧹 CCTV Data Cleanup Manager")
    print("=" * 50)
    
    # Show current data size
    data_info = cleanup_manager.get_data_size_info()
    print(f"\n📊 Current Data Usage: {data_info['total_size_mb']} MB")
    
    for file_path, info in data_info['files'].items():
        if info['size_mb'] > 0.1:  # Only show files larger than 0.1 MB
            print(f"  • {Path(file_path).name}: {info['size_mb']} MB")
    
    # Run cleanup
    print("\n🧹 Running cleanup...")
    results = cleanup_manager.run_full_cleanup()
    
    print(f"\n✅ Cleanup completed in {results['total_time_seconds']}s")
    print(f"💾 Total space freed: {results['total_freed_mb']} MB")
    
    if results['security_events']['processed']:
        se_result = results['security_events']
        print(f"📋 Security events: {se_result['removed_count']} events removed")
    
    if results['frames']['processed']:
        frame_result = results['frames']
        print(f"🖼️  Frames: {len(frame_result['removed_files'])} files removed")

if __name__ == "__main__":
    main()
