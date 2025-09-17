#!/usr/bin/env python3
"""
Notification Manager - Central coordinator for all mobile alerts
Integrates with CCTV security events to send real-time mobile notifications
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List
from .config_manager import NotificationConfig
from .mobile_alerts import MobileAlertService

logger = logging.getLogger(__name__)

class NotificationManager:
    """Main notification manager that coordinates all alert services."""
    
    def __init__(self, config_file: str = "data/notification_config.json"):
        self.config = NotificationConfig(config_file)
        self.mobile_alerts = MobileAlertService(self.config)
        self.running = False
        self.alert_thread = None
        self._pending_alerts = []
        self._alert_lock = threading.Lock()
        
    def start(self):
        """Start the notification manager."""
        if not self.running:
            self.running = True
            logger.info("Notification manager started")
            
            # Send startup alert if enabled
            if self.config.get('alert_settings.send_on_system_start', True):
                self.send_alert(
                    alert_type="system_start",
                    message="CCTV Security System has started and is monitoring for threats.",
                    severity="MEDIUM"
                )
    
    def stop(self):
        """Stop the notification manager."""
        if self.running:
            self.running = False
            if self.alert_thread and self.alert_thread.is_alive():
                self.alert_thread.join(timeout=5)
            logger.info("Notification manager stopped")
    
    def send_alert(self, alert_type: str, message: str, severity: str = "HIGH", 
                   image_path: Optional[str] = None, extra_data: Optional[Dict] = None) -> Dict[str, bool]:
        """Send a mobile alert."""
        try:
            return self.mobile_alerts.send_mobile_alert(
                alert_type=alert_type,
                message=message,
                severity=severity,
                image_path=image_path,
                extra_data=extra_data
            )
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return {"error": True, "message": str(e)}
    
    def send_security_alert(self, event_data: Dict[str, Any]) -> Dict[str, bool]:
        """Send alert for security events (unauthorized access, etc.)."""
        try:
            person_name = event_data.get('person_name', 'Unknown Person')
            authorized = event_data.get('authorized', False)
            confidence = event_data.get('confidence', 0.0)
            alert_level = event_data.get('alert_level', 'HIGH')
            
            if authorized:
                # Authorized person detected - low priority alert
                message = f"Authorized person detected: {person_name} (confidence: {confidence:.2f})"
                severity = "LOW"
                alert_type = "authorized_access"
            else:
                # Unauthorized person detected - high priority alert
                message = f"🚨 UNAUTHORIZED ACCESS DETECTED! Unknown person: {person_name} (confidence: {confidence:.2f})"
                severity = "HIGH" if alert_level == "HIGH" else "MEDIUM"
                alert_type = "unauthorized_access"
            
            # Include timestamp and bbox info in extra data
            extra_data = {
                "person_name": person_name,
                "authorized": authorized,
                "confidence": confidence,
                "bbox": event_data.get('bbox', []),
                "alert_level": alert_level,
                "event_timestamp": event_data.get('timestamp', time.time())
            }
            
            # Use current frame as image if available
            image_path = "data/current_frame.jpg" if self.config.get('alert_settings.include_screenshot', True) else None
            
            return self.send_alert(
                alert_type=alert_type,
                message=message,
                severity=severity,
                image_path=image_path,
                extra_data=extra_data
            )
            
        except Exception as e:
            logger.error(f"Error sending security alert: {e}")
            return {"error": True, "message": str(e)}
    
    def send_system_error_alert(self, error_message: str, error_type: str = "system_error") -> Dict[str, bool]:
        """Send alert for system errors."""
        message = f"CCTV System Error: {error_message}"
        return self.send_alert(
            alert_type=error_type,
            message=message,
            severity="CRITICAL"
        )
    
    def send_test_alert(self) -> Dict[str, bool]:
        """Send a test alert to verify mobile notifications are working."""
        return self.mobile_alerts.send_test_alert()
    
    def is_enabled(self) -> bool:
        """Check if notifications are enabled."""
        return self.config.is_enabled()
    
    def get_enabled_methods(self) -> List[str]:
        """Get list of enabled notification methods."""
        return self.config.get_enabled_methods()
    
    def get_config(self) -> NotificationConfig:
        """Get the configuration manager."""
        return self.config
    
    def get_mobile_service(self) -> MobileAlertService:
        """Get the mobile alert service."""
        return self.mobile_alerts
    
    def get_status(self) -> Dict[str, Any]:
        """Get notification system status."""
        enabled_methods = self.get_enabled_methods()
        alert_stats = self.mobile_alerts.get_alert_stats()
        
        return {
            "enabled": self.is_enabled(),
            "running": self.running,
            "enabled_methods": enabled_methods,
            "method_count": len(enabled_methods),
            "alert_stats": alert_stats,
            "config_file": self.config.config_file,
            "last_config_update": time.time()
        }
    
    def configure_method(self, method: str, **kwargs) -> bool:
        """Configure a specific notification method."""
        try:
            success = self.config.setup_quick_mobile(method, **kwargs)
            if success:
                # Refresh mobile alerts service with new config
                self.mobile_alerts = MobileAlertService(self.config)
                logger.info(f"Successfully configured {method} notifications")
            return success
        except Exception as e:
            logger.error(f"Error configuring {method}: {e}")
            return False
    
    def disable_method(self, method: str) -> bool:
        """Disable a specific notification method."""
        try:
            self.config.set(f'alert_methods.{method}.enabled', False)
            success = self.config.save_config()
            if success:
                self.mobile_alerts = MobileAlertService(self.config)
                logger.info(f"Disabled {method} notifications")
            return success
        except Exception as e:
            logger.error(f"Error disabling {method}: {e}")
            return False
    
    def enable_method(self, method: str) -> bool:
        """Enable a specific notification method (must be configured first)."""
        try:
            self.config.set(f'alert_methods.{method}.enabled', True)
            success = self.config.save_config()
            if success:
                self.mobile_alerts = MobileAlertService(self.config)
                logger.info(f"Enabled {method} notifications")
            return success
        except Exception as e:
            logger.error(f"Error enabling {method}: {e}")
            return False
    
    def update_alert_settings(self, settings: Dict[str, Any]) -> bool:
        """Update alert behavior settings."""
        try:
            for key, value in settings.items():
                self.config.set(f'alert_settings.{key}', value)
            
            success = self.config.save_config()
            if success:
                logger.info("Updated alert settings")
            return success
        except Exception as e:
            logger.error(f"Error updating alert settings: {e}")
            return False
    
    def get_setup_wizard_info(self) -> Dict[str, Any]:
        """Get setup wizard information for mobile configuration."""
        return self.config.create_setup_wizard()
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current notification configuration."""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "method_status": {}
        }
        
        if not self.is_enabled():
            validation_results["warnings"].append("Notifications are globally disabled")
        
        enabled_methods = self.get_enabled_methods()
        
        if not enabled_methods:
            validation_results["errors"].append("No notification methods are enabled")
            validation_results["valid"] = False
        
        # Validate each enabled method
        for method in enabled_methods:
            method_config = self.config.get(f'alert_methods.{method}', {})
            method_valid = True
            method_issues = []
            
            if method == "email":
                required_fields = ['username', 'password', 'from_email', 'to_email']
                for field in required_fields:
                    if not method_config.get(field, '').strip():
                        method_issues.append(f"Missing {field}")
                        method_valid = False
            
            elif method == "telegram":
                required_fields = ['bot_token', 'chat_id']
                for field in required_fields:
                    if not method_config.get(field, '').strip():
                        method_issues.append(f"Missing {field}")
                        method_valid = False
            
            elif method == "pushbullet":
                if not method_config.get('access_token', '').strip():
                    method_issues.append("Missing access_token")
                    method_valid = False
            
            elif method == "discord":
                if not method_config.get('webhook_url', '').strip():
                    method_issues.append("Missing webhook_url")
                    method_valid = False
            
            validation_results["method_status"][method] = {
                "valid": method_valid,
                "issues": method_issues
            }
            
            if not method_valid:
                validation_results["valid"] = False
                validation_results["errors"].extend([f"{method}: {issue}" for issue in method_issues])
        
        return validation_results
