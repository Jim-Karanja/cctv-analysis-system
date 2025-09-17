"""
CCTV Analysis System - Notification Service
Mobile alert and notification management for security events.
"""

from .notification_manager import NotificationManager
from .mobile_alerts import MobileAlertService
from .config_manager import NotificationConfig

__all__ = ['NotificationManager', 'MobileAlertService', 'NotificationConfig']
__version__ = '1.0.0'
