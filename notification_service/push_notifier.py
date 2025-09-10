"""
Push Notifier

Handles mobile push notifications using Firebase Cloud Messaging (FCM).
"""

import logging
import asyncio
from typing import List, Dict, Optional, Any
try:
    from pyfcm import FCMNotification
    FCM_AVAILABLE = True
except ImportError:
    FCM_AVAILABLE = False
    print("Warning: pyfcm not available. Push notifications disabled.")


class PushNotifier:
    """
    Handles mobile push notifications using Firebase Cloud Messaging.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # FCM configuration
        self.api_key = config.get('fcm_api_key')
        self.fcm = None
        
        if self.api_key and FCM_AVAILABLE:
            try:
                self.fcm = FCMNotification(api_key=self.api_key)
                self.logger.info("FCM push notifier initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize FCM: {e}")
        elif not FCM_AVAILABLE:
            self.logger.warning("FCM library not available - push notifications disabled")
        else:
            self.logger.warning("No FCM API key provided - push notifications disabled")
    
    async def send_notification(self, title: str, message: str, data: Dict[str, Any] = None, recipients: List[str] = None) -> bool:
        """
        Send a push notification.
        
        Args:
            title: Notification title
            message: Notification message
            data: Additional data to include
            recipients: List of device tokens (if None, uses default)
            
        Returns:
            True if sent successfully
        """
        if not self.fcm:
            self.logger.warning("FCM not initialized - cannot send notification")
            return False
        
        try:
            # Use default recipients if none provided
            if not recipients:
                recipients = self.config.get('default_recipients', [])
            
            if not recipients:
                self.logger.warning("No recipients configured for notifications")
                return False
            
            # Send to multiple devices
            result = self.fcm.notify_multiple_devices(
                registration_ids=recipients,
                message_title=title,
                message_body=message,
                data_message=data
            )
            
            if result.get('success', 0) > 0:
                self.logger.info(f"Push notification sent: {title}")
                return True
            else:
                self.logger.error(f"Failed to send push notification: {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending push notification: {e}")
            return False
