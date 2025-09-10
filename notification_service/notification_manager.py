"""
Notification Manager

Manages notification logic, triggers, and delivery coordination.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time

from .push_notifier import PushNotifier


class NotificationType(Enum):
    """Types of notifications that can be sent."""
    AUTHORIZED_ENTRY = "authorized_entry"
    UNAUTHORIZED_ENTRY = "unauthorized_entry"
    PERSON_RECOGNIZED = "person_recognized"
    UNKNOWN_PERSON = "unknown_person"
    SYSTEM_ALERT = "system_alert"


@dataclass
class NotificationRule:
    """Defines when and how to send notifications."""
    rule_id: str
    name: str
    notification_type: NotificationType
    conditions: Dict[str, Any]
    enabled: bool = True
    cooldown_seconds: int = 300  # 5 minutes default
    recipients: List[str] = None


@dataclass  
class NotificationEvent:
    """Represents a notification to be sent."""
    event_id: str
    notification_type: NotificationType
    title: str
    message: str
    timestamp: float
    source_id: str
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    metadata: Dict[str, Any] = None


class NotificationManager:
    """
    Manages notification rules, triggers, and delivery coordination.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize push notifier
        self.push_notifier = PushNotifier(config.get('push_notifier', {}))
        
        # Notification rules
        self.rules: Dict[str, NotificationRule] = {}
        
        # Cooldown tracking
        self.last_notification_times: Dict[str, float] = {}
        
        # Load default rules
        self._load_default_rules()
        
        self.logger.info("Notification manager initialized")
    
    def _load_default_rules(self):
        """Load default notification rules."""
        default_rules = [
            NotificationRule(
                rule_id="unauthorized_entry",
                name="Unauthorized Person Detected",
                notification_type=NotificationType.UNAUTHORIZED_ENTRY,
                conditions={'min_confidence': 0.7},
                cooldown_seconds=120
            ),
            NotificationRule(
                rule_id="authorized_entry",
                name="Authorized Person Entry",
                notification_type=NotificationType.AUTHORIZED_ENTRY,
                conditions={'min_confidence': 0.8},
                cooldown_seconds=600,
                enabled=False  # Disabled by default
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.rule_id] = rule
    
    def add_rule(self, rule: NotificationRule) -> bool:
        """
        Add a notification rule.
        
        Args:
            rule: NotificationRule to add
            
        Returns:
            True if rule was added successfully
        """
        try:
            self.rules[rule.rule_id] = rule
            self.logger.info(f"Added notification rule: {rule.name}")
            return True
        except Exception as e:
            self.logger.error(f"Error adding rule {rule.rule_id}: {e}")
            return False
    
    def should_send_notification(self, event: NotificationEvent, rule: NotificationRule) -> bool:
        """
        Determine if a notification should be sent based on rule conditions.
        
        Args:
            event: Notification event
            rule: Rule to evaluate
            
        Returns:
            True if notification should be sent
        """
        if not rule.enabled:
            return False
        
        # Check cooldown
        cooldown_key = f"{rule.rule_id}:{event.source_id}:{event.person_id or 'unknown'}"
        
        if cooldown_key in self.last_notification_times:
            time_since_last = time.time() - self.last_notification_times[cooldown_key]
            if time_since_last < rule.cooldown_seconds:
                return False
        
        # Check rule conditions
        conditions = rule.conditions
        
        if 'min_confidence' in conditions:
            confidence = event.metadata.get('confidence', 0.0) if event.metadata else 0.0
            if confidence < conditions['min_confidence']:
                return False
        
        if 'allowed_sources' in conditions:
            if event.source_id not in conditions['allowed_sources']:
                return False
        
        if 'restricted_areas' in conditions:
            area = event.metadata.get('area') if event.metadata else None
            if area and area in conditions['restricted_areas']:
                return True  # Always notify for restricted areas
        
        return True
    
    async def process_identification_result(self, identification_result) -> List[str]:
        """
        Process identification results and send appropriate notifications.
        
        Args:
            identification_result: Results from person identification
            
        Returns:
            List of notification IDs that were sent
        """
        sent_notifications = []
        
        try:
            for person in identification_result.identified_persons:
                if person.is_recognized:
                    # Authorized person detected
                    event = NotificationEvent(
                        event_id=f"auth_{identification_result.source_id}_{identification_result.frame_number}",
                        notification_type=NotificationType.AUTHORIZED_ENTRY,
                        title="Authorized Entry",
                        message=f"{person.person_name} entered the monitored area",
                        timestamp=identification_result.timestamp,
                        source_id=identification_result.source_id,
                        person_id=person.person_id,
                        person_name=person.person_name,
                        metadata={
                            'confidence': person.recognition_confidence,
                            'frame_number': identification_result.frame_number
                        }
                    )
                else:
                    # Unknown person detected
                    event = NotificationEvent(
                        event_id=f"unauth_{identification_result.source_id}_{identification_result.frame_number}",
                        notification_type=NotificationType.UNAUTHORIZED_ENTRY,
                        title="Unauthorized Entry Alert",
                        message="Unknown person detected in monitored area",
                        timestamp=identification_result.timestamp,
                        source_id=identification_result.source_id,
                        metadata={
                            'confidence': person.detection_confidence,
                            'frame_number': identification_result.frame_number
                        }
                    )
                
                # Check all rules for this event type
                for rule in self.rules.values():
                    if rule.notification_type == event.notification_type:
                        if self.should_send_notification(event, rule):
                            notification_id = await self._send_notification(event, rule)
                            if notification_id:
                                sent_notifications.append(notification_id)
                                
                                # Update cooldown tracking
                                cooldown_key = f"{rule.rule_id}:{event.source_id}:{event.person_id or 'unknown'}"
                                self.last_notification_times[cooldown_key] = time.time()
        
        except Exception as e:
            self.logger.error(f"Error processing identification results: {e}")
        
        return sent_notifications
    
    async def _send_notification(self, event: NotificationEvent, rule: NotificationRule) -> Optional[str]:
        """
        Send a notification using the configured notifier.
        
        Args:
            event: Notification event to send
            rule: Rule that triggered the notification
            
        Returns:
            Notification ID if sent successfully, None otherwise
        """
        try:
            # Send push notification
            success = await self.push_notifier.send_notification(
                title=event.title,
                message=event.message,
                data={
                    'event_id': event.event_id,
                    'source_id': event.source_id,
                    'person_id': event.person_id,
                    'timestamp': event.timestamp,
                    'type': event.notification_type.value
                },
                recipients=rule.recipients
            )
            
            if success:
                self.logger.info(f"Sent notification: {event.title} (Rule: {rule.name})")
                return event.event_id
            else:
                self.logger.error(f"Failed to send notification: {event.title}")
                return None
        
        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")
            return None
    
    def get_rules(self) -> List[NotificationRule]:
        """Get all notification rules."""
        return list(self.rules.values())
    
    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a notification rule.
        
        Args:
            rule_id: ID of rule to update
            updates: Dictionary of fields to update
            
        Returns:
            True if updated successfully
        """
        if rule_id not in self.rules:
            return False
        
        try:
            rule = self.rules[rule_id]
            
            if 'enabled' in updates:
                rule.enabled = updates['enabled']
            if 'cooldown_seconds' in updates:
                rule.cooldown_seconds = updates['cooldown_seconds']
            if 'conditions' in updates:
                rule.conditions.update(updates['conditions'])
            if 'recipients' in updates:
                rule.recipients = updates['recipients']
            
            self.logger.info(f"Updated rule {rule_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating rule {rule_id}: {e}")
            return False
