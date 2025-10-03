#!/usr/bin/env python3
"""
Configuration manager for notification settings.
Handles loading and saving notification preferences.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class NotificationConfig:
    """Manages notification configuration settings."""
    
    def __init__(self, config_file: str = "data/notification_config.json"):
        self.config_file = config_file
        self.config = self._load_default_config()
        self._ensure_data_dir()
        self.load_config()
    
    def _ensure_data_dir(self):
        """Ensure the data directory exists."""
        Path(os.path.dirname(self.config_file)).mkdir(parents=True, exist_ok=True)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration settings."""
        return {
            "enabled": False,
            "alert_methods": {
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",  # Use app passwords for Gmail
                    "from_email": "",
                    "to_email": "",
                    "subject_prefix": "[CCTV Alert]"
                },
                "sms": {
                    "enabled": False,
                    "service": "twilio",  # twilio, textbelt, etc.
                    "api_key": "",
                    "api_secret": "",
                    "from_number": "",
                    "to_number": ""
                },
                "pushbullet": {
                    "enabled": False,
                    "access_token": "",
                    "device_iden": ""  # Optional, for specific device
                },
                "telegram": {
                    "enabled": False,
                    "bot_token": "",
                    "chat_id": ""
                },
                "webhook": {
                    "enabled": False,
                    "url": "",
                    "method": "POST",
                    "headers": {
                        "Content-Type": "application/json"
                    }
                },
                "discord": {
                    "enabled": False,
                    "webhook_url": ""
                }
            },
            "alert_settings": {
                "send_on_unauthorized": True,
                "send_on_system_start": True,
                "send_on_system_error": True,
                "minimum_interval_seconds": 60,  # Prevent spam
                "alert_cooldown": 300,  # 5 minutes between similar alerts
                "include_screenshot": True,
                "alert_severity_levels": {
                    "LOW": False,
                    "MEDIUM": True,
                    "HIGH": True,
                    "CRITICAL": True
                }
            },
            "mobile_settings": {
                "quick_setup_mode": True,
                "preferred_method": "email",  # Primary mobile alert method
                "backup_method": "pushbullet",  # Backup if primary fails
                "test_alerts": True  # Enable test alert functionality
            }
        }
    
    def load_config(self) -> bool:
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                
                # Merge with defaults to ensure all keys exist
                self._merge_config(self.config, file_config)
                logger.info(f"Loaded notification config from {self.config_file}")
                return True
            else:
                logger.info("No config file found, using defaults")
                return False
        
        except Exception as e:
            logger.error(f"Error loading notification config: {e}")
            return False
    
    def save_config(self) -> bool:
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved notification config to {self.config_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving notification config: {e}")
            return False
    
    def _merge_config(self, default: Dict, loaded: Dict):
        """Recursively merge loaded config with defaults."""
        for key, value in loaded.items():
            if key in default:
                if isinstance(value, dict) and isinstance(default[key], dict):
                    self._merge_config(default[key], value)
                else:
                    default[key] = value
    
    def get(self, key: str, default=None) -> Any:
        """Get configuration value by key path (e.g., 'email.enabled')."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by key path."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def is_enabled(self) -> bool:
        """Check if notifications are globally enabled."""
        return self.get('enabled', False)
    
    def get_enabled_methods(self) -> list:
        """Get list of enabled notification methods."""
        methods = []
        for method, settings in self.get('alert_methods', {}).items():
            if settings.get('enabled', False):
                methods.append(method)
        return methods
    
    def setup_quick_mobile(self, method: str, **kwargs) -> bool:
        """Quick setup for mobile notifications."""
        try:
            if method == "email":
                self.set('alert_methods.email.enabled', True)
                self.set('alert_methods.email.username', kwargs.get('username', ''))
                self.set('alert_methods.email.password', kwargs.get('password', ''))
                self.set('alert_methods.email.from_email', kwargs.get('from_email', ''))
                self.set('alert_methods.email.to_email', kwargs.get('to_email', ''))
            
            elif method == "telegram":
                self.set('alert_methods.telegram.enabled', True)
                self.set('alert_methods.telegram.bot_token', kwargs.get('bot_token', ''))
                self.set('alert_methods.telegram.chat_id', kwargs.get('chat_id', ''))
            
            elif method == "pushbullet":
                self.set('alert_methods.pushbullet.enabled', True)
                self.set('alert_methods.pushbullet.access_token', kwargs.get('access_token', ''))
            
            elif method == "discord":
                self.set('alert_methods.discord.enabled', True)
                self.set('alert_methods.discord.webhook_url', kwargs.get('webhook_url', ''))
            
            # Enable notifications globally
            self.set('enabled', True)
            self.set('mobile_settings.preferred_method', method)
            
            return self.save_config()
        
        except Exception as e:
            logger.error(f"Error in quick mobile setup: {e}")
            return False
    
    def create_setup_wizard(self) -> Dict[str, Any]:
        """Create a setup wizard configuration for mobile alerts."""
        return {
            "title": "Mobile Alert Setup Wizard",
            "description": "Choose how you want to receive security alerts on your mobile device",
            "methods": [
                {
                    "name": "email",
                    "title": "📧 Email Notifications",
                    "description": "Receive alerts via email (works with Gmail, Outlook, etc.)",
                    "difficulty": "Easy",
                    "setup_fields": [
                        {"name": "from_email", "type": "email", "label": "Your Email Address"},
                        {"name": "to_email", "type": "email", "label": "Alert Destination Email"},
                        {"name": "username", "type": "email", "label": "Email Username"},
                        {"name": "password", "type": "password", "label": "Email Password/App Password"}
                    ]
                },
                {
                    "name": "telegram",
                    "title": "📱 Telegram Bot",
                    "description": "Fast notifications via Telegram messenger",
                    "difficulty": "Medium",
                    "setup_fields": [
                        {"name": "bot_token", "type": "text", "label": "Bot Token"},
                        {"name": "chat_id", "type": "text", "label": "Chat ID"}
                    ],
                    "instructions": [
                        "1. Message @BotFather on Telegram",
                        "2. Create new bot with /newbot",
                        "3. Copy the bot token",
                        "4. Message your bot, then get chat ID"
                    ]
                },
                {
                    "name": "pushbullet",
                    "title": "📬 Pushbullet",
                    "description": "Cross-platform push notifications",
                    "difficulty": "Easy",
                    "setup_fields": [
                        {"name": "access_token", "type": "text", "label": "Access Token"}
                    ],
                    "instructions": [
                        "1. Sign up at pushbullet.com",
                        "2. Go to Settings > Access Tokens",
                        "3. Create new token and copy it"
                    ]
                },
                {
                    "name": "discord",
                    "title": "🎮 Discord Webhook",
                    "description": "Notifications to Discord server/DM",
                    "difficulty": "Medium",
                    "setup_fields": [
                        {"name": "webhook_url", "type": "url", "label": "Webhook URL"}
                    ],
                    "instructions": [
                        "1. Create Discord server or use existing",
                        "2. Server Settings > Integrations > Webhooks",
                        "3. Create webhook and copy URL"
                    ]
                }
            ]
        }
