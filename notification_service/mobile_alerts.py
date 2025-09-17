#!/usr/bin/env python3
"""
Mobile Alert Service for CCTV Security System
Supports multiple mobile notification methods including email, SMS, push notifications, etc.
"""

import smtplib
import json
import time
import logging
import requests
import os
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class MobileAlertService:
    """Service for sending mobile alerts via various methods."""
    
    def __init__(self, config):
        self.config = config
        self.last_alert_times = {}  # For rate limiting
        self.alert_history = []
        
    def should_send_alert(self, alert_type: str, severity: str = "HIGH") -> bool:
        """Check if we should send an alert based on settings and rate limits."""
        if not self.config.is_enabled():
            return False
        
        # Check severity level
        if not self.config.get(f'alert_settings.alert_severity_levels.{severity}', True):
            return False
        
        # Check specific alert type settings
        setting_key = f'alert_settings.send_on_{alert_type}'
        if not self.config.get(setting_key, True):
            return False
        
        # Check rate limiting
        current_time = time.time()
        last_alert = self.last_alert_times.get(alert_type, 0)
        cooldown = self.config.get('alert_settings.alert_cooldown', 300)
        
        if current_time - last_alert < cooldown:
            logger.debug(f"Alert {alert_type} blocked by cooldown (last: {last_alert})")
            return False
        
        return True
    
    def send_mobile_alert(self, alert_type: str, message: str, severity: str = "HIGH", 
                         image_path: Optional[str] = None, extra_data: Optional[Dict] = None) -> Dict[str, bool]:
        """Send alert to all enabled mobile notification methods."""
        if not self.should_send_alert(alert_type, severity):
            return {"blocked": True, "reason": "Rate limited or disabled"}
        
        # Update rate limiting
        self.last_alert_times[alert_type] = time.time()
        
        # Get enabled methods
        enabled_methods = self.config.get_enabled_methods()
        results = {}
        
        if not enabled_methods:
            logger.warning("No notification methods enabled")
            return {"error": "No notification methods enabled"}
        
        # Prepare alert data
        alert_data = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "system": "CCTV Security System",
            "extra_data": extra_data or {}
        }
        
        # Try each enabled method
        for method in enabled_methods:
            try:
                if method == "email":
                    results[method] = self._send_email_alert(alert_data, image_path)
                elif method == "telegram":
                    results[method] = self._send_telegram_alert(alert_data, image_path)
                elif method == "pushbullet":
                    results[method] = self._send_pushbullet_alert(alert_data, image_path)
                elif method == "discord":
                    results[method] = self._send_discord_alert(alert_data, image_path)
                elif method == "webhook":
                    results[method] = self._send_webhook_alert(alert_data, image_path)
                elif method == "sms":
                    results[method] = self._send_sms_alert(alert_data)
                else:
                    results[method] = False
                    logger.warning(f"Unknown notification method: {method}")
                
            except Exception as e:
                logger.error(f"Error sending {method} alert: {e}")
                results[method] = False
        
        # Log alert
        self.alert_history.append({
            "timestamp": time.time(),
            "alert_data": alert_data,
            "results": results
        })
        
        # Keep history limited
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-50:]
        
        return results
    
    def _send_email_alert(self, alert_data: Dict, image_path: Optional[str] = None) -> bool:
        """Send email alert."""
        try:
            email_config = self.config.get('alert_methods.email', {})
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_config.get('from_email', '')
            msg['To'] = email_config.get('to_email', '')
            
            subject_prefix = email_config.get('subject_prefix', '[CCTV Alert]')
            msg['Subject'] = f"{subject_prefix} {alert_data['severity']} - {alert_data['type']}"
            
            # Email body
            body = self._create_email_body(alert_data)
            msg.attach(MIMEText(body, 'html'))
            
            # Attach image if available
            if image_path and os.path.exists(image_path) and self.config.get('alert_settings.include_screenshot', True):
                try:
                    with open(image_path, 'rb') as f:
                        img_data = f.read()
                    
                    img = MIMEImage(img_data)
                    img.add_header('Content-Disposition', 'attachment', filename='security_snapshot.jpg')
                    msg.attach(img)
                except Exception as e:
                    logger.warning(f"Failed to attach image: {e}")
            
            # Send email
            smtp_server = email_config.get('smtp_server', 'smtp.gmail.com')
            smtp_port = email_config.get('smtp_port', 587)
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(email_config.get('username', ''), email_config.get('password', ''))
                server.send_message(msg)
            
            logger.info(f"Email alert sent successfully to {email_config.get('to_email')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _send_telegram_alert(self, alert_data: Dict, image_path: Optional[str] = None) -> bool:
        """Send Telegram alert."""
        try:
            telegram_config = self.config.get('alert_methods.telegram', {})
            bot_token = telegram_config.get('bot_token', '')
            chat_id = telegram_config.get('chat_id', '')
            
            if not bot_token or not chat_id:
                logger.error("Telegram bot token or chat ID not configured")
                return False
            
            # Create message
            message = self._create_telegram_message(alert_data)
            
            base_url = f"https://api.telegram.org/bot{bot_token}"
            
            # Send text message
            text_url = f"{base_url}/sendMessage"
            text_data = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(text_url, json=text_data, timeout=10)
            
            if response.status_code == 200:
                # Send image if available
                if image_path and os.path.exists(image_path) and self.config.get('alert_settings.include_screenshot', True):
                    try:
                        photo_url = f"{base_url}/sendPhoto"
                        with open(image_path, 'rb') as photo:
                            files = {'photo': photo}
                            photo_data = {
                                'chat_id': chat_id,
                                'caption': f"🔒 Security snapshot from {alert_data['timestamp']}"
                            }
                            requests.post(photo_url, data=photo_data, files=files, timeout=10)
                    except Exception as e:
                        logger.warning(f"Failed to send Telegram photo: {e}")
                
                logger.info("Telegram alert sent successfully")
                return True
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False
    
    def _send_pushbullet_alert(self, alert_data: Dict, image_path: Optional[str] = None) -> bool:
        """Send Pushbullet alert."""
        try:
            pb_config = self.config.get('alert_methods.pushbullet', {})
            access_token = pb_config.get('access_token', '')
            
            if not access_token:
                logger.error("Pushbullet access token not configured")
                return False
            
            headers = {
                'Access-Token': access_token,
                'Content-Type': 'application/json'
            }
            
            # Create push data
            push_data = {
                'type': 'note',
                'title': f"🚨 CCTV {alert_data['severity']} Alert",
                'body': self._create_simple_message(alert_data)
            }
            
            # Add device filter if specified
            device_iden = pb_config.get('device_iden', '')
            if device_iden:
                push_data['device_iden'] = device_iden
            
            response = requests.post(
                'https://api.pushbullet.com/v2/pushes',
                headers=headers,
                json=push_data,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Pushbullet alert sent successfully")
                return True
            else:
                logger.error(f"Pushbullet API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Pushbullet alert: {e}")
            return False
    
    def _send_discord_alert(self, alert_data: Dict, image_path: Optional[str] = None) -> bool:
        """Send Discord webhook alert."""
        try:
            discord_config = self.config.get('alert_methods.discord', {})
            webhook_url = discord_config.get('webhook_url', '')
            
            if not webhook_url:
                logger.error("Discord webhook URL not configured")
                return False
            
            # Create embed
            color = self._get_alert_color(alert_data['severity'])
            embed = {
                'title': f"🚨 CCTV Security Alert - {alert_data['severity']}",
                'description': alert_data['message'],
                'color': color,
                'timestamp': alert_data['timestamp'],
                'fields': [
                    {'name': 'Alert Type', 'value': alert_data['type'], 'inline': True},
                    {'name': 'Severity', 'value': alert_data['severity'], 'inline': True},
                    {'name': 'System', 'value': alert_data['system'], 'inline': True}
                ],
                'footer': {'text': 'CCTV Analysis System'}
            }
            
            webhook_data = {
                'username': 'CCTV Security',
                'embeds': [embed]
            }
            
            response = requests.post(webhook_url, json=webhook_data, timeout=10)
            
            if response.status_code in [200, 204]:
                logger.info("Discord alert sent successfully")
                return True
            else:
                logger.error(f"Discord webhook error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False
    
    def _send_webhook_alert(self, alert_data: Dict, image_path: Optional[str] = None) -> bool:
        """Send generic webhook alert."""
        try:
            webhook_config = self.config.get('alert_methods.webhook', {})
            url = webhook_config.get('url', '')
            method = webhook_config.get('method', 'POST').upper()
            headers = webhook_config.get('headers', {})
            
            if not url:
                logger.error("Webhook URL not configured")
                return False
            
            # Prepare payload
            payload = {
                'alert': alert_data,
                'timestamp': time.time(),
                'source': 'cctv-analysis-system'
            }
            
            if method == 'POST':
                response = requests.post(url, json=payload, headers=headers, timeout=10)
            elif method == 'PUT':
                response = requests.put(url, json=payload, headers=headers, timeout=10)
            else:
                logger.error(f"Unsupported webhook method: {method}")
                return False
            
            if response.status_code in [200, 201, 202, 204]:
                logger.info("Webhook alert sent successfully")
                return True
            else:
                logger.error(f"Webhook error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False
    
    def _send_sms_alert(self, alert_data: Dict) -> bool:
        """Send SMS alert (placeholder - requires SMS service setup)."""
        logger.warning("SMS alerts not yet implemented - requires SMS service configuration")
        return False
    
    def _create_email_body(self, alert_data: Dict) -> str:
        """Create HTML email body."""
        severity_colors = {
            'LOW': '#28a745',
            'MEDIUM': '#ffc107', 
            'HIGH': '#fd7e14',
            'CRITICAL': '#dc3545'
        }
        
        color = severity_colors.get(alert_data['severity'], '#6c757d')
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa;">
            <div style="max-width: 600px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <div style="background: {color}; color: white; padding: 20px; border-radius: 8px 8px 0 0;">
                    <h1 style="margin: 0; font-size: 24px;">🚨 CCTV Security Alert</h1>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">Severity: {alert_data['severity']}</p>
                </div>
                
                <div style="padding: 20px;">
                    <h2 style="color: #333; margin-top: 0;">Alert Details</h2>
                    
                    <table style="width: 100%; border-collapse: collapse; margin: 15px 0;">
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd; font-weight: bold; width: 120px;">Type:</td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert_data['type']}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd; font-weight: bold;">Time:</td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert_data['timestamp']}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd; font-weight: bold;">System:</td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert_data['system']}</td>
                        </tr>
                    </table>
                    
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0;">
                        <h3 style="margin: 0 0 10px 0; color: #333;">Message:</h3>
                        <p style="margin: 0; color: #666; line-height: 1.5;">{alert_data['message']}</p>
                    </div>
                    
                    <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #666; font-size: 12px;">
                        This alert was automatically generated by your CCTV Analysis System.
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _create_telegram_message(self, alert_data: Dict) -> str:
        """Create Telegram message."""
        severity_emoji = {
            'LOW': '🟢',
            'MEDIUM': '🟡',
            'HIGH': '🟠', 
            'CRITICAL': '🔴'
        }
        
        emoji = severity_emoji.get(alert_data['severity'], '⚪')
        
        return f"""
🚨 <b>CCTV Security Alert</b> {emoji}

<b>Type:</b> {alert_data['type']}
<b>Severity:</b> {alert_data['severity']}
<b>Time:</b> {alert_data['timestamp']}

<b>Details:</b>
{alert_data['message']}

<i>Generated by CCTV Analysis System</i>
        """.strip()
    
    def _create_simple_message(self, alert_data: Dict) -> str:
        """Create simple text message for services like Pushbullet."""
        return f"""CCTV Security Alert ({alert_data['severity']})
Type: {alert_data['type']}
Time: {alert_data['timestamp']}

{alert_data['message']}
        """.strip()
    
    def _get_alert_color(self, severity: str) -> int:
        """Get Discord embed color for severity."""
        colors = {
            'LOW': 0x28a745,      # Green
            'MEDIUM': 0xffc107,   # Yellow
            'HIGH': 0xfd7e14,     # Orange
            'CRITICAL': 0xdc3545  # Red
        }
        return colors.get(severity, 0x6c757d)
    
    def send_test_alert(self) -> Dict[str, bool]:
        """Send a test alert to verify configuration."""
        return self.send_mobile_alert(
            alert_type="test",
            message="This is a test alert from your CCTV Security System. If you received this, your mobile notifications are working correctly!",
            severity="MEDIUM"
        )
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        if not self.alert_history:
            return {"total_alerts": 0, "success_rate": 0, "methods_used": []}
        
        total_alerts = len(self.alert_history)
        successful_alerts = 0
        methods_used = set()
        
        for alert in self.alert_history:
            results = alert.get('results', {})
            if any(results.values()):
                successful_alerts += 1
            methods_used.update(results.keys())
        
        success_rate = (successful_alerts / total_alerts) * 100 if total_alerts > 0 else 0
        
        return {
            "total_alerts": total_alerts,
            "successful_alerts": successful_alerts,
            "success_rate": round(success_rate, 2),
            "methods_used": list(methods_used),
            "last_alert": self.alert_history[-1]['timestamp'] if self.alert_history else None
        }
