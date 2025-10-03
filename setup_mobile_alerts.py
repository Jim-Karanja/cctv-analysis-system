#!/usr/bin/env python3
"""
Mobile Alert Setup Wizard for CCTV Security System

Easy setup script to configure mobile notifications for security alerts.
Supports multiple notification methods including email, Telegram, Pushbullet, and Discord.
"""

import sys
import os
from pathlib import Path

# Add the current directory to path to import notification service
sys.path.insert(0, str(Path(__file__).parent))

try:
    from notification_service import NotificationManager, NotificationConfig
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False

def print_header():
    """Print the setup wizard header."""
    print("=" * 60)
    print("📱 CCTV Mobile Alert Setup Wizard")
    print("=" * 60)
    print("Configure mobile notifications for your CCTV security system")
    print()

def print_method_info(method_info):
    """Print information about a notification method."""
    print(f"🔹 {method_info['title']}")
    print(f"   Description: {method_info['description']}")
    print(f"   Difficulty: {method_info['difficulty']}")
    
    if 'instructions' in method_info:
        print("   Setup Instructions:")
        for instruction in method_info['instructions']:
            print(f"     {instruction}")
    print()

def get_user_input(prompt, input_type="text", required=True):
    """Get user input with validation."""
    while True:
        if input_type == "password":
            import getpass
            value = getpass.getpass(f"{prompt}: ")
        else:
            value = input(f"{prompt}: ").strip()
        
        if required and not value:
            print("❌ This field is required. Please enter a value.")
            continue
        
        if input_type == "email" and value and "@" not in value:
            print("❌ Please enter a valid email address.")
            continue
        
        if input_type == "url" and value and not (value.startswith("http://") or value.startswith("https://")):
            print("❌ Please enter a valid URL starting with http:// or https://")
            continue
        
        return value

def setup_email_alerts(notification_manager):
    """Setup email notifications."""
    print("📧 Setting up Email Notifications")
    print("-" * 40)
    print("For Gmail users:")
    print("1. Enable 2-factor authentication in your Google account")
    print("2. Generate an 'App Password' for this application")
    print("3. Use your Gmail address and the app password below")
    print()
    
    config = {}
    config['from_email'] = get_user_input("Your email address (sender)")
    config['to_email'] = get_user_input("Alert destination email (can be same as sender)")
    config['username'] = get_user_input("Email username (usually same as sender email)")
    
    print("\n⚠️  For security, your password will be hidden as you type")
    config['password'] = get_user_input("Email password/app password", "password")
    
    # Test configuration
    print("\n🔧 Testing email configuration...")
    success = notification_manager.configure_method("email", **config)
    
    if success:
        print("✅ Email notifications configured successfully!")
        
        # Send test email
        test_choice = input("\nWould you like to send a test email? (y/n): ").lower().strip()
        if test_choice in ['y', 'yes']:
            result = notification_manager.send_test_alert()
            if result and any(result.values()):
                print("📧 Test email sent! Check your inbox.")
            else:
                print("❌ Test email failed. Please check your configuration.")
        
        return True
    else:
        print("❌ Failed to configure email notifications. Please check your settings.")
        return False

def setup_telegram_alerts(notification_manager):
    """Setup Telegram notifications."""
    print("📱 Setting up Telegram Notifications")
    print("-" * 40)
    print("Follow these steps:")
    print("1. Open Telegram and message @BotFather")
    print("2. Send '/newbot' and follow the instructions")
    print("3. Copy the bot token provided by BotFather")
    print("4. Message your new bot (send any message)")
    print("5. Visit: https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates")
    print("6. Find your chat_id in the response")
    print()
    
    config = {}
    config['bot_token'] = get_user_input("Bot token from BotFather")
    config['chat_id'] = get_user_input("Your chat ID")
    
    print("\n🔧 Testing Telegram configuration...")
    success = notification_manager.configure_method("telegram", **config)
    
    if success:
        print("✅ Telegram notifications configured successfully!")
        
        # Send test message
        test_choice = input("\nWould you like to send a test message? (y/n): ").lower().strip()
        if test_choice in ['y', 'yes']:
            result = notification_manager.send_test_alert()
            if result and any(result.values()):
                print("📱 Test message sent! Check your Telegram.")
            else:
                print("❌ Test message failed. Please check your configuration.")
        
        return True
    else:
        print("❌ Failed to configure Telegram notifications. Please check your settings.")
        return False

def setup_pushbullet_alerts(notification_manager):
    """Setup Pushbullet notifications."""
    print("📬 Setting up Pushbullet Notifications")
    print("-" * 40)
    print("Follow these steps:")
    print("1. Sign up for a free account at pushbullet.com")
    print("2. Install the Pushbullet app on your mobile device")
    print("3. Go to Account Settings > Access Tokens")
    print("4. Create a new access token and copy it")
    print()
    
    config = {}
    config['access_token'] = get_user_input("Pushbullet access token")
    
    print("\n🔧 Testing Pushbullet configuration...")
    success = notification_manager.configure_method("pushbullet", **config)
    
    if success:
        print("✅ Pushbullet notifications configured successfully!")
        
        # Send test push
        test_choice = input("\nWould you like to send a test push notification? (y/n): ").lower().strip()
        if test_choice in ['y', 'yes']:
            result = notification_manager.send_test_alert()
            if result and any(result.values()):
                print("📬 Test push sent! Check your devices.")
            else:
                print("❌ Test push failed. Please check your configuration.")
        
        return True
    else:
        print("❌ Failed to configure Pushbullet notifications. Please check your settings.")
        return False

def setup_discord_alerts(notification_manager):
    """Setup Discord notifications."""
    print("🎮 Setting up Discord Notifications")
    print("-" * 40)
    print("Follow these steps:")
    print("1. Create a Discord server or use an existing one")
    print("2. Go to Server Settings > Integrations > Webhooks")
    print("3. Create a new webhook")
    print("4. Copy the webhook URL")
    print()
    
    config = {}
    config['webhook_url'] = get_user_input("Discord webhook URL", "url")
    
    print("\n🔧 Testing Discord configuration...")
    success = notification_manager.configure_method("discord", **config)
    
    if success:
        print("✅ Discord notifications configured successfully!")
        
        # Send test message
        test_choice = input("\nWould you like to send a test message? (y/n): ").lower().strip()
        if test_choice in ['y', 'yes']:
            result = notification_manager.send_test_alert()
            if result and any(result.values()):
                print("🎮 Test message sent! Check your Discord server.")
            else:
                print("❌ Test message failed. Please check your configuration.")
        
        return True
    else:
        print("❌ Failed to configure Discord notifications. Please check your settings.")
        return False

def configure_alert_settings(notification_manager):
    """Configure alert behavior settings."""
    print("\n⚙️ Alert Settings Configuration")
    print("-" * 40)
    
    settings = {}
    
    # Alert types
    print("Choose which types of events should send mobile alerts:")
    settings['send_on_unauthorized'] = input("Send alerts for unauthorized access? (y/n) [default: y]: ").lower().strip()
    settings['send_on_unauthorized'] = settings['send_on_unauthorized'] != 'n'
    
    settings['send_on_system_start'] = input("Send alert when system starts? (y/n) [default: y]: ").lower().strip()
    settings['send_on_system_start'] = settings['send_on_system_start'] != 'n'
    
    settings['send_on_system_error'] = input("Send alerts for system errors? (y/n) [default: y]: ").lower().strip()
    settings['send_on_system_error'] = settings['send_on_system_error'] != 'n'
    
    # Include screenshots
    settings['include_screenshot'] = input("Include security camera snapshots with alerts? (y/n) [default: y]: ").lower().strip()
    settings['include_screenshot'] = settings['include_screenshot'] != 'n'
    
    # Alert cooldown
    print("\nAlert rate limiting:")
    cooldown_input = input("Minutes between similar alerts (prevents spam) [default: 5]: ").strip()
    try:
        cooldown_minutes = int(cooldown_input) if cooldown_input else 5
        settings['alert_cooldown'] = cooldown_minutes * 60  # Convert to seconds
    except ValueError:
        settings['alert_cooldown'] = 300  # 5 minutes default
    
    # Apply settings
    success = notification_manager.update_alert_settings(settings)
    
    if success:
        print("✅ Alert settings configured successfully!")
    else:
        print("❌ Failed to update alert settings.")
    
    return success

def show_current_status(notification_manager):
    """Show current notification status."""
    print("\n📊 Current Mobile Alert Status")
    print("-" * 40)
    
    status = notification_manager.get_status()
    
    print(f"Global notifications: {'✅ Enabled' if status['enabled'] else '❌ Disabled'}")
    print(f"Configured methods: {status['method_count']}")
    
    if status['enabled_methods']:
        print("Active notification methods:")
        for method in status['enabled_methods']:
            print(f"  • {method.title()}")
    else:
        print("No notification methods configured")
    
    # Validation
    validation = notification_manager.validate_configuration()
    if not validation['valid']:
        print("\n⚠️ Configuration Issues:")
        for error in validation['errors']:
            print(f"  • {error}")
    
    print()

def main():
    """Main setup wizard."""
    if not NOTIFICATIONS_AVAILABLE:
        print("❌ Notification service not available!")
        print("Please ensure all required packages are installed.")
        return
    
    print_header()
    
    # Initialize notification manager
    notification_manager = NotificationManager()
    
    # Show current status
    show_current_status(notification_manager)
    
    # Get setup wizard info
    wizard_info = notification_manager.get_setup_wizard_info()
    
    print(wizard_info['description'])
    print("\nAvailable notification methods:")
    print()
    
    # Show available methods
    for method in wizard_info['methods']:
        print_method_info(method)
    
    # Method selection loop
    configured_methods = []
    
    while True:
        print("Choose a notification method to configure:")
        print("1. 📧 Email")
        print("2. 📱 Telegram")
        print("3. 📬 Pushbullet")
        print("4. 🎮 Discord")
        print("5. ⚙️ Configure alert settings")
        print("6. 📊 Show current status")
        print("7. 🧪 Send test alert")
        print("8. ✅ Finish setup")
        print()
        
        choice = input("Enter your choice (1-8): ").strip()
        
        if choice == '1':
            if setup_email_alerts(notification_manager):
                configured_methods.append("Email")
        
        elif choice == '2':
            if setup_telegram_alerts(notification_manager):
                configured_methods.append("Telegram")
        
        elif choice == '3':
            if setup_pushbullet_alerts(notification_manager):
                configured_methods.append("Pushbullet")
        
        elif choice == '4':
            if setup_discord_alerts(notification_manager):
                configured_methods.append("Discord")
        
        elif choice == '5':
            configure_alert_settings(notification_manager)
        
        elif choice == '6':
            show_current_status(notification_manager)
        
        elif choice == '7':
            print("\n🧪 Sending test alert to all configured methods...")
            result = notification_manager.send_test_alert()
            if result and any(result.values()):
                successful = [method for method, success in result.items() if success]
                print(f"✅ Test alert sent successfully via: {', '.join(successful)}")
            else:
                print("❌ Test alert failed. Please check your configuration.")
        
        elif choice == '8':
            break
        
        else:
            print("❌ Invalid choice. Please enter a number between 1-8.")
        
        print()
    
    # Final summary
    print("\n🎉 Mobile Alert Setup Complete!")
    print("=" * 60)
    
    if configured_methods:
        print(f"✅ Configured methods: {', '.join(configured_methods)}")
        print("\n📱 Your CCTV system will now send mobile alerts for:")
        print("  • Unauthorized access attempts")
        print("  • System startup notifications")
        print("  • Critical system errors")
        print("\n🔧 To modify settings later, run this setup script again.")
    else:
        print("⚠️  No notification methods were configured.")
        print("Run this script again to set up mobile alerts.")
    
    print("\n🚀 Start your CCTV system to begin receiving mobile alerts!")
    print("Command: python threaded_security_video.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Setup cancelled by user.")
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        print("Please check your configuration and try again.")
