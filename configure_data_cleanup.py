#!/usr/bin/env python3
"""
Data Cleanup Configuration Script

Easy configuration for automatic data cleanup and memory optimization.
"""

import sys
import json
from pathlib import Path

# Add the current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from data_cleanup import DataCleanupManager
    CLEANUP_AVAILABLE = True
except ImportError:
    CLEANUP_AVAILABLE = False

def print_header():
    """Print configuration header."""
    print("=" * 60)
    print("🧹 CCTV Data Cleanup Configuration")
    print("=" * 60)
    print("Configure automatic data cleanup for memory optimization")
    print()

def show_current_config(cleanup_manager):
    """Display current configuration."""
    print("📋 Current Configuration:")
    print("-" * 40)
    
    config = cleanup_manager.config
    
    print(f"Cleanup enabled: {'✅ Yes' if config['enabled'] else '❌ No'}")
    print(f"Cleanup interval: Every {config['auto_cleanup_interval_hours']} hours")
    print(f"Backup before cleanup: {'✅ Yes' if config['backup_before_cleanup'] else '❌ No'}")
    
    # Security events policy
    se_policy = config['retention_policies']['security_events']
    print(f"\n📋 Security Events:")
    print(f"  • Max events: {se_policy['max_events']}")
    print(f"  • Max age (authorized): {se_policy['max_age_days']} days")
    print(f"  • Max age (unauthorized): {se_policy['unauthorized_max_age_days']} days")
    
    # Frames policy
    frame_policy = config['retention_policies']['frames']
    print(f"\n🖼️  Frame Files:")
    print(f"  • Keep current frame only: {'✅ Yes' if frame_policy['keep_latest_only'] else '❌ No'}")
    print(f"  • Max age: {frame_policy['max_age_minutes']} minutes")
    print()

def configure_basic_settings(cleanup_manager):
    """Configure basic cleanup settings."""
    print("⚙️ Basic Settings Configuration")
    print("-" * 40)
    
    # Enable/disable cleanup
    enabled_input = input("Enable automatic cleanup? (y/n) [current: enabled]: ").lower().strip()
    if enabled_input:
        cleanup_manager.config['enabled'] = enabled_input in ['y', 'yes']
    
    # Cleanup interval
    interval_input = input("Cleanup interval (hours) [current: 24]: ").strip()
    if interval_input:
        try:
            cleanup_manager.config['auto_cleanup_interval_hours'] = int(interval_input)
        except ValueError:
            print("⚠️ Invalid interval, keeping current value")
    
    # Backup setting
    backup_input = input("Create backup before major cleanup? (y/n) [current: yes]: ").lower().strip()
    if backup_input:
        cleanup_manager.config['backup_before_cleanup'] = backup_input in ['y', 'yes']

def configure_retention_policies(cleanup_manager):
    """Configure data retention policies."""
    print("\n📋 Data Retention Policies")
    print("-" * 40)
    
    # Security events
    print("\n🔒 Security Events:")
    
    max_events_input = input(f"Maximum events to keep [current: {cleanup_manager.config['retention_policies']['security_events']['max_events']}]: ").strip()
    if max_events_input:
        try:
            cleanup_manager.config['retention_policies']['security_events']['max_events'] = int(max_events_input)
        except ValueError:
            print("⚠️ Invalid number, keeping current value")
    
    auth_age_input = input(f"Keep authorized events for how many days? [current: {cleanup_manager.config['retention_policies']['security_events']['max_age_days']}]: ").strip()
    if auth_age_input:
        try:
            cleanup_manager.config['retention_policies']['security_events']['max_age_days'] = int(auth_age_input)
        except ValueError:
            print("⚠️ Invalid number, keeping current value")
    
    unauth_age_input = input(f"Keep UNAUTHORIZED events for how many days? [current: {cleanup_manager.config['retention_policies']['security_events']['unauthorized_max_age_days']}]: ").strip()
    if unauth_age_input:
        try:
            cleanup_manager.config['retention_policies']['security_events']['unauthorized_max_age_days'] = int(unauth_age_input)
        except ValueError:
            print("⚠️ Invalid number, keeping current value")
    
    # Frame files
    print("\n🖼️ Frame Files:")
    
    keep_current_only = input("Keep only current frame file? (y/n) [current: yes]: ").lower().strip()
    if keep_current_only:
        cleanup_manager.config['retention_policies']['frames']['keep_latest_only'] = keep_current_only in ['y', 'yes']
    
    frame_age_input = input(f"Remove old frames after how many minutes? [current: {cleanup_manager.config['retention_policies']['frames']['max_age_minutes']}]: ").strip()
    if frame_age_input:
        try:
            cleanup_manager.config['retention_policies']['frames']['max_age_minutes'] = int(frame_age_input)
        except ValueError:
            print("⚠️ Invalid number, keeping current value")

def test_cleanup(cleanup_manager):
    """Test the cleanup process."""
    print("\n🧪 Testing Cleanup Process")
    print("-" * 40)
    
    # Show current data usage
    data_info = cleanup_manager.get_data_size_info()
    print(f"Current data usage: {data_info['total_size_mb']} MB")
    
    print("\nFiles larger than 0.1 MB:")
    for file_path, info in data_info['files'].items():
        if info['size_mb'] > 0.1:
            filename = Path(file_path).name
            print(f"  • {filename}: {info['size_mb']} MB")
    
    # Ask if user wants to run cleanup
    run_cleanup = input("\nRun cleanup now? (y/n): ").lower().strip()
    
    if run_cleanup in ['y', 'yes']:
        print("\n🧹 Running cleanup...")
        results = cleanup_manager.run_full_cleanup()
        
        if results.get('skipped'):
            print(f"⚠️ Cleanup skipped: {results.get('reason')}")
        else:
            print(f"✅ Cleanup completed in {results['total_time_seconds']}s")
            print(f"💾 Total space freed: {results['total_freed_mb']} MB")
            
            if results['security_events']['processed']:
                se_result = results['security_events']
                print(f"📋 Security events: {se_result['removed_count']} events removed")
                print(f"   File size: {se_result['file_size_before_mb']} → {se_result['file_size_after_mb']} MB")
            
            if results['frames']['processed'] and results['frames']['removed_files']:
                frame_result = results['frames']
                print(f"🖼️ Frames: {len(frame_result['removed_files'])} files removed")

def optimize_memory(cleanup_manager):
    """Run memory optimization."""
    print("\n🚀 Memory Optimization")
    print("-" * 40)
    
    print("This will optimize data structures to use less memory.")
    optimize = input("Run memory optimization? (y/n): ").lower().strip()
    
    if optimize in ['y', 'yes']:
        print("\n⚡ Optimizing...")
        results = cleanup_manager.optimize_memory_usage()
        
        if results['processed']:
            print("✅ Memory optimization completed")
            for optimization in results['optimizations']:
                print(f"  • {optimization}")
        else:
            print("❌ Memory optimization failed")
            if 'error' in results:
                print(f"Error: {results['error']}")

def main():
    """Main configuration interface."""
    if not CLEANUP_AVAILABLE:
        print("❌ Data cleanup system not available!")
        print("Please ensure the data_cleanup.py file is present.")
        return
    
    print_header()
    
    # Initialize cleanup manager
    cleanup_manager = DataCleanupManager()
    
    while True:
        # Show current status
        show_current_config(cleanup_manager)
        
        print("Configuration Options:")
        print("1. ⚙️  Configure basic settings")
        print("2. 📋 Configure retention policies")
        print("3. 🧪 Test cleanup process")
        print("4. 🚀 Optimize memory usage")
        print("5. 📊 Show data usage statistics")
        print("6. 💾 Save configuration")
        print("7. ✅ Exit")
        print()
        
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == '1':
            configure_basic_settings(cleanup_manager)
        
        elif choice == '2':
            configure_retention_policies(cleanup_manager)
        
        elif choice == '3':
            test_cleanup(cleanup_manager)
        
        elif choice == '4':
            optimize_memory(cleanup_manager)
        
        elif choice == '5':
            data_info = cleanup_manager.get_data_size_info()
            print(f"\n📊 Data Usage Statistics:")
            print(f"Total usage: {data_info['total_size_mb']} MB")
            print("\nFile breakdown:")
            for file_path, info in sorted(data_info['files'].items(), key=lambda x: x[1]['size_mb'], reverse=True):
                if info['size_mb'] > 0.01:  # Show files > 0.01 MB
                    filename = Path(file_path).name
                    print(f"  {filename}: {info['size_mb']} MB")
        
        elif choice == '6':
            if cleanup_manager.save_config():
                print("✅ Configuration saved successfully!")
            else:
                print("❌ Failed to save configuration")
        
        elif choice == '7':
            # Auto-save before exit
            if cleanup_manager.save_config():
                print("✅ Configuration saved")
            break
        
        else:
            print("❌ Invalid choice. Please enter a number between 1-7.")
        
        print()
    
    print("\n🧹 Data Cleanup Configuration Complete!")
    print("Your CCTV system will now automatically manage data storage efficiently.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Configuration cancelled by user.")
    except Exception as e:
        print(f"\n❌ Configuration error: {e}")
