"""
Database Module

Handles personnel management and event logging for the CCTV analysis system.
"""

from .personnel_manager import PersonnelManager
from .event_logger import EventLogger

__all__ = ['PersonnelManager', 'EventLogger']
