"""
Event Logger

Logs all detection and recognition events to the database for audit and analysis.
"""

import logging
import sqlite3
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import time


@dataclass
class DetectionEvent:
    """Represents a detection/recognition event."""
    event_id: str
    timestamp: float
    source_id: str
    frame_number: int
    person_id: Optional[str]
    person_name: Optional[str]
    confidence: float
    bbox: tuple
    event_type: str  # 'detection', 'recognition', 'entry', 'exit'
    metadata: Dict[str, Any] = None


class EventLogger:
    """
    Logs detection and recognition events to database for audit and analysis.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.db_path = config.get('db_path', 'data/events.db')
        self.connection = None
        
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the events database."""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            
            self._create_tables()
            
            self.logger.info(f"Event database initialized at {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize event database: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.connection.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                source_id TEXT NOT NULL,
                frame_number INTEGER,
                person_id TEXT,
                person_name TEXT,
                confidence REAL,
                bbox TEXT,
                event_type TEXT,
                metadata TEXT
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_source_id ON events(source_id)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_person_id ON events(person_id)
        ''')
        
        self.connection.commit()
    
    def log_event(self, event: DetectionEvent) -> bool:
        """
        Log a detection/recognition event.
        
        Args:
            event: DetectionEvent to log
            
        Returns:
            True if logged successfully
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT INTO events 
                (event_id, timestamp, source_id, frame_number, person_id, 
                 person_name, confidence, bbox, event_type, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id,
                event.timestamp,
                event.source_id,
                event.frame_number,
                event.person_id,
                event.person_name,
                event.confidence,
                json.dumps(event.bbox),
                event.event_type,
                json.dumps(event.metadata) if event.metadata else None
            ))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error logging event {event.event_id}: {e}")
            return False
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.logger.info("Event database connection closed")
