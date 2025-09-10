"""
Personnel Manager

Manages personnel database operations including CRUD operations for authorized personnel.
"""

import logging
import sqlite3
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import time


@dataclass
class PersonnelRecord:
    """Represents a personnel record in the database."""
    person_id: str
    name: str
    department: str
    access_level: str
    is_active: bool = True
    created_at: float = None
    updated_at: float = None
    metadata: Dict[str, Any] = None


class PersonnelManager:
    """
    Manages the personnel database with CRUD operations for authorized personnel.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Database configuration
        self.db_path = config.get('db_path', 'data/personnel.db')
        self.connection = None
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the personnel database."""
        try:
            # Create database directory if it doesn't exist
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Enable column access by name
            
            # Create tables
            self._create_tables()
            
            self.logger.info(f"Personnel database initialized at {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize personnel database: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.connection.cursor()
        
        # Personnel table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS personnel (
                person_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT,
                access_level TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_at REAL,
                updated_at REAL,
                metadata TEXT
            )
        ''')
        
        self.connection.commit()
    
    def add_person(self, person: PersonnelRecord) -> bool:
        """
        Add a new person to the personnel database.
        
        Args:
            person: Personnel record to add
            
        Returns:
            True if added successfully
        """
        try:
            cursor = self.connection.cursor()
            
            current_time = time.time()
            
            cursor.execute('''
                INSERT INTO personnel 
                (person_id, name, department, access_level, is_active, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                person.person_id,
                person.name,
                person.department,
                person.access_level,
                person.is_active,
                current_time,
                current_time,
                json.dumps(person.metadata) if person.metadata else None
            ))
            
            self.connection.commit()
            
            self.logger.info(f"Added person {person.name} (ID: {person.person_id}) to database")
            return True
            
        except sqlite3.IntegrityError:
            self.logger.error(f"Person with ID {person.person_id} already exists")
            return False
        except Exception as e:
            self.logger.error(f"Error adding person {person.person_id}: {e}")
            return False
    
    def get_person(self, person_id: str) -> Optional[PersonnelRecord]:
        """
        Get a person by ID.
        
        Args:
            person_id: ID of person to retrieve
            
        Returns:
            PersonnelRecord or None if not found
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('SELECT * FROM personnel WHERE person_id = ?', (person_id,))
            row = cursor.fetchone()
            
            if row:
                return PersonnelRecord(
                    person_id=row['person_id'],
                    name=row['name'],
                    department=row['department'],
                    access_level=row['access_level'],
                    is_active=bool(row['is_active']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else None
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving person {person_id}: {e}")
            return None
    
    def update_person(self, person_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update person information.
        
        Args:
            person_id: ID of person to update
            updates: Dictionary of fields to update
            
        Returns:
            True if updated successfully
        """
        try:
            cursor = self.connection.cursor()
            
            # Build update query dynamically
            set_clauses = []
            values = []
            
            for field, value in updates.items():
                if field in ['name', 'department', 'access_level', 'is_active']:
                    set_clauses.append(f"{field} = ?")
                    values.append(value)
                elif field == 'metadata':
                    set_clauses.append("metadata = ?")
                    values.append(json.dumps(value) if value else None)
            
            if not set_clauses:
                return False
            
            # Add updated_at timestamp
            set_clauses.append("updated_at = ?")
            values.append(time.time())
            
            # Add person_id for WHERE clause
            values.append(person_id)
            
            query = f"UPDATE personnel SET {', '.join(set_clauses)} WHERE person_id = ?"
            
            cursor.execute(query, values)
            self.connection.commit()
            
            if cursor.rowcount > 0:
                self.logger.info(f"Updated person {person_id}")
                return True
            else:
                self.logger.warning(f"Person {person_id} not found for update")
                return False
                
        except Exception as e:
            self.logger.error(f"Error updating person {person_id}: {e}")
            return False
    
    def remove_person(self, person_id: str) -> bool:
        """
        Remove a person from the database.
        
        Args:
            person_id: ID of person to remove
            
        Returns:
            True if removed successfully
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('DELETE FROM personnel WHERE person_id = ?', (person_id,))
            self.connection.commit()
            
            if cursor.rowcount > 0:
                self.logger.info(f"Removed person {person_id} from database")
                return True
            else:
                self.logger.warning(f"Person {person_id} not found for removal")
                return False
                
        except Exception as e:
            self.logger.error(f"Error removing person {person_id}: {e}")
            return False
    
    def get_all_personnel(self, active_only: bool = True) -> List[PersonnelRecord]:
        """
        Get all personnel records.
        
        Args:
            active_only: If True, only return active personnel
            
        Returns:
            List of PersonnelRecord objects
        """
        try:
            cursor = self.connection.cursor()
            
            if active_only:
                cursor.execute('SELECT * FROM personnel WHERE is_active = 1')
            else:
                cursor.execute('SELECT * FROM personnel')
            
            rows = cursor.fetchall()
            
            personnel = []
            for row in rows:
                personnel.append(PersonnelRecord(
                    person_id=row['person_id'],
                    name=row['name'],
                    department=row['department'],
                    access_level=row['access_level'],
                    is_active=bool(row['is_active']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else None
                ))
            
            return personnel
            
        except Exception as e:
            self.logger.error(f"Error retrieving personnel: {e}")
            return []
    
    def search_personnel(self, query: str, field: str = 'name') -> List[PersonnelRecord]:
        """
        Search personnel by field.
        
        Args:
            query: Search query
            field: Field to search in ('name', 'department', 'access_level')
            
        Returns:
            List of matching PersonnelRecord objects
        """
        try:
            cursor = self.connection.cursor()
            
            if field not in ['name', 'department', 'access_level']:
                field = 'name'
            
            sql_query = f"SELECT * FROM personnel WHERE {field} LIKE ? AND is_active = 1"
            cursor.execute(sql_query, (f"%{query}%",))
            
            rows = cursor.fetchall()
            
            personnel = []
            for row in rows:
                personnel.append(PersonnelRecord(
                    person_id=row['person_id'],
                    name=row['name'],
                    department=row['department'],
                    access_level=row['access_level'],
                    is_active=bool(row['is_active']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else None
                ))
            
            return personnel
            
        except Exception as e:
            self.logger.error(f"Error searching personnel: {e}")
            return []
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.logger.info("Personnel database connection closed")
