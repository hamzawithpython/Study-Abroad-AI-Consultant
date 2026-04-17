# tools/database_mcp.py
# MCP Server 2: Database MCP (SQLite)
# Handles all database operations
# - Stores student profiles
# - Saves search history
# - Tracks recommendations given
# - Stores conversation sessions

import sqlite3
import json
import os
from datetime import datetime

# ─────────────────────────────────────────
# DATABASE MCP SERVER CLASS
# ─────────────────────────────────────────

class DatabaseMCPServer:
    """
    MCP Server for database operations using SQLite.
    
    This server provides tools for:
    1. Storing student profiles persistently
    2. Logging search history
    3. Saving recommendations
    4. Tracking conversation sessions
    
    SQLite is a lightweight database that:
    - Requires NO installation
    - Creates a single .db file
    - Perfect for local projects
    - Works exactly like a real database
    """
    
    def __init__(self, db_path: str = "data/university_finder.db"):
        """
        Args:
            db_path: path to SQLite database file
        """
        
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database tables
        self._create_tables()
        
        print("✅ Database MCP Server initialized")
        print(f"   🗄️  Database: {os.path.abspath(db_path)}")
    
    # ─────────────────────────────────────────
    # DATABASE SETUP
    # ─────────────────────────────────────────
    
    def _get_connection(self):
        """Creates and returns a database connection."""
        return sqlite3.connect(self.db_path)
    
    def _create_tables(self):
        """
        Creates all required database tables.
        Like setting up a new database from scratch.
        """
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Table 1: Student Profiles
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS student_profiles (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT,
                profile_data    TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table 2: Search History
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT,
                search_query    TEXT,
                results_count   INTEGER,
                countries       TEXT,
                field           TEXT,
                searched_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table 3: Recommendations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT,
                universities    TEXT,
                report_text     TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table 4: Conversation Sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT UNIQUE,
                started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message_count   INTEGER DEFAULT 0,
                status          TEXT DEFAULT 'active'
            )
        """)
        
        conn.commit()
        conn.close()
    
    # ─────────────────────────────────────────
    # TOOL 1: SAVE STUDENT PROFILE
    # ─────────────────────────────────────────
    
    def save_student_profile(self, profile: dict,
                              session_id: str = "default") -> dict:
        """
        Saves student profile to database.
        
        Args:
            profile: student profile dictionary
            session_id: unique session identifier
        
        Returns:
            dict with success status
        """
        
        try:
            conn   = self._get_connection()
            cursor = conn.cursor()
            
            profile_json = json.dumps(profile)
            
            # Check if profile exists for this session
            cursor.execute(
                "SELECT id FROM student_profiles WHERE session_id = ?",
                (session_id,)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update existing profile
                cursor.execute("""
                    UPDATE student_profiles 
                    SET profile_data = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                """, (profile_json, session_id))
                action = "updated"
            else:
                # Insert new profile
                cursor.execute("""
                    INSERT INTO student_profiles 
                    (session_id, profile_data)
                    VALUES (?, ?)
                """, (session_id, profile_json))
                action = "created"
            
            conn.commit()
            conn.close()
            
            return {
                "success":    True,
                "action":     action,
                "session_id": session_id,
                "message":    f"Profile {action} in database ✅",
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Database error: {e}",
            }
    
    # ─────────────────────────────────────────
    # TOOL 2: LOG SEARCH
    # ─────────────────────────────────────────
    
    def log_search(self, session_id: str, query: str,
                   results_count: int, countries: str = "",
                   field: str = "") -> dict:
        """
        Logs every university search to database.
        Useful for tracking what was searched.
        
        Args:
            session_id: current session
            query: the search query used
            results_count: how many results found
            countries: countries searched for
            field: field of study searched
        """
        
        try:
            conn   = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO search_history 
                (session_id, search_query, results_count, 
                 countries, field)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, query, results_count, 
                  countries, field))
            
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "message": "Search logged to database ✅",
            }
            
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    # ─────────────────────────────────────────
    # TOOL 3: SAVE RECOMMENDATION
    # ─────────────────────────────────────────
    
    def save_recommendation(self, session_id: str,
                             universities: list,
                             report_text: str) -> dict:
        """
        Saves recommendations given to a student.
        
        Args:
            session_id: current session
            universities: list of recommended universities
            report_text: full recommendation text
        """
        
        try:
            conn   = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO recommendations 
                (session_id, universities, report_text)
                VALUES (?, ?, ?)
            """, (
                session_id,
                json.dumps(universities),
                report_text
            ))
            
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "message": "Recommendation saved to database ✅",
            }
            
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    # ─────────────────────────────────────────
    # TOOL 4: GET SEARCH HISTORY
    # ─────────────────────────────────────────
    
    def get_search_history(self, session_id: str = None,
                            limit: int = 10) -> dict:
        """
        Retrieves search history from database.
        
        Args:
            session_id: filter by session (None = all)
            limit: max number of records
        
        Returns:
            dict with search history
        """
        
        try:
            conn   = self._get_connection()
            cursor = conn.cursor()
            
            if session_id:
                cursor.execute("""
                    SELECT search_query, results_count, 
                           countries, field, searched_at
                    FROM search_history
                    WHERE session_id = ?
                    ORDER BY searched_at DESC
                    LIMIT ?
                """, (session_id, limit))
            else:
                cursor.execute("""
                    SELECT search_query, results_count,
                           countries, field, searched_at
                    FROM search_history
                    ORDER BY searched_at DESC
                    LIMIT ?
                """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            history = []
            for row in rows:
                history.append({
                    "query":          row[0],
                    "results_count":  row[1],
                    "countries":      row[2],
                    "field":          row[3],
                    "searched_at":    row[4],
                })
            
            return {
                "success": True,
                "count":   len(history),
                "history": history,
            }
            
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    # ─────────────────────────────────────────
    # TOOL 5: GET STATS
    # ─────────────────────────────────────────
    
    def get_stats(self) -> dict:
        """
        Returns database statistics.
        Useful for showing project demo metrics.
        """
        
        try:
            conn   = self._get_connection()
            cursor = conn.cursor()
            
            # Count records in each table
            cursor.execute(
                "SELECT COUNT(*) FROM student_profiles"
            )
            profiles_count = cursor.fetchone()[0]
            
            cursor.execute(
                "SELECT COUNT(*) FROM search_history"
            )
            searches_count = cursor.fetchone()[0]
            
            cursor.execute(
                "SELECT COUNT(*) FROM recommendations"
            )
            recommendations_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "success":          True,
                "profiles_saved":   profiles_count,
                "searches_logged":  searches_count,
                "recommendations":  recommendations_count,
            }
            
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def get_db_path(self) -> str:
        """Returns absolute path to database file."""
        return os.path.abspath(self.db_path)