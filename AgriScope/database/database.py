"""
AgriScope - Database Module
SQLite database for storing prediction logs.
"""

import sqlite3
import os
from datetime import datetime


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agriscope.db")


def connect_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Create (or connect to) the SQLite database and return a connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _create_table(conn)
    return conn


def _create_table(conn: sqlite3.Connection) -> None:
    """Create prediction_logs table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            district        TEXT NOT NULL,
            season          TEXT NOT NULL,
            temperature     REAL,
            humidity        REAL,
            rainfall        REAL,
            predicted_crop  TEXT,
            predicted_yield REAL,
            timestamp       TEXT NOT NULL
        )
    """)
    conn.commit()


def save_prediction(
    district: str,
    season: str,
    temperature: float,
    humidity: float,
    rainfall: float,
    predicted_crop: str,
    predicted_yield: float,
    db_path: str = DB_PATH,
) -> int:
    """
    Insert a prediction record into prediction_logs.
    Returns the new row id.
    """
    conn = connect_db(db_path)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        INSERT INTO prediction_logs
            (district, season, temperature, humidity, rainfall,
             predicted_crop, predicted_yield, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (district, season, temperature, humidity, rainfall,
         predicted_crop, predicted_yield, timestamp),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def fetch_predictions(limit: int = 100, db_path: str = DB_PATH) -> list:
    """
    Fetch the most recent predictions from the database.
    Returns a list of dicts.
    """
    conn = connect_db(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM prediction_logs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def clear_predictions(db_path: str = DB_PATH) -> None:
    """Delete all records from prediction_logs (used for testing)."""
    conn = connect_db(db_path)
    conn.execute("DELETE FROM prediction_logs")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    # Quick test
    pid = save_prediction(
        district="Ahmedabad",
        season="Kharif",
        temperature=29.5,
        humidity=70.0,
        rainfall=5.2,
        predicted_crop="TOTAL GROUNDNUT",
        predicted_yield=2500.0,
    )
    print(f"Saved prediction with id={pid}")
    records = fetch_predictions()
    print(f"Total records: {len(records)}")
    print(records[0])
