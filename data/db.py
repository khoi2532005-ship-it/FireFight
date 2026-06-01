import sqlite3, json, os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detections.db")

def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                filename   TEXT,
                lat        REAL,
                lon        REAL,
                results    TEXT,
                confidence REAL,
                risk_level INTEGER DEFAULT 0,
                mode       TEXT,
                ts         TEXT
            )
        """)

def save_detection(filename, detections, metadata, confidence=0.0, mode="Detection", risk_level=0):
    init_db()
    with _conn() as con:
        con.execute(
            "INSERT INTO detections (filename, lat, lon, results, confidence, mode, ts, risk_level) VALUES (?,?,?,?,?,?,?,?)",
            (filename, metadata.get("Latitude"), metadata.get("Longitude"),
             json.dumps(detections), confidence, mode, datetime.utcnow().isoformat(), risk_level)
        )

def update_risk_level(detection_id, risk_level):
    init_db()
    with _conn() as con:
        con.execute("UPDATE detections SET risk_level = ? WHERE id = ?", (risk_level, detection_id))

def load_all_detections():
    init_db()
    with _conn() as con:
        rows = con.execute("SELECT * FROM detections ORDER BY ts DESC").fetchall()
    return [dict(r) for r in rows]

def clear_db():
    init_db()
    with _conn() as con:
        con.execute("DELETE FROM detections")