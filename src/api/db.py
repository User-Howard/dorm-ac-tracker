from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path("seats.db")


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS seats (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                seat_id   TEXT    NOT NULL,
                user_name TEXT    NOT NULL,
                x1        INTEGER NOT NULL,
                y1        INTEGER NOT NULL,
                x2        INTEGER NOT NULL,
                y2        INTEGER NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS occupancy_log (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                seat_id   TEXT    NOT NULL,
                user_name TEXT    NOT NULL,
                event     TEXT    NOT NULL,
                timestamp TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now', 'localtime'))
            )
        """)


def get_seats() -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, seat_id, user_name, x1, y1, x2, y2 FROM seats ORDER BY id"
        ).fetchall()
    return [
        {"id": r[0], "seat_id": r[1], "user_name": r[2],
         "x1": r[3], "y1": r[4], "x2": r[5], "y2": r[6]}
        for r in rows
    ]


def log_occupancy(seat_id: str, user_name: str, event: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO occupancy_log (seat_id, user_name, event) VALUES (?,?,?)",
            (seat_id, user_name, event),
        )


def replace_seats(seats: list[dict]) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM seats")
        conn.executemany(
            "INSERT INTO seats (seat_id, user_name, x1, y1, x2, y2) VALUES (?,?,?,?,?,?)",
            [(s["seat_id"], s["user_name"], s["x1"], s["y1"], s["x2"], s["y2"])
             for s in seats],
        )
