import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


class SessionDatabase:
    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path)
        self._create_tables()

    def start_session(self) -> int:
        cursor = self.connection.cursor()
        cursor.execute("INSERT INTO sessions (start_time) VALUES (?)", (self._now(),))
        self.connection.commit()
        return int(cursor.lastrowid)

    def end_session(self, session_id: int, final_sentence: str) -> None:
        self.connection.execute(
            "UPDATE sessions SET end_time = ?, final_sentence = ? WHERE id = ?",
            (self._now(), final_sentence, session_id),
        )
        self.connection.commit()

    def save_event(self, session_id: int, letter: str = "", word: str = "", sentence: str = "") -> None:
        self.connection.execute(
            """
            INSERT INTO recognized_text (session_id, letter, word, sentence, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, letter, word, sentence, self._now()),
        )
        self.connection.commit()

    def recent_sessions(self, limit: int = 10) -> List[Tuple]:
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT id, start_time, end_time, final_sentence
            FROM sessions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return cursor.fetchall()

    def close(self) -> None:
        self.connection.close()

    def _create_tables(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                final_sentence TEXT DEFAULT ''
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS recognized_text (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                letter TEXT DEFAULT '',
                word TEXT DEFAULT '',
                sentence TEXT DEFAULT '',
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
            """
        )
        self.connection.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")
