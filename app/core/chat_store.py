"""
Persistent chat store — saves every question/answer turn to SQLite.

Two tables:
  chat_sessions  — one row per session (id, started_at, question_count)
  chat_turns     — one row per Q/A exchange

On agent startup the store can restore the last session's entity context
so pronoun resolution (e.g. "what about its warranty?") still works across
restarts.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from configs import settings

CHAT_DB_PATH: Path = settings.PROCESSED_DIR / "chat_history.db"

_local = threading.local()


def _conn() -> sqlite3.Connection:
    """Thread-local SQLite connection (auto-creates tables on first use)."""
    if not getattr(_local, "conn", None):
        CHAT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(CHAT_DB_PATH), check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id    TEXT PRIMARY KEY,
                started_at    TEXT NOT NULL,
                last_active   TEXT NOT NULL,
                question_count INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS chat_turns (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL,
                ts              TEXT NOT NULL,
                question        TEXT NOT NULL,
                response        TEXT NOT NULL,
                tokens          INTEGER DEFAULT 0,
                cache_hit       INTEGER DEFAULT 0,
                branch          TEXT    DEFAULT '',
                execution_time_ms REAL  DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
            );
            CREATE INDEX IF NOT EXISTS idx_turns_session
                ON chat_turns(session_id);
            CREATE INDEX IF NOT EXISTS idx_turns_ts
                ON chat_turns(ts DESC);
        """)
        con.commit()
        _local.conn = con
    return _local.conn


class ChatStore:
    """Thread-safe persistent chat store."""

    # -- Write --

    def save_turn(
        self,
        session_id: str,
        question: str,
        response: str,
        tokens: int = 0,
        cache_hit: bool = False,
        branch: str = "",
        execution_time_ms: float = 0.0,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        con = _conn()
        con.execute(
            """
            INSERT INTO chat_sessions (session_id, started_at, last_active, question_count)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(session_id) DO UPDATE SET
                last_active    = excluded.last_active,
                question_count = question_count + 1
            """,
            (session_id, now, now),
        )
        con.execute(
            """
            INSERT INTO chat_turns
                (session_id, ts, question, response, tokens, cache_hit, branch, execution_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, now, question, response,
             tokens, int(cache_hit), branch, round(execution_time_ms, 1)),
        )
        con.commit()

    # -- Read --

    def get_recent_questions(self, n: int = 15) -> List[Dict[str, Any]]:
        """Last *n* questions across all sessions, newest first."""
        rows = _conn().execute(
            """
            SELECT ts, session_id, question, branch, tokens, cache_hit, execution_time_ms
            FROM   chat_turns
            ORDER  BY ts DESC
            LIMIT  ?
            """,
            (n,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_session_turns(self, session_id: str) -> List[Dict[str, Any]]:
        """All turns for a session, oldest first."""
        rows = _conn().execute(
            """
            SELECT ts, question, response, tokens, cache_hit, branch, execution_time_ms
            FROM   chat_turns
            WHERE  session_id = ?
            ORDER  BY ts ASC
            """,
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_sessions(self, n: int = 10) -> List[Dict[str, Any]]:
        """Recent sessions, newest first."""
        rows = _conn().execute(
            """
            SELECT session_id, started_at, last_active, question_count
            FROM   chat_sessions
            ORDER  BY last_active DESC
            LIMIT  ?
            """,
            (n,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_last_session_context(self) -> Optional[Dict[str, Any]]:
        """
        Return the last few questions from the most recent session.
        Used on startup to restore entity context for pronoun resolution.
        """
        sessions = self.list_sessions(n=1)
        if not sessions:
            return None
        sid = sessions[0]["session_id"]
        turns = self.get_session_turns(sid)
        if not turns:
            return None
        return {
            "session_id": sid,
            "last_active": sessions[0]["last_active"],
            "recent_turns": turns[-4:],  # last 4 turns
        }

    def search(self, query: str, n: int = 10) -> List[Dict[str, Any]]:
        """Full-text search over past questions."""
        pattern = f"%{query}%"
        rows = _conn().execute(
            """
            SELECT ts, session_id, question, response, branch
            FROM   chat_turns
            WHERE  question LIKE ? OR response LIKE ?
            ORDER  BY ts DESC
            LIMIT  ?
            """,
            (pattern, pattern, n),
        ).fetchall()
        return [dict(r) for r in rows]

    def total_questions(self) -> int:
        row = _conn().execute("SELECT COUNT(*) FROM chat_turns").fetchone()
        return row[0] if row else 0

    def total_sessions(self) -> int:
        row = _conn().execute("SELECT COUNT(*) FROM chat_sessions").fetchone()
        return row[0] if row else 0


# Module-level singleton
_store: Optional[ChatStore] = None


def get_chat_store() -> ChatStore:
    global _store
    if _store is None:
        _store = ChatStore()
    return _store
