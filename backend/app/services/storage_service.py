from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Iterable, List, Optional

from ..models.entities import CitationEntity, DocumentEntity, DraftVersion
from .research_service import SourceDocument


class StorageService:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                url TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS draft_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                outline TEXT NOT NULL,
                research_notes TEXT NOT NULL,
                draft TEXT NOT NULL,
                review TEXT NOT NULL,
                revised TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS citations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                title TEXT NOT NULL,
                url TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def upsert_documents(self, docs: Iterable[SourceDocument]) -> List[DocumentEntity]:
        now = datetime.utcnow().isoformat()
        cur = self._conn.cursor()
        entities: List[DocumentEntity] = []
        for doc in docs:
            cur.execute(
                """
                INSERT INTO documents (doc_id, title, content, url, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    title=excluded.title,
                    content=excluded.content,
                    url=excluded.url
                """,
                (doc.doc_id, doc.title, doc.content, doc.url, now),
            )
            entities.append(
                DocumentEntity(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    content=doc.content,
                    url=doc.url,
                    created_at=now,
                )
            )
        self._conn.commit()
        return entities

    def list_documents(self) -> List[DocumentEntity]:
        cur = self._conn.cursor()
        rows = cur.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [
            DocumentEntity(
                doc_id=row["doc_id"],
                title=row["title"],
                content=row["content"],
                url=row["url"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_documents_by_ids(self, doc_ids: List[str]) -> List[DocumentEntity]:
        if not doc_ids:
            return []
        placeholders = ",".join("?" for _ in doc_ids)
        cur = self._conn.cursor()
        rows = cur.execute(
            f"SELECT * FROM documents WHERE doc_id IN ({placeholders})",
            tuple(doc_ids),
        ).fetchall()
        by_id = {
            row["doc_id"]: DocumentEntity(
                doc_id=row["doc_id"],
                title=row["title"],
                content=row["content"],
                url=row["url"],
                created_at=row["created_at"],
            )
            for row in rows
        }
        return [by_id[doc_id] for doc_id in doc_ids if doc_id in by_id]

    def search_documents(self, query: str, top_k: int = 5) -> List[DocumentEntity]:
        pattern = f"%{query}%"
        cur = self._conn.cursor()
        rows = cur.execute(
            """
            SELECT * FROM documents
            WHERE title LIKE ? OR content LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (pattern, pattern, top_k),
        ).fetchall()
        return [
            DocumentEntity(
                doc_id=row["doc_id"],
                title=row["title"],
                content=row["content"],
                url=row["url"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def delete_document(self, doc_id: str) -> bool:
        cur = self._conn.cursor()
        cur.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def save_draft_version(
        self,
        *,
        topic: str,
        outline: str,
        research_notes: str,
        draft: str,
        review: str,
        revised: str,
    ) -> int:
        now = datetime.utcnow().isoformat()
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO draft_versions (topic, outline, research_notes, draft, review, revised, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (topic, outline, research_notes, draft, review, revised, now),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def list_draft_versions(self, limit: Optional[int] = None) -> List[DraftVersion]:
        cur = self._conn.cursor()
        query = "SELECT * FROM draft_versions ORDER BY created_at DESC"
        if limit:
            query += " LIMIT ?"
            rows = cur.execute(query, (limit,)).fetchall()
        else:
            rows = cur.execute(query).fetchall()
        return [
            DraftVersion(
                version_id=row["id"],
                topic=row["topic"],
                outline=row["outline"],
                research_notes=row["research_notes"],
                draft=row["draft"],
                review=row["review"],
                revised=row["revised"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_draft_version(self, version_id: int) -> Optional[DraftVersion]:
        cur = self._conn.cursor()
        row = cur.execute(
            "SELECT * FROM draft_versions WHERE id = ?",
            (version_id,),
        ).fetchone()
        if not row:
            return None
        return DraftVersion(
            version_id=row["id"],
            topic=row["topic"],
            outline=row["outline"],
            research_notes=row["research_notes"],
            draft=row["draft"],
            review=row["review"],
            revised=row["revised"],
            created_at=row["created_at"],
        )

    def get_previous_version(self, version_id: int) -> Optional[DraftVersion]:
        cur = self._conn.cursor()
        row = cur.execute(
            """
            SELECT * FROM draft_versions
            WHERE id < ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (version_id,),
        ).fetchone()
        if not row:
            return None
        return DraftVersion(
            version_id=row["id"],
            topic=row["topic"],
            outline=row["outline"],
            research_notes=row["research_notes"],
            draft=row["draft"],
            review=row["review"],
            revised=row["revised"],
            created_at=row["created_at"],
        )

    def delete_draft_version(self, version_id: int) -> bool:
        cur = self._conn.cursor()
        cur.execute("DELETE FROM draft_versions WHERE id = ?", (version_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def save_citations(self, citations: Iterable[CitationEntity]) -> List[CitationEntity]:
        now = datetime.utcnow().isoformat()
        cur = self._conn.cursor()
        saved: List[CitationEntity] = []
        for citation in citations:
            cur.execute(
                """
                INSERT INTO citations (label, title, url, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (citation.label, citation.title, citation.url, now),
            )
            saved.append(
                CitationEntity(
                    citation_id=int(cur.lastrowid),
                    label=citation.label,
                    title=citation.title,
                    url=citation.url,
                    created_at=now,
                )
            )
        self._conn.commit()
        return saved
