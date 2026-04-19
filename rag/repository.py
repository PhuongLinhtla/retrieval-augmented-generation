from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3

import numpy as np

from .schemas import ChunkRecord


class MetadataRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    def _column_exists(self, conn: sqlite3.Connection, table: str, column: str) -> bool:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(str(row["name"]) == column for row in rows)

    def _init_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_path TEXT NOT NULL UNIQUE,
            source_name TEXT NOT NULL,
            file_type TEXT NOT NULL,
            doc_type TEXT NOT NULL DEFAULT 'book',
            title TEXT NOT NULL DEFAULT '',
            subject TEXT NOT NULL DEFAULT '',
            grade TEXT NOT NULL DEFAULT '',
            chapter TEXT NOT NULL DEFAULT '',
            published_year TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT '',
            total_pages INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            page_number INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

        CREATE TABLE IF NOT EXISTS parent_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            page_number INTEGER NOT NULL,
            parent_index INTEGER NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            section_path TEXT NOT NULL DEFAULT '',
            keywords TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
            UNIQUE(document_id, parent_index)
        );

        CREATE INDEX IF NOT EXISTS idx_parent_chunks_document_id ON parent_chunks(document_id);

        CREATE TABLE IF NOT EXISTS child_parent_map (
            child_chunk_id INTEGER PRIMARY KEY,
            parent_chunk_id INTEGER NOT NULL,
            FOREIGN KEY(child_chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
            FOREIGN KEY(parent_chunk_id) REFERENCES parent_chunks(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_child_parent_map_parent_id ON child_parent_map(parent_chunk_id);

        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id INTEGER PRIMARY KEY,
            dim INTEGER NOT NULL,
            vector BLOB NOT NULL,
            FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
        );
        """
        with self._connect() as conn:
            conn.executescript(schema)
            if not self._column_exists(conn, "documents", "doc_type"):
                conn.execute(
                    "ALTER TABLE documents ADD COLUMN doc_type TEXT NOT NULL DEFAULT 'book'"
                )
            if not self._column_exists(conn, "documents", "title"):
                conn.execute("ALTER TABLE documents ADD COLUMN title TEXT NOT NULL DEFAULT ''")
            if not self._column_exists(conn, "documents", "subject"):
                conn.execute("ALTER TABLE documents ADD COLUMN subject TEXT NOT NULL DEFAULT ''")
            if not self._column_exists(conn, "documents", "grade"):
                conn.execute("ALTER TABLE documents ADD COLUMN grade TEXT NOT NULL DEFAULT ''")
            if not self._column_exists(conn, "documents", "chapter"):
                conn.execute("ALTER TABLE documents ADD COLUMN chapter TEXT NOT NULL DEFAULT ''")
            if not self._column_exists(conn, "documents", "published_year"):
                conn.execute(
                    "ALTER TABLE documents ADD COLUMN published_year TEXT NOT NULL DEFAULT ''"
                )
            if not self._column_exists(conn, "documents", "source"):
                conn.execute("ALTER TABLE documents ADD COLUMN source TEXT NOT NULL DEFAULT ''")

            if not self._column_exists(conn, "parent_chunks", "section_path"):
                conn.execute(
                    "ALTER TABLE parent_chunks ADD COLUMN section_path TEXT NOT NULL DEFAULT ''"
                )
            if not self._column_exists(conn, "parent_chunks", "keywords"):
                conn.execute(
                    "ALTER TABLE parent_chunks ADD COLUMN keywords TEXT NOT NULL DEFAULT ''"
                )
            conn.commit()

    def upsert_document(
        self,
        source_path: str,
        source_name: str,
        file_type: str,
        doc_type: str,
        total_pages: int,
        title: str = "",
        subject: str = "",
        grade: str = "",
        chapter: str = "",
        published_year: str = "",
        source: str = "",
    ) -> int:
        normalized_title = title.strip() or source_name.strip()
        normalized_source = source.strip() or source_name.strip()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    source_path,
                    source_name,
                    file_type,
                    doc_type,
                    title,
                    subject,
                    grade,
                    chapter,
                    published_year,
                    source,
                    total_pages
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_path) DO UPDATE SET
                    source_name=excluded.source_name,
                    file_type=excluded.file_type,
                    doc_type=excluded.doc_type,
                    title=excluded.title,
                    subject=excluded.subject,
                    grade=excluded.grade,
                    chapter=excluded.chapter,
                    published_year=excluded.published_year,
                    source=excluded.source,
                    total_pages=excluded.total_pages,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    source_path,
                    source_name,
                    file_type,
                    doc_type,
                    normalized_title,
                    subject.strip(),
                    grade.strip(),
                    chapter.strip(),
                    published_year.strip(),
                    normalized_source,
                    max(1, total_pages),
                ),
            )
            row = conn.execute(
                "SELECT id FROM documents WHERE source_path = ?",
                (source_path,),
            ).fetchone()
            conn.commit()

        if row is None:
            raise RuntimeError("Could not upsert document")
        return int(row["id"])

    def delete_chunks_for_document(self, document_id: int) -> list[int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM chunks WHERE document_id = ?",
                (document_id,),
            ).fetchall()
            chunk_ids = [int(row["id"]) for row in rows]

            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.commit()

        return chunk_ids

    def delete_parent_chunks_for_document(self, document_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM parent_chunks WHERE document_id = ?", (document_id,))
            conn.commit()

    def insert_parent_chunks(
        self,
        document_id: int,
        parent_rows: list[tuple],
    ) -> dict[int, int]:
        if not parent_rows:
            return {}

        mapping: dict[int, int] = {}
        with self._connect() as conn:
            for row in parent_rows:
                if len(row) >= 6:
                    page_number, parent_index, title, section_path, keywords, content = row[:6]
                elif len(row) == 5:
                    page_number, parent_index, title, section_path, content = row
                    keywords = ""
                else:
                    page_number, parent_index, title, content = row[:4]
                    section_path = ""
                    keywords = ""

                cursor = conn.execute(
                    """
                    INSERT INTO parent_chunks (
                        document_id,
                        page_number,
                        parent_index,
                        title,
                        section_path,
                        keywords,
                        content
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        int(page_number),
                        int(parent_index),
                        str(title or ""),
                        str(section_path or ""),
                        str(keywords or ""),
                        str(content or ""),
                    ),
                )
                mapping[int(parent_index)] = int(cursor.lastrowid)
            conn.commit()
        return mapping

    def insert_child_parent_map(self, mappings: list[tuple[int, int]]) -> None:
        if not mappings:
            return

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO child_parent_map (child_chunk_id, parent_chunk_id)
                VALUES (?, ?)
                """,
                mappings,
            )
            conn.commit()

    def insert_chunks(
        self,
        document_id: int,
        chunk_rows: list[tuple[int, int, str]],
    ) -> list[int]:
        inserted_ids: list[int] = []
        with self._connect() as conn:
            for page_number, chunk_index, content in chunk_rows:
                cursor = conn.execute(
                    """
                    INSERT INTO chunks (document_id, page_number, chunk_index, content)
                    VALUES (?, ?, ?, ?)
                    """,
                    (document_id, page_number, chunk_index, content),
                )
                inserted_ids.append(int(cursor.lastrowid))
            conn.commit()
        return inserted_ids

    def insert_embeddings(self, chunk_ids: list[int], vectors: np.ndarray) -> None:
        if len(chunk_ids) == 0:
            return
        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2D array")
        if vectors.shape[0] != len(chunk_ids):
            raise ValueError("chunk_ids and vectors size mismatch")

        dim = int(vectors.shape[1])
        rows = [
            (int(chunk_id), dim, sqlite3.Binary(vectors[idx].astype(np.float32).tobytes()))
            for idx, chunk_id in enumerate(chunk_ids)
        ]

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings (chunk_id, dim, vector)
                VALUES (?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def fetch_all_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id, dim, vector FROM embeddings ORDER BY chunk_id"
            ).fetchall()

        if not rows:
            return np.empty((0,), dtype=np.int64), np.empty((0, 0), dtype=np.float32)

        dim = int(rows[0]["dim"])
        ids = np.zeros((len(rows),), dtype=np.int64)
        vectors = np.zeros((len(rows), dim), dtype=np.float32)

        for idx, row in enumerate(rows):
            ids[idx] = int(row["chunk_id"])
            vectors[idx] = np.frombuffer(row["vector"], dtype=np.float32)

        return ids, vectors

    def get_chunks_by_ids(self, chunk_ids: list[int]) -> list[ChunkRecord]:
        if not chunk_ids:
            return []

        placeholders = ",".join("?" for _ in chunk_ids)
        query = f"""
            SELECT
                c.id AS chunk_id,
                c.document_id,
                d.source_name,
                d.source_path,
                d.source,
                d.file_type,
                d.doc_type,
                d.title,
                d.subject,
                d.grade,
                d.chapter,
                d.published_year,
                c.page_number,
                c.chunk_index,
                c.content,
                m.parent_chunk_id,
                p.page_number AS parent_page_number,
                p.parent_index,
                p.title AS parent_title,
                p.section_path,
                p.keywords,
                p.content AS parent_content
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            LEFT JOIN child_parent_map m ON m.child_chunk_id = c.id
            LEFT JOIN parent_chunks p ON p.id = m.parent_chunk_id
            WHERE c.id IN ({placeholders})
        """

        with self._connect() as conn:
            rows = conn.execute(query, chunk_ids).fetchall()

        by_id = {
            int(row["chunk_id"]): ChunkRecord(
                chunk_id=int(row["chunk_id"]),
                document_id=int(row["document_id"]),
                source_name=str(row["source_name"]),
                source_path=str(row["source_path"]),
                source=str(row["source"] or row["source_name"]),
                page_number=int(row["page_number"]),
                chunk_index=int(row["chunk_index"]),
                content=str(row["content"]),
                file_type=str(row["file_type"]),
                doc_type=str(row["doc_type"]),
                title=str(row["title"] or ""),
                subject=str(row["subject"] or ""),
                grade=str(row["grade"] or ""),
                chapter=str(row["chapter"] or ""),
                published_year=str(row["published_year"] or ""),
                parent_chunk_id=(
                    int(row["parent_chunk_id"])
                    if row["parent_chunk_id"] is not None
                    else None
                ),
                parent_page_number=(
                    int(row["parent_page_number"])
                    if row["parent_page_number"] is not None
                    else None
                ),
                parent_index=(
                    int(row["parent_index"])
                    if row["parent_index"] is not None
                    else None
                ),
                parent_title=str(row["parent_title"] or ""),
                parent_content=str(row["parent_content"] or ""),
                section_path=str(row["section_path"] or ""),
                keywords=str(row["keywords"] or ""),
            )
            for row in rows
        }

        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    def list_documents(self) -> list[dict[str, str | int]]:
        query = """
            SELECT
                d.id,
                d.source_name,
                d.source_path,
                d.source,
                d.file_type,
                d.doc_type,
                d.title,
                d.subject,
                d.grade,
                d.chapter,
                d.published_year,
                d.total_pages,
                d.updated_at,
                COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id
            ORDER BY d.updated_at DESC
        """

        with self._connect() as conn:
            rows = conn.execute(query).fetchall()

        return [
            {
                "id": int(row["id"]),
                "source_name": str(row["source_name"]),
                "source_path": str(row["source_path"]),
                "source": str(row["source"] or row["source_name"]),
                "file_type": str(row["file_type"]),
                "doc_type": str(row["doc_type"]),
                "title": str(row["title"] or ""),
                "subject": str(row["subject"] or ""),
                "grade": str(row["grade"] or ""),
                "chapter": str(row["chapter"] or ""),
                "published_year": str(row["published_year"] or ""),
                "total_pages": int(row["total_pages"]),
                "chunk_count": int(row["chunk_count"]),
                "updated_at": str(row["updated_at"]),
            }
            for row in rows
        ]

    def fetch_sparse_corpus(self) -> list[dict[str, str | int | None]]:
        query = """
            SELECT
                c.id AS chunk_id,
                c.content,
                c.page_number,
                d.id AS document_id,
                d.source_name,
                d.source,
                d.doc_type,
                d.title,
                d.subject,
                d.grade,
                d.chapter,
                d.published_year,
                p.id AS parent_chunk_id,
                p.title AS parent_title,
                p.section_path,
                p.keywords
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            LEFT JOIN child_parent_map m ON m.child_chunk_id = c.id
            LEFT JOIN parent_chunks p ON p.id = m.parent_chunk_id
            ORDER BY c.id
        """

        with self._connect() as conn:
            rows = conn.execute(query).fetchall()

        return [
            {
                "chunk_id": int(row["chunk_id"]),
                "content": str(row["content"]),
                "page_number": int(row["page_number"]),
                "document_id": int(row["document_id"]),
                "source_name": str(row["source_name"]),
                "source": str(row["source"] or row["source_name"]),
                "doc_type": str(row["doc_type"] or ""),
                "title": str(row["title"] or ""),
                "subject": str(row["subject"] or ""),
                "grade": str(row["grade"] or ""),
                "chapter": str(row["chapter"] or ""),
                "published_year": str(row["published_year"] or ""),
                "parent_chunk_id": (
                    int(row["parent_chunk_id"])
                    if row["parent_chunk_id"] is not None
                    else None
                ),
                "parent_title": str(row["parent_title"] or ""),
                "section_path": str(row["section_path"] or ""),
                "keywords": str(row["keywords"] or ""),
            }
            for row in rows
        ]

    def count_chunks(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS total FROM chunks").fetchone()
        return int(row["total"]) if row else 0
