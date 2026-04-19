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

    def _init_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_path TEXT NOT NULL UNIQUE,
            source_name TEXT NOT NULL,
            file_type TEXT NOT NULL,
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

        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id INTEGER PRIMARY KEY,
            dim INTEGER NOT NULL,
            vector BLOB NOT NULL,
            FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
        );
        """
        with self._connect() as conn:
            conn.executescript(schema)
            conn.commit()

    def upsert_document(
        self,
        source_path: str,
        source_name: str,
        file_type: str,
        total_pages: int,
    ) -> int:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (source_path, source_name, file_type, total_pages)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source_path) DO UPDATE SET
                    source_name=excluded.source_name,
                    file_type=excluded.file_type,
                    total_pages=excluded.total_pages,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (source_path, source_name, file_type, max(1, total_pages)),
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
                c.page_number,
                c.chunk_index,
                c.content
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
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
                page_number=int(row["page_number"]),
                chunk_index=int(row["chunk_index"]),
                content=str(row["content"]),
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
                d.file_type,
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
                "file_type": str(row["file_type"]),
                "total_pages": int(row["total_pages"]),
                "chunk_count": int(row["chunk_count"]),
                "updated_at": str(row["updated_at"]),
            }
            for row in rows
        ]

    def count_chunks(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS total FROM chunks").fetchone()
        return int(row["total"]) if row else 0
