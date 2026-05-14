from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import sqlite3

from driftscope.core.scope_compat import ScopeRules
from driftscope.core.schema import MemoryEntry, Scope, SupersedeLink
from driftscope.core.topic_tree import TopicTree, category_of_path, normalize_leaf_suffix

_KEEP_VALID_END = object()


class MemoryBase:
    def __init__(
        self,
        db_path: str = ":memory:",
        *,
        topic_tree: TopicTree | None = None,
        scope_rules: ScopeRules | None = None,
        embedder=None,
    ) -> None:
        self.topic_tree = topic_tree or TopicTree.load_default()
        self.scope_rules = scope_rules or ScopeRules.load_default()
        self.embedder = embedder
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self):
        with self.conn:
            yield

    def add(self, m: MemoryEntry, *, commit: bool = True) -> None:
        if m.topic_id is not None and not self.is_known_topic(m.topic_id):
            raise ValueError(f"unknown topic_id: {m.topic_id}")

        self._insert_memory(m)
        if self.embedder is not None:
            try:
                text = m.summary_for_retrieval or m.content
                vector = self.embedder.embed([text])[0]
                self.set_embedding(m.id, model=self.embedder.model, vector=vector)
            except Exception:
                import logging

                logging.getLogger(__name__).warning(
                    "embedder.embed failed for memory %s; continuing without dense vector",
                    m.id,
                )
        if commit:
            self.conn.commit()

    def get(self, id: str) -> MemoryEntry:
        row = self.conn.execute("SELECT * FROM memories WHERE id = ?", (id,)).fetchone()
        if row is None:
            raise KeyError(id)
        return self._row_to_memory(row)

    def update_state(
        self,
        id: str,
        new_state: str,
        revoked_at: datetime | None = None,
        valid_end: datetime | None | object = _KEEP_VALID_END,
        *,
        commit: bool = True,
    ) -> None:
        current = self.get(id)
        if new_state == "revoked" and revoked_at is None:
            raise ValueError("revoked state requires revoked_at")
        if new_state != "revoked":
            revoked_at = None

        self.conn.execute(
            "UPDATE memories SET state = ?, revoked_at = ?, valid_end = ? WHERE id = ?",
            (
                new_state,
                revoked_at.isoformat() if revoked_at else None,
                current.valid_time.end.isoformat() if valid_end is _KEEP_VALID_END and current.valid_time.end else (
                    valid_end.isoformat() if isinstance(valid_end, datetime) else None
                ),
                current.id,
            ),
        )
        if commit:
            self.conn.commit()

    def add_supersede_link(self, new_id: str, link: SupersedeLink, *, commit: bool = True) -> None:
        self.get(new_id)
        self.get(link.target)
        self._insert_link(new_id, link)
        if commit:
            self.conn.commit()

    def query_by_topic(self, topic_id: str, scope: Scope, time: datetime) -> list[MemoryEntry]:
        rows = self.conn.execute(
            """
            SELECT * FROM memories
            WHERE topic_id = ?
              AND state = 'active'
              AND valid_start <= ?
              AND (valid_end IS NULL OR valid_end >= ?)
            ORDER BY ingest_time DESC, id ASC
            """,
            (topic_id, time.isoformat(), time.isoformat()),
        ).fetchall()
        return [memory for row in rows if self.scope_rules.can_read(scope, (memory := self._row_to_memory(row)).scope)]

    def set_embedding(self, memory_id: str, *, model: str, vector) -> None:
        import numpy as np

        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO memory_embeddings (memory_id, model, dim, vector)
            VALUES (?, ?, ?, ?)
            """,
            (memory_id, model, int(arr.shape[0]), arr.tobytes()),
        )
        self.conn.commit()

    def get_embedding(self, memory_id: str):
        import numpy as np

        row = self.conn.execute(
            "SELECT model, dim, vector FROM memory_embeddings WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return None
        vec = np.frombuffer(row["vector"], dtype=np.float32).reshape(int(row["dim"]))
        return row["model"], vec

    def query_visible_with_vectors(self, scope: Scope, time: datetime):
        import numpy as np

        memories = self.query_visible(scope, time)
        if not memories:
            return []
        ids = [m.id for m in memories]
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"SELECT memory_id, dim, vector FROM memory_embeddings WHERE memory_id IN ({placeholders})",
            ids,
        ).fetchall()
        vectors_by_id: dict[str, object] = {}
        for row in rows:
            vectors_by_id[row["memory_id"]] = np.frombuffer(row["vector"], dtype=np.float32).reshape(int(row["dim"]))
        return [(memory, vectors_by_id.get(memory.id)) for memory in memories]

    # --- topic leaves (seed + runtime-registered) --------------------------

    def is_known_topic(self, topic_id: str) -> bool:
        if self.topic_tree.has_topic(topic_id):
            return True
        row = self.conn.execute(
            "SELECT 1 FROM topic_leaves WHERE path = ?",
            (topic_id,),
        ).fetchone()
        return row is not None

    def canonicalize_topic(
        self,
        category: str,
        leaf_suffix: str,
        *,
        similarity_threshold: float = 0.85,
    ) -> str | None:
        """Resolve (category, leaf_suffix) to a canonical topic path.

        Returns None when the category is unknown or the suffix can't be
        normalized. When an embedder is configured, a new suffix whose
        embedding is cosine-similar to an existing leaf in the same category
        (above ``similarity_threshold``) is collapsed onto that existing
        leaf; otherwise the new leaf is registered and returned verbatim.
        """
        if not self.topic_tree.has_category(category):
            return None
        suffix = normalize_leaf_suffix(leaf_suffix)
        if suffix is None:
            return None

        candidate_path = f"{category}.{suffix}"
        if self.topic_tree.has_topic(candidate_path):
            self._upsert_topic_leaf(
                category=category,
                suffix=suffix,
                path=candidate_path,
                is_seed=True,
                embedding=None,
            )
            return candidate_path

        row = self.conn.execute(
            "SELECT path FROM topic_leaves WHERE category = ? AND suffix = ?",
            (category, suffix),
        ).fetchone()
        if row is not None:
            return row["path"]

        if self.embedder is None:
            self._upsert_topic_leaf(
                category=category,
                suffix=suffix,
                path=candidate_path,
                is_seed=False,
                embedding=None,
            )
            self.conn.commit()
            return candidate_path

        try:
            new_vec = self.embedder.embed([_suffix_to_text(suffix)])[0]
        except Exception:
            import logging

            logging.getLogger(__name__).warning(
                "canonicalize_topic: embedder.embed failed for suffix=%r; registering without vector",
                suffix,
            )
            self._upsert_topic_leaf(
                category=category,
                suffix=suffix,
                path=candidate_path,
                is_seed=False,
                embedding=None,
            )
            self.conn.commit()
            return candidate_path

        self._ensure_seeds_embedded(category)

        import numpy as np

        peers = self._known_leaves_in_category_with_vectors(category)
        best_path: str | None = None
        best_cos = -1.0
        for peer_path, peer_vec in peers:
            if peer_vec is None:
                continue
            denom = float(np.linalg.norm(peer_vec)) * float(np.linalg.norm(new_vec))
            if denom <= 0.0:
                continue
            cos = float(np.dot(peer_vec, new_vec) / denom)
            if cos > best_cos:
                best_cos = cos
                best_path = peer_path

        if best_path is not None and best_cos >= similarity_threshold:
            return best_path

        self._upsert_topic_leaf(
            category=category,
            suffix=suffix,
            path=candidate_path,
            is_seed=False,
            embedding=new_vec,
        )
        self.conn.commit()
        return candidate_path

    def known_leaves_in_category(self, category: str):
        """Return [(path, vector|None)] for all leaves in a category.

        Seeds are embedded lazily on first call when an embedder is wired.
        """
        if not self.topic_tree.has_category(category):
            return []
        self._ensure_seeds_embedded(category)
        return self._known_leaves_in_category_with_vectors(category)

    def all_known_leaves(self):
        """Return [(path, category, vector|None)] across all categories.

        Seeds in every category are embedded lazily on first call when an
        embedder is wired. Used by the retriever's embedding-based
        query→leaf matcher to predict a topic when no seed keyword fires.
        """
        import numpy as np

        for category in self.topic_tree.category_ids():
            self._ensure_seeds_embedded(category)

        rows = self.conn.execute(
            "SELECT path, category, embedding_dim, embedding FROM topic_leaves"
        ).fetchall()
        result: list[tuple[str, str, object]] = []
        for row in rows:
            if row["embedding"] is None or row["embedding_dim"] is None:
                result.append((row["path"], row["category"], None))
                continue
            vec = np.frombuffer(row["embedding"], dtype=np.float32).reshape(int(row["embedding_dim"]))
            result.append((row["path"], row["category"], vec))
        return result

    def _ensure_seeds_embedded(self, category: str) -> None:
        if self.embedder is None:
            return
        for leaf in self.topic_tree.seeds_in_category(category):
            row = self.conn.execute(
                "SELECT embedding FROM topic_leaves WHERE path = ?",
                (leaf.path,),
            ).fetchone()
            if row is not None and row["embedding"] is not None:
                continue
            suffix = leaf.path.rsplit(".", 1)[-1]
            try:
                vec = self.embedder.embed([_suffix_to_text(suffix)])[0]
            except Exception:
                continue
            self._upsert_topic_leaf(
                category=category,
                suffix=suffix,
                path=leaf.path,
                is_seed=True,
                embedding=vec,
            )
        self.conn.commit()

    def _known_leaves_in_category_with_vectors(self, category: str):
        import numpy as np

        rows = self.conn.execute(
            "SELECT path, embedding_dim, embedding FROM topic_leaves WHERE category = ?",
            (category,),
        ).fetchall()
        result: list[tuple[str, object]] = []
        for row in rows:
            if row["embedding"] is None or row["embedding_dim"] is None:
                result.append((row["path"], None))
                continue
            vec = np.frombuffer(row["embedding"], dtype=np.float32).reshape(int(row["embedding_dim"]))
            result.append((row["path"], vec))
        return result

    def _upsert_topic_leaf(
        self,
        *,
        category: str,
        suffix: str,
        path: str,
        is_seed: bool,
        embedding,
    ) -> None:
        import numpy as np

        if embedding is None:
            model = None
            dim = None
            blob = None
        else:
            arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
            model = getattr(self.embedder, "model", None)
            dim = int(arr.shape[0])
            blob = arr.tobytes()
        self.conn.execute(
            """
            INSERT INTO topic_leaves (category, suffix, path, is_seed, embedding_model, embedding_dim, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(category, suffix) DO UPDATE SET
                path = excluded.path,
                is_seed = excluded.is_seed,
                embedding_model = COALESCE(excluded.embedding_model, topic_leaves.embedding_model),
                embedding_dim = COALESCE(excluded.embedding_dim, topic_leaves.embedding_dim),
                embedding = COALESCE(excluded.embedding, topic_leaves.embedding)
            """,
            (category, suffix, path, int(is_seed), model, dim, blob),
        )

    def query_visible(self, scope: Scope, time: datetime) -> list[MemoryEntry]:
        rows = self.conn.execute(
            """
            SELECT * FROM memories
            WHERE state = 'active'
              AND valid_start <= ?
              AND (valid_end IS NULL OR valid_end >= ?)
            ORDER BY ingest_time DESC, id ASC
            """,
            (time.isoformat(), time.isoformat()),
        ).fetchall()
        return [memory for row in rows if self.scope_rules.can_read(scope, (memory := self._row_to_memory(row)).scope)]

    def query_revoked_within(
        self,
        window_days: int,
        scope: Scope | None = None,
        time: datetime | None = None,
    ) -> list[MemoryEntry]:
        reference = time or datetime.now(UTC)
        lower_bound = reference - timedelta(days=window_days)
        rows = self.conn.execute(
            """
            SELECT * FROM memories
            WHERE state = 'revoked'
              AND revoked_at IS NOT NULL
              AND revoked_at >= ?
            ORDER BY revoked_at DESC, id ASC
            """,
            (lower_bound.isoformat(),),
        ).fetchall()
        items = [self._row_to_memory(row) for row in rows]
        if scope is None:
            return items
        return [memory for memory in items if memory.scope == scope]

    def get_supersede_chain(self, id: str, direction: str) -> list[MemoryEntry]:
        if direction not in {"forward", "backward"}:
            raise ValueError("direction must be 'forward' or 'backward'")

        query = (
            "SELECT target_id AS related_id FROM supersede_links WHERE source_id = ? ORDER BY target_id ASC"
            if direction == "backward"
            else "SELECT source_id AS related_id FROM supersede_links WHERE target_id = ? ORDER BY source_id ASC"
        )

        seen: set[str] = set()
        frontier = [id]
        results: list[MemoryEntry] = []
        while frontier:
            current = frontier.pop(0)
            rows = self.conn.execute(query, (current,)).fetchall()
            for row in rows:
                related_id = row["related_id"]
                if related_id in seen:
                    continue
                seen.add(related_id)
                results.append(self.get(related_id))
                frontier.append(related_id)
        return results

    def rollback(self, id: str) -> bool:
        memory = self.get(id)
        if memory.state != "revoked":
            return False
        self.conn.execute(
            "UPDATE memories SET state = 'active', revoked_at = NULL, valid_end = NULL WHERE id = ?",
            (id,),
        )
        self.conn.commit()
        return True

    def dump_json(self, path: str) -> None:
        rows = self.conn.execute("SELECT id FROM memories ORDER BY ingest_time ASC, id ASC").fetchall()
        payload = {
            "memories": [self.get(row["id"]).model_dump(mode="json") for row in rows],
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_json(self, path: str) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        with self.conn:
            self.conn.execute("DELETE FROM supersede_links")
            self.conn.execute("DELETE FROM memories")
        for item in payload.get("memories", []):
            self.add(MemoryEntry.model_validate(item))

    def _create_schema(self) -> None:
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    type TEXT NOT NULL,
                    topic_id TEXT,
                    scope_kind TEXT NOT NULL,
                    scope_ref TEXT,
                    src TEXT NOT NULL,
                    origin_role TEXT NOT NULL DEFAULT 'user',
                    source_kind TEXT NOT NULL DEFAULT 'explicit',
                    conf_json TEXT NOT NULL,
                    valid_start TEXT NOT NULL,
                    valid_end TEXT,
                    ingest_time TEXT NOT NULL,
                    state TEXT NOT NULL,
                    revoked_at TEXT,
                    sensitive INTEGER NOT NULL DEFAULT 0,
                    summary_for_retrieval TEXT,
                    event_time TEXT,
                    evidence TEXT,
                    importance REAL,
                    sensitivity TEXT,
                    ttl_days INTEGER
                );

                CREATE TABLE IF NOT EXISTS supersede_links (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    transition_type TEXT NOT NULL,
                    PRIMARY KEY (source_id, target_id),
                    FOREIGN KEY(source_id) REFERENCES memories(id) ON DELETE CASCADE,
                    FOREIGN KEY(target_id) REFERENCES memories(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_memories_topic ON memories(topic_id);
                CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope_kind, scope_ref);
                CREATE INDEX IF NOT EXISTS idx_memories_state ON memories(state);

                CREATE TABLE IF NOT EXISTS topic_leaves (
                    category TEXT NOT NULL,
                    suffix TEXT NOT NULL,
                    path TEXT NOT NULL UNIQUE,
                    is_seed INTEGER NOT NULL DEFAULT 0,
                    embedding_model TEXT,
                    embedding_dim INTEGER,
                    embedding BLOB,
                    PRIMARY KEY (category, suffix)
                );

                CREATE INDEX IF NOT EXISTS idx_topic_leaves_category ON topic_leaves(category);
                """
            )
            self._ensure_new_columns()

    def _ensure_new_columns(self) -> None:
        existing = {row["name"] for row in self.conn.execute("PRAGMA table_info(memories)").fetchall()}
        additions = [
            ("event_time", "TEXT"),
            ("evidence", "TEXT"),
            ("importance", "REAL"),
            ("sensitivity", "TEXT"),
            ("ttl_days", "INTEGER"),
        ]
        for column, col_type in additions:
            if column not in existing:
                self.conn.execute(f"ALTER TABLE memories ADD COLUMN {column} {col_type}")

    def _insert_memory(self, m: MemoryEntry) -> None:
        self.conn.execute(
            """
            INSERT INTO memories (
                id, content, type, topic_id, scope_kind, scope_ref, src,
                origin_role, source_kind,
                conf_json, valid_start, valid_end, ingest_time, state,
                revoked_at, sensitive, summary_for_retrieval,
                event_time, evidence, importance, sensitivity, ttl_days
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                m.id,
                m.content,
                m.type,
                m.topic_id,
                m.scope.kind,
                m.scope.ref,
                m.src,
                m.origin_role,
                m.source_kind,
                json.dumps(m.conf.model_dump(mode="json")),
                m.valid_time.start.isoformat(),
                m.valid_time.end.isoformat() if m.valid_time.end else None,
                m.ingest_time.isoformat(),
                m.state,
                m.revoked_at.isoformat() if m.revoked_at else None,
                int(m.sensitive),
                m.summary_for_retrieval,
                m.event_time.isoformat() if m.event_time else None,
                m.evidence,
                m.importance,
                m.sensitivity,
                m.ttl_days,
            ),
        )
        for link in m.supersedes:
            self._insert_link(m.id, link)

    def _insert_link(self, source_id: str, link: SupersedeLink) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO supersede_links (source_id, target_id, mode, transition_type)
            VALUES (?, ?, ?, ?)
            """,
            (source_id, link.target, link.mode, link.transition_type),
        )

    def _row_to_memory(self, row: sqlite3.Row) -> MemoryEntry:
        link_rows = self.conn.execute(
            """
            SELECT target_id, mode, transition_type
            FROM supersede_links
            WHERE source_id = ?
            ORDER BY target_id ASC
            """,
            (row["id"],),
        ).fetchall()
        return MemoryEntry.model_validate(
            {
                "id": row["id"],
                "content": row["content"],
                "type": row["type"],
                "topic_id": row["topic_id"],
                "scope": {
                    "kind": row["scope_kind"],
                    "ref": row["scope_ref"],
                },
                "src": row["src"],
                "origin_role": row["origin_role"],
                "source_kind": row["source_kind"],
                "conf": json.loads(row["conf_json"]),
                "valid_time": {
                    "start": row["valid_start"],
                    "end": row["valid_end"],
                },
                "ingest_time": row["ingest_time"],
                "state": row["state"],
                "revoked_at": row["revoked_at"],
                "supersedes": [
                    {
                        "target": item["target_id"],
                        "mode": item["mode"],
                        "transition_type": item["transition_type"],
                    }
                    for item in link_rows
                ],
                "sensitive": bool(row["sensitive"]),
                "summary_for_retrieval": row["summary_for_retrieval"],
                "event_time": _optional_column(row, "event_time"),
                "evidence": _optional_column(row, "evidence"),
                "importance": _optional_column(row, "importance"),
                "sensitivity": _optional_column(row, "sensitivity"),
                "ttl_days": _optional_column(row, "ttl_days"),
            }
        )


def _optional_column(row: sqlite3.Row, name: str):
    try:
        return row[name]
    except (IndexError, KeyError):
        return None


def _suffix_to_text(suffix: str) -> str:
    return suffix.replace("_", " ").strip() or suffix
