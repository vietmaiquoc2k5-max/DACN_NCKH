"""Transactional private SQLite persistence and authenticated role boundaries."""
from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import re
import secrets
import sqlite3
import time
import uuid
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Iterator

from ground_truth_studio.config import Settings
from ground_truth_studio.integrity_service import catalog, frozen, sha256
from ground_truth_studio.models import utc_now

CSV_FIELDS = {
    "positive_annotator_02.csv": ["video_id", "relative_path", "source_sha256", "reviewer_id", "reviewed_at_utc", "annotation_events_json", "evidence_ref", "signoff"],
    "positive_adjudication.csv": ["gt_index", "gt_snapshot_sha256", "video_id", "label", "start_sec", "end_sec", "reviewer_02_event_index", "approval", "adjudicator_id", "adjudicated_at_utc", "evidence_ref", "notes"],
    "negative_annotator_01.csv": ["video_id", "relative_path", "source_sha256", "reviewer_id", "reviewed_at_utc", "decision", "fire_intervals_json", "smoke_intervals_json", "evidence_ref", "signoff"],
    "negative_annotator_02.csv": ["video_id", "relative_path", "source_sha256", "reviewer_id", "reviewed_at_utc", "decision", "fire_intervals_json", "smoke_intervals_json", "evidence_ref", "signoff"],
    "negative_adjudication.csv": ["video_id", "source_sha256", "approval", "adjudicator_id", "adjudicated_at_utc", "evidence_ref", "notes"],
}


class ConflictError(ValueError):
    """A revision or external file changed; no data has been overwritten."""


class ReviewRepository:
    """Own private storage; all user-facing reads require an opaque login token."""

    def __init__(self, settings: Settings):
        self.settings = settings
        frozen(settings)
        settings.private.mkdir(parents=True, exist_ok=True)
        self.path = settings.private / "ground_truth_studio.sqlite3"
        with self.connection() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, person_id TEXT UNIQUE NOT NULL,
              role TEXT NOT NULL, identity_ref TEXT NOT NULL, salt TEXT NOT NULL, password TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS auth_sessions(token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, expires REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS videos(id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS review_sessions(id TEXT PRIMARY KEY, video_id TEXT NOT NULL,
              user_id TEXT NOT NULL, slot TEXT NOT NULL, status TEXT NOT NULL, revision INTEGER NOT NULL,
              source TEXT, payload TEXT NOT NULL, submitted_at TEXT);
            CREATE TABLE IF NOT EXISTS review_heads(video_id TEXT NOT NULL,slot TEXT NOT NULL,
              review_id TEXT NOT NULL,PRIMARY KEY(video_id,slot));
            CREATE TABLE IF NOT EXISTS adjudications(id TEXT PRIMARY KEY, video_id TEXT NOT NULL,
              user_id TEXT NOT NULL, decision TEXT NOT NULL, data TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS csv_snapshots(id INTEGER PRIMARY KEY, filename TEXT NOT NULL,
              sha256 TEXT NOT NULL, content TEXT NOT NULL, imported_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS audit_log(id INTEGER PRIMARY KEY, actor TEXT NOT NULL,
              action TEXT NOT NULL, ref TEXT NOT NULL, data TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS exports(id TEXT PRIMARY KEY, data TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS immutable_submission BEFORE UPDATE ON review_sessions
              WHEN OLD.status='SUBMITTED' BEGIN SELECT RAISE(ABORT,'Submitted review is locked'); END;
            """)
            for row in catalog(settings):
                db.execute("INSERT INTO videos VALUES(?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data",
                           (row["video_id"], json.dumps(row)))
        self.snapshot_csv()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Use WAL, bounded busy waiting and a transaction per write operation."""
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=10000")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def log(db: sqlite3.Connection, actor: str, action: str, ref: str, data: Any) -> None:
        db.execute("INSERT INTO audit_log(actor,action,ref,data,created_at) VALUES(?,?,?,?,?)",
                   (actor, action, ref, json.dumps(data, ensure_ascii=False), utc_now()))

    def provision(self, reviewer_id: str, person_id: str, role: str, password: str, identity_ref: str) -> None:
        """Operator-only provisioning; the operator must actually verify human identity."""
        if not re.fullmatch(r"[A-Za-z0-9_.-]{3,64}", reviewer_id) or not person_id.strip() or not identity_ref.strip():
            raise ValueError("Reviewer ID, stable real-person ID and identity-check reference required")
        if role not in {"annotator_01", "annotator_02", "adjudicator", "admin"} or len(password) < 12:
            raise ValueError("Fixed role and password of at least 12 characters required")
        salt = secrets.token_hex(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 260_000).hex()
        with self.connection() as db:
            db.execute("INSERT INTO users VALUES(?,?,?,?,?,?)", (reviewer_id, person_id.strip(), role, identity_ref, salt, digest))
            self.log(db, "operator", "PROVISION_ACCOUNT", reviewer_id, {"role": role, "identity_ref": identity_ref})

    def login(self, reviewer_id: str, password: str) -> str:
        """Authenticate a fixed-role account and return a time-limited capability."""
        with self.connection() as db:
            row = db.execute("SELECT * FROM users WHERE id=?", (reviewer_id,)).fetchone()
            salt = row["salt"] if row else "00" * 16
            digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 260_000).hex()
            if row is None or not hmac.compare_digest(digest, row["password"]):
                raise PermissionError("Invalid credentials")
            token = secrets.token_urlsafe(32)
            db.execute("INSERT INTO auth_sessions VALUES(?,?,?)", (hashlib.sha256(token.encode()).hexdigest(), reviewer_id, time.time() + 12 * 3600))
            self.log(db, reviewer_id, "LOGIN", reviewer_id, {})
            return token

    def actor(self, token: str, roles: set[str] | None = None) -> dict[str, Any]:
        """Authorize server-side; changing a display name cannot change an identity."""
        with self.connection() as db:
            row = db.execute("SELECT u.id,u.person_id,u.role,u.identity_ref FROM auth_sessions s JOIN users u ON s.user_id=u.id WHERE s.token_hash=? AND s.expires>?",
                             (hashlib.sha256(token.encode()).hexdigest(), time.time())).fetchone()
        if row is None or (roles is not None and row["role"] not in roles):
            raise PermissionError("This session is not authorized")
        return dict(row)

    def logout(self, token: str) -> None:
        with self.connection() as db:
            db.execute("DELETE FROM auth_sessions WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),))

    def videos(self) -> list[dict[str, Any]]:
        """Return source identity only; no catalog onset, historical notes or model output."""
        with self.connection() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT data FROM videos ORDER BY id")]

    def video(self, video_id: str) -> dict[str, Any]:
        rows = {row["video_id"]: row for row in self.videos()}
        if video_id not in rows:
            raise ValueError("Unknown catalog video ID")
        return rows[video_id]

    def own_reviews(self, token: str) -> list[dict[str, Any]]:
        actor = self.actor(token)
        with self.connection() as db:
            reviews = [self.decode_review(row) for row in db.execute("SELECT r.* FROM review_sessions r JOIN review_heads h ON h.review_id=r.id WHERE user_id=?", (actor["id"],))]
            for review in reviews:
                latest = db.execute("SELECT decision,data FROM adjudications WHERE video_id=? ORDER BY rowid DESC LIMIT 1", (review["video_id"],)).fetchone()
                review["adjudication_status"] = latest["decision"] if (latest and review["status"] == "SUBMITTED"
                    and review["id"] in json.loads(latest["data"])["review_ids"]) else None
            return reviews

    @staticmethod
    def decode_review(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        result["source"] = json.loads(result["source"]) if result["source"] else None
        return result

    def review(self, token: str, review_id: str) -> dict[str, Any]:
        actor = self.actor(token)
        with self.connection() as db:
            row = db.execute("SELECT * FROM review_sessions WHERE id=? AND user_id=?", (review_id, actor["id"])).fetchone()
        if row is None:
            raise PermissionError("Only your own review is visible here")
        return self.decode_review(row)

    def start_review(self, token: str, video_id: str) -> dict[str, Any]:
        actor = self.actor(token, {"annotator_01", "annotator_02"})
        self.video(video_id)
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT r.* FROM review_sessions r JOIN review_heads h ON h.review_id=r.id WHERE h.video_id=? AND h.slot=?", (video_id, actor["role"])).fetchone()
            latest = db.execute("SELECT decision,data FROM adjudications WHERE video_id=? ORDER BY rowid DESC LIMIT 1", (video_id,)).fetchone()
            reopen = bool(row and row["status"] == "SUBMITTED" and latest and latest["decision"] == "REQUIRE_REVIEW"
                          and row["id"] in json.loads(latest["data"])["review_ids"])
            if row and row["user_id"] != actor["id"]:
                raise PermissionError("This review slot is assigned to another registered account")
            if row and not reopen:
                review_id = row["id"]
            else:
                review_id = str(uuid.uuid4())
                payload = json.loads(row["payload"]) if reopen else {"events": [], "uncertain_intervals": [], "decision": "", "evidence_ref": "", "notes": "", "attest_full_view": False}
                payload["attest_full_view"] = False
                db.execute("INSERT INTO review_sessions VALUES(?,?,?,?,?,?,?,?,?)",
                           (review_id, video_id, actor["id"], actor["role"], "DRAFT", 0, None,
                            json.dumps(payload), None))
                db.execute("INSERT INTO review_heads VALUES(?,?,?) ON CONFLICT(video_id,slot) DO UPDATE SET review_id=excluded.review_id", (video_id, actor["role"], review_id))
                self.log(db, actor["id"], "START_DRAFT", review_id, {})
        return self.review(token, review_id)

    def save(self, token: str, review_id: str, revision: int, payload: dict[str, Any], source: dict[str, Any] | None,
             *, submit: bool = False) -> dict[str, Any]:
        """Compare-and-swap drafts; a second process cannot silently overwrite progress."""
        actor = self.actor(token, {"annotator_01", "annotator_02"})
        with self.connection() as db:
            cursor = db.execute("UPDATE review_sessions SET revision=revision+1,payload=?,source=?,status=?,submitted_at=? WHERE id=? AND user_id=? AND revision=? AND status='DRAFT'",
                                (json.dumps(payload, ensure_ascii=False), json.dumps(source) if source else None,
                                 "SUBMITTED" if submit else "DRAFT", utc_now() if submit else None,
                                 review_id, actor["id"], revision))
            if cursor.rowcount != 1:
                raise ConflictError("Review is locked or changed in another session; reload before editing")
            self.log(db, actor["id"], "SUBMIT_REVIEW" if submit else "SAVE_DRAFT", review_id,
                     {"revision": revision + 1, "payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()})
        return self.review(token, review_id)

    def submitted_pair(self, token: str, video_id: str) -> list[dict[str, Any]]:
        actor = self.actor(token, {"adjudicator"})
        with self.connection() as db:
            rows = list(db.execute("SELECT r.*,u.person_id FROM review_sessions r JOIN review_heads h ON h.review_id=r.id JOIN users u ON u.id=r.user_id WHERE r.video_id=? AND r.status='SUBMITTED' ORDER BY r.slot", (video_id,)))
            latest = db.execute("SELECT decision,data FROM adjudications WHERE video_id=? ORDER BY rowid DESC LIMIT 1", (video_id,)).fetchone()
        if latest and latest["decision"] == "REQUIRE_REVIEW" and any(row["id"] in json.loads(latest["data"])["review_ids"] for row in rows):
            raise PermissionError("Both requested re-reviews must be independently submitted")
        if len(rows) != 2 or rows[0]["person_id"] == rows[1]["person_id"] or actor["person_id"] in {row["person_id"] for row in rows}:
            raise PermissionError("Two locked reviews by distinct registered people are required before adjudication")
        return [self.decode_review(row) for row in rows]

    def historical(self, token: str, video_id: str) -> list[dict[str, Any]]:
        self.submitted_pair(token, video_id)
        _, rows = frozen(self.settings)
        return [dict(row, historical_event_id=f"GT11-{index}", gt_index=index) for index, row in enumerate(rows, 1) if row["video_id"] == video_id]

    def snapshot_csv(self, *, accept_external_changes: bool = False) -> list[str]:
        """Snapshot existing private CSVs without writing them or accepting their signoffs."""
        conflicts = []
        for name, fields in CSV_FIELDS.items():
            path = self.settings.private / name
            if not path.exists():
                continue
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
            if not reader.fieldnames or not set(fields).issubset(reader.fieldnames) or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError(f"Malformed CSV header: {name}")
            rows = list(reader)
            if any(None in row or any(value is None for value in row.values()) for row in rows):
                raise ValueError(f"Malformed CSV row: {name}")
            key = "gt_index" if name == "positive_adjudication.csv" else "video_id"
            if any(not row[key] for row in rows) or len({row[key] for row in rows}) != len(rows):
                raise ValueError(f"Duplicate/missing identity: {name}")
            with self.connection() as db:
                old = db.execute("SELECT sha256 FROM csv_snapshots WHERE filename=? ORDER BY id DESC LIMIT 1", (name,)).fetchone()
                if old and old[0] != digest and not accept_external_changes:
                    conflicts.append(name)
                elif not old or old[0] != digest:
                    db.execute("INSERT INTO csv_snapshots(filename,sha256,content,imported_at) VALUES(?,?,?,?)", (name, digest, content.decode("utf-8-sig"), utc_now()))
        return conflicts

    def own_csv_rows(self, token: str, name: str) -> list[dict[str, str]]:
        actor = self.actor(token, {"annotator_01", "annotator_02"})
        allowed = {f"negative_{actor['role']}.csv"}
        if actor["role"] == "annotator_02":
            allowed.add("positive_annotator_02.csv")
        if name not in allowed:
            raise PermissionError("Other reviewers' CSVs and historical adjudication are hidden")
        with self.connection() as db:
            row = db.execute("SELECT content FROM csv_snapshots WHERE filename=? ORDER BY id DESC LIMIT 1", (name,)).fetchone()
        return [item for item in csv.DictReader(io.StringIO(row[0])) if item.get("reviewer_id") == actor["id"]] if row else []

    def backup(self) -> Path:
        """Create a coherent SQLite backup including committed WAL state."""
        target = self.settings.private / "backups" / (utc_now().replace(":", "-") + "_" + uuid.uuid4().hex + ".sqlite3")
        target.parent.mkdir(exist_ok=True)
        with closing(sqlite3.connect(self.path)) as source, closing(sqlite3.connect(target)) as destination:
            source.backup(destination)
        return target
