"""Crash-safe SQLite store for Research Campaigns (§7.2).

Persistence for ``research_campaign.v1`` plus every controller-local record the
campaign needs: experiments, candidates, executions, events, errors, reviews,
decisions, repairs, idempotency records and generated reports.

The store lives under the Vibe runtime root (``~/.vibe-trading`` by default)
in its own SQLite file — it does not reuse the scheduled-job JSON store, because
a campaign is a state machine, not a prompt with a schedule (§7.2.1).

Design notes:

- SQLite WAL + ``synchronous=FULL`` makes every committed transaction crash
  safe (§5.3): a SIGKILL at any point leaves the previous committed state.
- :meth:`apply_bundle` inserts event rows, advances the per-experiment event
  cursor and applies all derived state in a single transaction, satisfying the
  "状态、队列写入和事件游标更新必须在同一本地事务提交" rule (§7.2.1).
- ``message_id`` is the dedupe key for at-least-once delivery (§4.3).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.config.paths import get_runtime_root

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_DEFAULT_DB_RELPATH = "research_controller/campaigns.db"


def _default_db_path() -> Path:
    return get_runtime_root() / _DEFAULT_DB_RELPATH


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def _loads(raw: str | None, default: Any) -> Any:
    if raw is None or raw == "":
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


class CampaignStore:
    """SQLite-backed store for a Vibe Research Controller.

    Args:
        db_path: Explicit SQLite path. Defaults to
            ``~/.vibe-trading/research_controller/campaigns.db``.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path: Path = db_path if db_path is not None else _default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._schema_version = _SCHEMA_VERSION
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _tx(self, conn: sqlite3.Connection, fn: Any, *args: Any) -> Any:
        try:
            result = fn(conn, *args)
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS campaigns (
                  campaign_id TEXT PRIMARY KEY,
                  research_goal_id TEXT NOT NULL,
                  status TEXT NOT NULL,
                  current_stage TEXT NOT NULL,
                  config_sha256 TEXT NOT NULL,
                  protocol_bundle_sha256 TEXT NOT NULL,
                  data_snapshot_id TEXT,
                  universe_snapshot_id TEXT,
                  gate_policy_version TEXT NOT NULL,
                  factor_inventory_json TEXT NOT NULL,
                  queue_counts_json TEXT NOT NULL,
                  budget_usage_json TEXT NOT NULL,
                  last_event_sequences_json TEXT NOT NULL,
                  config_json TEXT NOT NULL,
                  blocked_reason TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiments (
                  experiment_id TEXT PRIMARY KEY,
                  campaign_id TEXT NOT NULL,
                  round_id TEXT NOT NULL,
                  objective TEXT NOT NULL,
                  hypothesis_json TEXT NOT NULL,
                  status TEXT NOT NULL,
                  phase TEXT NOT NULL,
                  data_snapshot_id TEXT,
                  universe_snapshot_id TEXT,
                  gate_policy_version TEXT NOT NULL,
                  last_decision_action TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS candidates (
                  candidate_id TEXT NOT NULL,
                  candidate_version INTEGER NOT NULL,
                  campaign_id TEXT NOT NULL,
                  experiment_id TEXT NOT NULL,
                  parent_version INTEGER,
                  repair_of_error_id TEXT,
                  contract_version TEXT NOT NULL,
                  manifest_json TEXT NOT NULL,
                  source_sha256 TEXT NOT NULL,
                  status TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  PRIMARY KEY (candidate_id, candidate_version)
                );
                CREATE TABLE IF NOT EXISTS executions (
                  execution_id TEXT PRIMARY KEY,
                  campaign_id TEXT NOT NULL,
                  experiment_id TEXT NOT NULL,
                  candidate_id TEXT,
                  candidate_version INTEGER,
                  execution_type TEXT NOT NULL,
                  status TEXT NOT NULL,
                  progress REAL NOT NULL,
                  current_stage TEXT,
                  error_id TEXT,
                  evidence_id TEXT,
                  review_id TEXT,
                  created_at TEXT NOT NULL,
                  started_at TEXT,
                  finished_at TEXT
                );
                CREATE TABLE IF NOT EXISTS events (
                  message_id TEXT PRIMARY KEY,
                  campaign_id TEXT NOT NULL,
                  experiment_id TEXT NOT NULL,
                  sequence INTEGER NOT NULL,
                  producer TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  correlation_id TEXT,
                  causation_id TEXT,
                  payload_json TEXT NOT NULL,
                  artifact_refs_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  UNIQUE (experiment_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS errors (
                  error_id TEXT PRIMARY KEY,
                  campaign_id TEXT NOT NULL,
                  experiment_id TEXT NOT NULL,
                  execution_id TEXT NOT NULL,
                  attempt_id TEXT NOT NULL,
                  candidate_id TEXT,
                  candidate_version INTEGER,
                  stage TEXT NOT NULL,
                  category TEXT NOT NULL,
                  code TEXT NOT NULL,
                  retryable INTEGER NOT NULL,
                  repairable INTEGER NOT NULL,
                  message TEXT NOT NULL,
                  source_location_json TEXT,
                  observed_json TEXT,
                  allowed_actions_json TEXT,
                  forbidden_changes_json TEXT,
                  error_fingerprint TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reviews (
                  review_id TEXT PRIMARY KEY,
                  campaign_id TEXT NOT NULL,
                  experiment_id TEXT NOT NULL,
                  execution_id TEXT NOT NULL,
                  classification TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  root_causes_json TEXT NOT NULL,
                  gate_assessment_json TEXT,
                  repair_contract_json TEXT,
                  recommended_action TEXT NOT NULL,
                  recommended_tests_json TEXT,
                  limitations_json TEXT,
                  model TEXT NOT NULL,
                  prompt_version TEXT NOT NULL,
                  prompt_hash TEXT NOT NULL,
                  input_evidence_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evidence (
                  evidence_id TEXT PRIMARY KEY,
                  campaign_id TEXT NOT NULL,
                  experiment_id TEXT NOT NULL,
                  execution_id TEXT NOT NULL,
                  evidence_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decisions (
                  decision_id TEXT PRIMARY KEY,
                  campaign_id TEXT NOT NULL,
                  experiment_id TEXT NOT NULL,
                  round_id TEXT NOT NULL,
                  action TEXT NOT NULL,
                  rationale TEXT NOT NULL,
                  source_evidence_ids_json TEXT NOT NULL,
                  source_review_id TEXT,
                  next_candidate_id TEXT,
                  next_candidate_version INTEGER,
                  created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS repairs (
                  candidate_id TEXT NOT NULL,
                  candidate_version INTEGER NOT NULL,
                  campaign_id TEXT NOT NULL,
                  experiment_id TEXT NOT NULL,
                  parent_version INTEGER NOT NULL,
                  repair_of_error_id TEXT NOT NULL,
                  error_fingerprint TEXT NOT NULL,
                  change_summary TEXT NOT NULL,
                  preserved_constraints_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  PRIMARY KEY (candidate_id, candidate_version)
                );
                CREATE TABLE IF NOT EXISTS idempotency (
                  action TEXT NOT NULL,
                  idempotency_key TEXT NOT NULL,
                  request_sha256 TEXT NOT NULL,
                  response_status INTEGER NOT NULL,
                  response_json TEXT NOT NULL,
                  resource_id TEXT,
                  created_at TEXT NOT NULL,
                  PRIMARY KEY (action, idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS budget_usage (
                  usage_type TEXT NOT NULL,
                  used_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS campaign_reports (
                  campaign_id TEXT NOT NULL,
                  report_id TEXT NOT NULL,
                  report_md TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  PRIMARY KEY (campaign_id, report_id)
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_campaign ON experiments(campaign_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_candidates_campaign ON candidates(campaign_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_executions_campaign ON executions(campaign_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_experiment ON events(experiment_id, sequence)")
            conn.commit()

    # ------------------------------------------------------------------
    # Single-transaction bundle (§7.2.1)
    # ------------------------------------------------------------------

    def apply_bundle(self, bundle: dict[str, Any]) -> None:
        """Apply events + cursor + derived state atomically in one transaction.

        Accepted keys (all optional):

        - ``events``: list of ``(campaign_id, experiment_id, item_dict)`` —
          inserted with ``INSERT OR IGNORE``; already-seen ``message_id`` rows
          are skipped.
        - ``cursor``: ``(campaign_id, experiment_id, sequence)`` — advances the
          per-experiment event cursor to at least *sequence*.
        - ``errors`` / ``reviews`` / ``evidence`` / ``decisions`` / ``repairs``:
          lists of record dicts (upsert / insert-or-ignore).
        - ``executions`` / ``experiments`` / ``candidates`` / ``campaigns``:
          lists of ``(id_tuple, fields_dict)`` partial updates.
        """
        if not bundle:
            return

        def _run(conn: sqlite3.Connection) -> None:
            for campaign_id, experiment_id, item in bundle.get("events", []):
                self._insert_event_row(
                    conn,
                    {
                        "campaign_id": campaign_id,
                        "experiment_id": experiment_id,
                        "message_id": item["message_id"],
                        "sequence": item.get("sequence", 0),
                        "producer": item.get("producer", ""),
                        "event_type": item.get("event_type", ""),
                        "correlation_id": item.get("correlation_id"),
                        "causation_id": item.get("causation_id"),
                        "payload": item.get("payload") or {},
                        "artifact_refs": item.get("artifact_refs") or [],
                        "created_at": item.get("created_at", _now()),
                    },
                )
            if bundle.get("cursor"):
                campaign_id, experiment_id, sequence = bundle["cursor"]
                seqs_row = conn.execute(
                    "SELECT last_event_sequences_json FROM campaigns WHERE campaign_id=?", (campaign_id,)
                ).fetchone()
                seqs = _loads(seqs_row["last_event_sequences_json"], {}) if seqs_row else {}
                seqs[experiment_id] = max(seqs.get(experiment_id, 0), int(sequence))
                conn.execute(
                    "UPDATE campaigns SET last_event_sequences_json=?, updated_at=? WHERE campaign_id=?",
                    (_dumps(seqs), _now(), campaign_id),
                )
            for record in bundle.get("errors", []):
                self._upsert_error_row(conn, record)
            for record in bundle.get("reviews", []):
                self._upsert_review_row(conn, record)
            for record in bundle.get("evidence", []):
                self._upsert_evidence_row(conn, record)
            for record in bundle.get("decisions", []):
                try:
                    self._insert_decision_row(conn, record)
                except sqlite3.IntegrityError:
                    pass  # immutable: already recorded
            for record in bundle.get("repairs", []):
                try:
                    self._insert_repair_row(conn, record)
                except sqlite3.IntegrityError:
                    pass
            for execution_id, fields in bundle.get("executions", []):
                self._update_execution_row(conn, execution_id, fields)
            for experiment_id, fields in bundle.get("experiments", []):
                self._update_experiment_row(conn, experiment_id, fields)
            for candidate_id, candidate_version, fields in bundle.get("candidates", []):
                self._update_candidate_row(conn, candidate_id, candidate_version, fields)
            for campaign_id, fields in bundle.get("campaigns", []):
                self._update_campaign_row(conn, campaign_id, fields)

        self._tx(self._connect(), _run)

    # ------------------------------------------------------------------
    # Row writers (accept an open connection)
    # ------------------------------------------------------------------

    @staticmethod
    def _insert_event_row(conn: sqlite3.Connection, record: dict[str, Any]) -> None:
        conn.execute(
            """INSERT OR IGNORE INTO events (message_id, campaign_id, experiment_id, sequence,
               producer, event_type, correlation_id, causation_id, payload_json,
               artifact_refs_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record["message_id"],
                record["campaign_id"],
                record["experiment_id"],
                record["sequence"],
                record["producer"],
                record["event_type"],
                record.get("correlation_id"),
                record.get("causation_id"),
                _dumps(record.get("payload", {})),
                _dumps(record.get("artifact_refs", [])),
                record["created_at"],
            ),
        )

    @staticmethod
    def _upsert_error_row(conn: sqlite3.Connection, record: dict[str, Any]) -> None:
        conn.execute(
            """INSERT INTO errors (error_id, campaign_id, experiment_id, execution_id,
               attempt_id, candidate_id, candidate_version, stage, category, code, retryable,
               repairable, message, source_location_json, observed_json, allowed_actions_json,
               forbidden_changes_json, error_fingerprint, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(error_id) DO UPDATE SET message=excluded.message""",
            (
                record["error_id"],
                record["campaign_id"],
                record["experiment_id"],
                record["execution_id"],
                record["attempt_id"],
                record.get("candidate_id"),
                record.get("candidate_version"),
                record["stage"],
                record["category"],
                record["code"],
                1 if record.get("retryable") else 0,
                1 if record.get("repairable") else 0,
                record["message"],
                _dumps(record.get("source_location")),
                _dumps(record.get("observed")),
                _dumps(record.get("allowed_actions", [])),
                _dumps(record.get("forbidden_changes", [])),
                record["error_fingerprint"],
                record["created_at"],
            ),
        )

    @staticmethod
    def _upsert_review_row(conn: sqlite3.Connection, record: dict[str, Any]) -> None:
        conn.execute(
            """INSERT INTO reviews (review_id, campaign_id, experiment_id, execution_id,
               classification, confidence, root_causes_json, gate_assessment_json,
               repair_contract_json, recommended_action, recommended_tests_json,
               limitations_json, model, prompt_version, prompt_hash, input_evidence_hash,
               created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(review_id) DO UPDATE SET classification=excluded.classification""",
            (
                record["review_id"],
                record["campaign_id"],
                record["experiment_id"],
                record["execution_id"],
                record["classification"],
                record["confidence"],
                _dumps(record.get("root_causes", [])),
                _dumps(record.get("gate_assessment")),
                _dumps(record.get("repair_contract")),
                record["recommended_action"],
                _dumps(record.get("recommended_tests", [])),
                _dumps(record.get("limitations", [])),
                record["model"],
                record["prompt_version"],
                record["prompt_hash"],
                record["input_evidence_hash"],
                record["created_at"],
            ),
        )

    @staticmethod
    def _upsert_evidence_row(conn: sqlite3.Connection, record: dict[str, Any]) -> None:
        conn.execute(
            """INSERT OR IGNORE INTO evidence (evidence_id, campaign_id, experiment_id,
               execution_id, evidence_json, created_at)
               VALUES (?,?,?,?,?,?)""",
            (
                record["evidence_id"],
                record["campaign_id"],
                record["experiment_id"],
                record["execution_id"],
                _dumps(record.get("evidence")),
                record["created_at"],
            ),
        )

    @staticmethod
    def _insert_decision_row(conn: sqlite3.Connection, record: dict[str, Any]) -> None:
        conn.execute(
            """INSERT INTO decisions (decision_id, campaign_id, experiment_id, round_id, action,
               rationale, source_evidence_ids_json, source_review_id, next_candidate_id,
               next_candidate_version, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record["decision_id"],
                record["campaign_id"],
                record["experiment_id"],
                record["round_id"],
                record["action"],
                record["rationale"],
                _dumps(record.get("source_evidence_ids", [])),
                record.get("source_review_id"),
                record.get("next_candidate_id"),
                record.get("next_candidate_version"),
                record["created_at"],
            ),
        )

    @staticmethod
    def _insert_repair_row(conn: sqlite3.Connection, record: dict[str, Any]) -> None:
        conn.execute(
            """INSERT INTO repairs (candidate_id, candidate_version, campaign_id, experiment_id,
               parent_version, repair_of_error_id, error_fingerprint, change_summary,
               preserved_constraints_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                record["candidate_id"],
                record["candidate_version"],
                record["campaign_id"],
                record["experiment_id"],
                record["parent_version"],
                record["repair_of_error_id"],
                record["error_fingerprint"],
                record["change_summary"],
                _dumps(record.get("preserved_constraints", [])),
                record["created_at"],
            ),
        )

    @staticmethod
    def _update_campaign_row(conn: sqlite3.Connection, campaign_id: str, fields: dict[str, Any]) -> None:
        allowed = {
            "status",
            "current_stage",
            "data_snapshot_id",
            "universe_snapshot_id",
            "gate_policy_version",
            "factor_inventory_json",
            "queue_counts_json",
            "budget_usage_json",
            "last_event_sequences_json",
            "blocked_reason",
            "updated_at",
        }
        sets, values = [], []
        for key, value in fields.items():
            if key not in allowed:
                raise ValueError(f"unknown campaign field: {key}")
            sets.append(f"{key}=?")
            values.append(value)
        values.append(campaign_id)
        conn.execute(f"UPDATE campaigns SET {', '.join(sets)} WHERE campaign_id=?", values)

    @staticmethod
    def _update_experiment_row(conn: sqlite3.Connection, experiment_id: str, fields: dict[str, Any]) -> None:
        allowed = {"status", "phase", "last_decision_action", "updated_at"}
        sets, values = [], []
        for key, value in fields.items():
            if key not in allowed:
                raise ValueError(f"unknown experiment field: {key}")
            sets.append(f"{key}=?")
            values.append(value)
        values.append(experiment_id)
        conn.execute(f"UPDATE experiments SET {', '.join(sets)} WHERE experiment_id=?", values)

    @staticmethod
    def _update_candidate_row(
        conn: sqlite3.Connection, candidate_id: str, candidate_version: int, fields: dict[str, Any]
    ) -> None:
        allowed = {"status"}
        sets, values = [], []
        for key, value in fields.items():
            if key not in allowed:
                raise ValueError(f"unknown candidate field: {key}")
            sets.append(f"{key}=?")
            values.append(value)
        values.extend([candidate_id, candidate_version])
        conn.execute(f"UPDATE candidates SET {', '.join(sets)} WHERE candidate_id=? AND candidate_version=?", values)

    @staticmethod
    def _update_execution_row(conn: sqlite3.Connection, execution_id: str, fields: dict[str, Any]) -> None:
        allowed = {
            "status",
            "progress",
            "current_stage",
            "error_id",
            "evidence_id",
            "review_id",
            "started_at",
            "finished_at",
        }
        sets, values = [], []
        for key, value in fields.items():
            if key not in allowed:
                raise ValueError(f"unknown execution field: {key}")
            sets.append(f"{key}=?")
            values.append(value)
        values.append(execution_id)
        conn.execute(f"UPDATE executions SET {', '.join(sets)} WHERE execution_id=?", values)

    @staticmethod
    def _insert_campaign_row(conn: sqlite3.Connection, record: dict[str, Any]) -> None:
        conn.execute(
            """INSERT INTO campaigns (campaign_id, research_goal_id, status, current_stage,
               config_sha256, protocol_bundle_sha256, data_snapshot_id, universe_snapshot_id,
               gate_policy_version, factor_inventory_json, queue_counts_json, budget_usage_json,
               last_event_sequences_json, config_json, blocked_reason, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record["campaign_id"],
                record["research_goal_id"],
                record["status"],
                record["current_stage"],
                record["config_sha256"],
                record["protocol_bundle_sha256"],
                record.get("data_snapshot_id"),
                record.get("universe_snapshot_id"),
                record["gate_policy_version"],
                _dumps(record.get("factor_inventory", {})),
                _dumps(record.get("queue_counts", {})),
                _dumps(record.get("budget_usage", {})),
                _dumps(record.get("last_event_sequences", {})),
                _dumps(record.get("config", {})),
                record.get("blocked_reason"),
                record["created_at"],
                record["updated_at"],
            ),
        )

    @staticmethod
    def _insert_experiment_row(conn: sqlite3.Connection, record: dict[str, Any]) -> None:
        conn.execute(
            """INSERT INTO experiments (experiment_id, campaign_id, round_id, objective,
               hypothesis_json, status, phase, data_snapshot_id, universe_snapshot_id,
               gate_policy_version, last_decision_action, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(experiment_id) DO UPDATE SET
                 status=excluded.status, phase=excluded.phase,
                 data_snapshot_id=excluded.data_snapshot_id,
                 universe_snapshot_id=excluded.universe_snapshot_id,
                 gate_policy_version=excluded.gate_policy_version,
                 last_decision_action=excluded.last_decision_action,
                 updated_at=excluded.updated_at""",
            (
                record["experiment_id"],
                record["campaign_id"],
                record["round_id"],
                record["objective"],
                _dumps(record.get("hypothesis", {})),
                record["status"],
                record["phase"],
                record.get("data_snapshot_id"),
                record.get("universe_snapshot_id"),
                record["gate_policy_version"],
                record.get("last_decision_action"),
                record["created_at"],
                record["updated_at"],
            ),
        )

    @staticmethod
    def _insert_candidate_row(conn: sqlite3.Connection, record: dict[str, Any]) -> None:
        conn.execute(
            """INSERT INTO candidates (candidate_id, candidate_version, campaign_id,
               experiment_id, parent_version, repair_of_error_id, contract_version,
               manifest_json, source_sha256, status, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(candidate_id, candidate_version) DO UPDATE SET status=excluded.status""",
            (
                record["candidate_id"],
                record["candidate_version"],
                record["campaign_id"],
                record["experiment_id"],
                record.get("parent_version"),
                record.get("repair_of_error_id"),
                record["contract_version"],
                _dumps(record.get("manifest", {})),
                record["source_sha256"],
                record["status"],
                record["created_at"],
            ),
        )

    @staticmethod
    def _insert_execution_row(conn: sqlite3.Connection, record: dict[str, Any]) -> None:
        conn.execute(
            """INSERT INTO executions (execution_id, campaign_id, experiment_id, candidate_id,
               candidate_version, execution_type, status, progress, current_stage, error_id,
               evidence_id, review_id, created_at, started_at, finished_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(execution_id) DO UPDATE SET
                 status=excluded.status, progress=excluded.progress,
                 current_stage=excluded.current_stage, error_id=excluded.error_id,
                 evidence_id=excluded.evidence_id, review_id=excluded.review_id,
                 started_at=excluded.started_at, finished_at=excluded.finished_at""",
            (
                record["execution_id"],
                record["campaign_id"],
                record["experiment_id"],
                record.get("candidate_id"),
                record.get("candidate_version"),
                record["execution_type"],
                record["status"],
                record.get("progress", 0.0),
                record.get("current_stage"),
                record.get("error_id"),
                record.get("evidence_id"),
                record.get("review_id"),
                record["created_at"],
                record.get("started_at"),
                record.get("finished_at"),
            ),
        )

    # ------------------------------------------------------------------
    # Campaigns
    # ------------------------------------------------------------------

    def create_campaign(self, record: dict[str, Any]) -> None:
        self._tx(self._connect(), self._insert_campaign_row, record)

    def update_campaign(self, campaign_id: str, **fields: Any) -> None:
        if fields:
            self._tx(self._connect(), self._update_campaign_row, campaign_id, fields)

    def set_campaign_cursor(self, campaign_id: str, experiment_id: str, sequence: int) -> None:
        def _run(conn: sqlite3.Connection) -> None:
            row = conn.execute(
                "SELECT last_event_sequences_json FROM campaigns WHERE campaign_id=?", (campaign_id,)
            ).fetchone()
            seqs = _loads(row["last_event_sequences_json"], {}) if row else {}
            seqs[experiment_id] = int(sequence)
            conn.execute(
                "UPDATE campaigns SET last_event_sequences_json=?, updated_at=? WHERE campaign_id=?",
                (_dumps(seqs), _now(), campaign_id),
            )

        self._tx(self._connect(), _run)

    def get_campaign(self, campaign_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM campaigns WHERE campaign_id=?", (campaign_id,)).fetchone()
        return self._campaign_row(row) if row else None

    def list_campaigns(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        query = "SELECT * FROM campaigns"
        params: list[Any] = []
        if status is not None:
            query += " WHERE status=?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._campaign_row(r) for r in rows]

    @staticmethod
    def _campaign_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "campaign_id": row["campaign_id"],
            "research_goal_id": row["research_goal_id"],
            "status": row["status"],
            "current_stage": row["current_stage"],
            "config_sha256": row["config_sha256"],
            "protocol_bundle_sha256": row["protocol_bundle_sha256"],
            "data_snapshot_id": row["data_snapshot_id"],
            "universe_snapshot_id": row["universe_snapshot_id"],
            "gate_policy_version": row["gate_policy_version"],
            "factor_inventory": _loads(row["factor_inventory_json"], {}),
            "queue_counts": _loads(row["queue_counts_json"], {}),
            "budget_usage": _loads(row["budget_usage_json"], {}),
            "last_event_sequences": _loads(row["last_event_sequences_json"], {}),
            "config": _loads(row["config_json"], {}),
            "blocked_reason": row["blocked_reason"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    def upsert_experiment(self, record: dict[str, Any]) -> None:
        self._tx(self._connect(), self._insert_experiment_row, record)

    def update_experiment(self, experiment_id: str, **fields: Any) -> None:
        if fields:
            self._tx(self._connect(), self._update_experiment_row, experiment_id, fields)

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
        return self._experiment_row(row) if row else None

    def list_experiments(self, campaign_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM experiments WHERE campaign_id=? ORDER BY created_at ASC", (campaign_id,)
            ).fetchall()
        return [self._experiment_row(r) for r in rows]

    @staticmethod
    def _experiment_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "experiment_id": row["experiment_id"],
            "campaign_id": row["campaign_id"],
            "round_id": row["round_id"],
            "objective": row["objective"],
            "hypothesis": _loads(row["hypothesis_json"], {}),
            "status": row["status"],
            "phase": row["phase"],
            "data_snapshot_id": row["data_snapshot_id"],
            "universe_snapshot_id": row["universe_snapshot_id"],
            "gate_policy_version": row["gate_policy_version"],
            "last_decision_action": row["last_decision_action"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # ------------------------------------------------------------------
    # Candidates
    # ------------------------------------------------------------------

    def upsert_candidate(self, record: dict[str, Any]) -> None:
        self._tx(self._connect(), self._insert_candidate_row, record)

    def update_candidate(self, candidate_id: str, candidate_version: int, **fields: Any) -> None:
        if fields:
            self._tx(self._connect(), self._update_candidate_row, candidate_id, candidate_version, fields)

    def list_candidates(self, campaign_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM candidates WHERE campaign_id=? ORDER BY created_at ASC, candidate_version ASC",
                (campaign_id,),
            ).fetchall()
        return [self._candidate_row(r) for r in rows]

    def list_candidates_for_experiment(self, campaign_id: str, experiment_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM candidates WHERE campaign_id=? AND experiment_id=?
                   ORDER BY candidate_version ASC""",
                (campaign_id, experiment_id),
            ).fetchall()
        return [self._candidate_row(r) for r in rows]

    @staticmethod
    def _candidate_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "candidate_id": row["candidate_id"],
            "candidate_version": row["candidate_version"],
            "campaign_id": row["campaign_id"],
            "experiment_id": row["experiment_id"],
            "parent_version": row["parent_version"],
            "repair_of_error_id": row["repair_of_error_id"],
            "contract_version": row["contract_version"],
            "manifest": _loads(row["manifest_json"], {}),
            "source_sha256": row["source_sha256"],
            "status": row["status"],
            "created_at": row["created_at"],
        }

    # ------------------------------------------------------------------
    # Executions
    # ------------------------------------------------------------------

    def upsert_execution(self, record: dict[str, Any]) -> None:
        self._tx(self._connect(), self._insert_execution_row, record)

    def update_execution(self, execution_id: str, **fields: Any) -> None:
        if fields:
            self._tx(self._connect(), self._update_execution_row, execution_id, fields)

    def list_executions(self, campaign_id: str, experiment_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM executions WHERE campaign_id=?"
        params: list[Any] = [campaign_id]
        if experiment_id is not None:
            query += " AND experiment_id=?"
            params.append(experiment_id)
        query += " ORDER BY created_at ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._execution_row(r) for r in rows]

    def get_execution(self, execution_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM executions WHERE execution_id=?", (execution_id,)).fetchone()
        return self._execution_row(row) if row else None

    @staticmethod
    def _execution_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "execution_id": row["execution_id"],
            "campaign_id": row["campaign_id"],
            "experiment_id": row["experiment_id"],
            "candidate_id": row["candidate_id"],
            "candidate_version": row["candidate_version"],
            "execution_type": row["execution_type"],
            "status": row["status"],
            "progress": row["progress"],
            "current_stage": row["current_stage"],
            "error_id": row["error_id"],
            "evidence_id": row["evidence_id"],
            "review_id": row["review_id"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
        }

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def event_exists(self, message_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM events WHERE message_id=?", (message_id,)).fetchone()
        return row is not None

    def insert_event(self, record: dict[str, Any]) -> bool:
        try:
            self._tx(self._connect(), self._insert_event_row, record)
        except sqlite3.IntegrityError:
            return False
        return True

    def list_events(self, campaign_id: str, experiment_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM events WHERE campaign_id=?"
        params: list[Any] = [campaign_id]
        if experiment_id is not None:
            query += " AND experiment_id=?"
            params.append(experiment_id)
        query += " ORDER BY sequence ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "message_id": r["message_id"],
                "experiment_id": r["experiment_id"],
                "sequence": r["sequence"],
                "producer": r["producer"],
                "event_type": r["event_type"],
                "correlation_id": r["correlation_id"],
                "causation_id": r["causation_id"],
                "payload": _loads(r["payload_json"], {}),
                "artifact_refs": _loads(r["artifact_refs_json"], []),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Errors / reviews / decisions / repairs
    # ------------------------------------------------------------------

    def upsert_error(self, record: dict[str, Any]) -> None:
        self._tx(self._connect(), self._upsert_error_row, record)

    def get_error(self, error_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM errors WHERE error_id=?", (error_id,)).fetchone()
        if row is None:
            return None
        return {
            "error_id": row["error_id"],
            "campaign_id": row["campaign_id"],
            "experiment_id": row["experiment_id"],
            "execution_id": row["execution_id"],
            "attempt_id": row["attempt_id"],
            "candidate_id": row["candidate_id"],
            "candidate_version": row["candidate_version"],
            "stage": row["stage"],
            "category": row["category"],
            "code": row["code"],
            "retryable": bool(row["retryable"]),
            "repairable": bool(row["repairable"]),
            "message": row["message"],
            "source_location": _loads(row["source_location_json"], None),
            "observed": _loads(row["observed_json"], None),
            "allowed_actions": _loads(row["allowed_actions_json"], []),
            "forbidden_changes": _loads(row["forbidden_changes_json"], []),
            "error_fingerprint": row["error_fingerprint"],
            "created_at": row["created_at"],
        }

    def list_errors(self, campaign_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM errors WHERE campaign_id=? ORDER BY created_at ASC", (campaign_id,)
            ).fetchall()
        return [
            {
                "error_id": r["error_id"],
                "experiment_id": r["experiment_id"],
                "execution_id": r["execution_id"],
                "candidate_id": r["candidate_id"],
                "candidate_version": r["candidate_version"],
                "stage": r["stage"],
                "category": r["category"],
                "code": r["code"],
                "retryable": bool(r["retryable"]),
                "repairable": bool(r["repairable"]),
                "message": r["message"],
                "error_fingerprint": r["error_fingerprint"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def upsert_review(self, record: dict[str, Any]) -> None:
        self._tx(self._connect(), self._upsert_review_row, record)

    def get_review(self, review_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM reviews WHERE review_id=?", (review_id,)).fetchone()
        if row is None:
            return None
        return {
            "review_id": row["review_id"],
            "experiment_id": row["experiment_id"],
            "execution_id": row["execution_id"],
            "classification": row["classification"],
            "confidence": row["confidence"],
            "root_causes": _loads(row["root_causes_json"], []),
            "gate_assessment": _loads(row["gate_assessment_json"], None),
            "repair_contract": _loads(row["repair_contract_json"], None),
            "recommended_action": row["recommended_action"],
            "recommended_tests": _loads(row["recommended_tests_json"], []),
            "limitations": _loads(row["limitations_json"], []),
            "model": row["model"],
            "prompt_version": row["prompt_version"],
            "prompt_hash": row["prompt_hash"],
            "input_evidence_hash": row["input_evidence_hash"],
            "created_at": row["created_at"],
        }

    def list_reviews(self, campaign_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM reviews WHERE campaign_id=? ORDER BY created_at ASC", (campaign_id,)
            ).fetchall()
        return [
            {
                "review_id": r["review_id"],
                "experiment_id": r["experiment_id"],
                "execution_id": r["execution_id"],
                "classification": r["classification"],
                "confidence": r["confidence"],
                "recommended_action": r["recommended_action"],
                "repair_contract": _loads(r["repair_contract_json"], None),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def upsert_evidence(self, record: dict[str, Any]) -> None:
        self._tx(self._connect(), self._upsert_evidence_row, record)

    def get_evidence(self, evidence_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM evidence WHERE evidence_id=?", (evidence_id,)).fetchone()
        if row is None:
            return None
        return {
            "evidence_id": row["evidence_id"],
            "experiment_id": row["experiment_id"],
            "execution_id": row["execution_id"],
            "evidence": _loads(row["evidence_json"], {}),
            "created_at": row["created_at"],
        }

    def list_evidence(self, campaign_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM evidence WHERE campaign_id=? ORDER BY created_at ASC", (campaign_id,)
            ).fetchall()
        return [
            {
                "evidence_id": r["evidence_id"],
                "experiment_id": r["experiment_id"],
                "execution_id": r["execution_id"],
                "evidence": _loads(r["evidence_json"], {}),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def insert_decision(self, record: dict[str, Any]) -> bool:
        try:
            self._tx(self._connect(), self._insert_decision_row, record)
        except sqlite3.IntegrityError:
            return False
        return True

    def list_decisions(self, campaign_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM decisions WHERE campaign_id=? ORDER BY created_at ASC", (campaign_id,)
            ).fetchall()
        return [
            {
                "decision_id": r["decision_id"],
                "experiment_id": r["experiment_id"],
                "round_id": r["round_id"],
                "action": r["action"],
                "rationale": r["rationale"],
                "source_evidence_ids": _loads(r["source_evidence_ids_json"], []),
                "source_review_id": r["source_review_id"],
                "next_candidate_id": r["next_candidate_id"],
                "next_candidate_version": r["next_candidate_version"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def insert_repair(self, record: dict[str, Any]) -> bool:
        try:
            self._tx(self._connect(), self._insert_repair_row, record)
        except sqlite3.IntegrityError:
            return False
        return True

    def list_repairs(self, campaign_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM repairs WHERE campaign_id=? ORDER BY created_at ASC", (campaign_id,)
            ).fetchall()
        return [
            {
                "candidate_id": r["candidate_id"],
                "candidate_version": r["candidate_version"],
                "experiment_id": r["experiment_id"],
                "parent_version": r["parent_version"],
                "repair_of_error_id": r["repair_of_error_id"],
                "error_fingerprint": r["error_fingerprint"],
                "change_summary": r["change_summary"],
                "preserved_constraints": _loads(r["preserved_constraints_json"], []),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Idempotency records (§4.3 / §5.11)
    # ------------------------------------------------------------------

    def idempotency_lookup(self, action: str, idempotency_key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM idempotency WHERE action=? AND idempotency_key=?",
                (action, idempotency_key),
            ).fetchone()
        if row is None:
            return None
        return {
            "action": row["action"],
            "idempotency_key": row["idempotency_key"],
            "request_sha256": row["request_sha256"],
            "response_status": row["response_status"],
            "response_json": row["response_json"],
            "resource_id": row["resource_id"],
            "created_at": row["created_at"],
        }

    def save_idempotency(
        self,
        action: str,
        idempotency_key: str,
        request_sha256: str,
        response_status: int,
        response_json: str,
        resource_id: str | None,
    ) -> None:
        def _run(conn: sqlite3.Connection) -> None:
            conn.execute(
                """INSERT INTO idempotency (action, idempotency_key, request_sha256,
                   response_status, response_json, resource_id, created_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(action, idempotency_key) DO UPDATE SET
                     response_status=excluded.response_status,
                     response_json=excluded.response_json,
                     resource_id=excluded.resource_id""",
                (action, idempotency_key, request_sha256, response_status, response_json, resource_id, _now()),
            )

        self._tx(self._connect(), _run)

    # ------------------------------------------------------------------
    # Rolling budget usage (§13.3)
    # ------------------------------------------------------------------

    def record_budget_usage(self, usage_type: str, used_at: str | None = None) -> None:
        def _run(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO budget_usage (usage_type, used_at) VALUES (?,?)",
                (usage_type, used_at or _now()),
            )

        self._tx(self._connect(), _run)

    def rolling_budget_usage(self, usage_type: str, window_seconds: int = 24 * 3600) -> int:
        """Return usage of *usage_type* within the trailing rolling 24h window.

        Budget usage rows store ISO-8601 UTC timestamps in the same format, so a
        lexicographic comparison against the window boundary is a valid ordering.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=window_seconds)).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                """SELECT COUNT(*) AS c FROM budget_usage
                   WHERE usage_type=? AND used_at >= ?""",
                (usage_type, cutoff),
            ).fetchone()
            return int(row["c"])

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def save_report(self, campaign_id: str, report_id: str, report_md: str) -> None:
        def _run(conn: sqlite3.Connection) -> None:
            conn.execute(
                """INSERT INTO campaign_reports (campaign_id, report_id, report_md, created_at)
                   VALUES (?,?,?,?)""",
                (campaign_id, report_id, report_md, _now()),
            )

        self._tx(self._connect(), _run)

    def get_latest_report(self, campaign_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM campaign_reports WHERE campaign_id=?
                   ORDER BY created_at DESC LIMIT 1""",
                (campaign_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "campaign_id": row["campaign_id"],
            "report_id": row["report_id"],
            "report_md": row["report_md"],
            "created_at": row["created_at"],
        }
