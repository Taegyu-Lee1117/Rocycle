"""PostgreSQL access used by the Rocycle web API."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row

from .config import FINAL_DIR, Settings
from .statistics import CLASS_NAMES, period_bucket_index, period_definition


DESTINATIONS = (
    "plastic_bin",
    "can_bin",
    "paper_bin",
    "battery_bin",
    "human_handoff",
)

REVIEW_FINAL_DESTINATIONS = {
    "plastic_bag": "plastic_bin",
}


class Database:
    def __init__(self, settings: Settings):
        self.settings = settings

    @contextmanager
    def connect(self) -> Iterator[psycopg.Connection]:
        with psycopg.connect(
            host=self.settings.db_host,
            port=self.settings.db_port,
            dbname=self.settings.db_name,
            user=self.settings.db_user,
            password=self.settings.db_password,
            row_factory=dict_row,
        ) as connection:
            yield connection

    def initialize_schema(self) -> None:
        schema_path = FINAL_DIR / "DB" / "rocycle_project_schema.sql"
        schema = schema_path.read_text(encoding="utf-8")
        with self.connect() as connection:
            connection.execute(schema, prepare=False)

    def ping(self) -> bool:
        with self.connect() as connection:
            row = connection.execute("SELECT 1 AS ok").fetchone()
        return bool(row and row["ok"] == 1)

    def list_reviews(self) -> list[dict[str, Any]]:
        query = """
            SELECT
                pl.id,
                wt.class_name AS predicted_class,
                pl.confidence,
                pl.review_reason,
                pl.destination,
                pl.created_at
            FROM processing_log AS pl
            LEFT JOIN waste_type AS wt ON wt.id = pl.waste_type_id
            WHERE pl.status = 'review_pending'
            ORDER BY pl.created_at ASC, pl.id ASC
        """
        with self.connect() as connection:
            return list(connection.execute(query).fetchall())

    def complete_review(
        self, processing_log_id: int, final_class: str, reviewed_by: str | None
    ) -> dict[str, Any] | None:
        if final_class not in CLASS_NAMES:
            raise ValueError(f"지원하지 않는 클래스입니다: {final_class}")
        with self.connect() as connection:
            waste_type = connection.execute(
                "SELECT id, class_name, destination FROM waste_type WHERE class_name = %s",
                (final_class,),
            ).fetchone()
            if waste_type is None:
                raise ValueError(f"DB에 클래스가 없습니다: {final_class}")
            final_destination = REVIEW_FINAL_DESTINATIONS.get(
                final_class, waste_type["destination"]
            )

            updated = connection.execute(
                """
                UPDATE processing_log
                SET final_waste_type_id = %s,
                    final_destination = %s,
                    status = 'completed',
                    result = 'success',
                    reviewed_at = CURRENT_TIMESTAMP,
                    completed_at = CURRENT_TIMESTAMP,
                    reviewed_by = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND status = 'review_pending'
                RETURNING id, status, final_destination, reviewed_at, completed_at
                """,
                (
                    waste_type["id"],
                    final_destination,
                    reviewed_by or "operator",
                    processing_log_id,
                ),
            ).fetchone()
            if updated is None:
                return None

            connection.execute(
                """
                INSERT INTO robot_event (processing_log_id, event_type, detail)
                VALUES (%s, 'review_completed', %s)
                """,
                (
                    processing_log_id,
                    f"final_class={final_class}; reviewed_by={reviewed_by or 'operator'}",
                ),
            )
            return {
                **updated,
                "final_class": waste_type["class_name"],
            }

    def create_processing_log(self, payload: dict[str, Any]) -> dict[str, Any]:
        predicted_class = payload["predicted_class"]
        with self.connect() as connection:
            waste_type = connection.execute(
                "SELECT id, destination FROM waste_type WHERE class_name = %s",
                (predicted_class,),
            ).fetchone()
            if waste_type is None:
                raise ValueError(f"DB에 클래스가 없습니다: {predicted_class}")

            destination = payload.get("destination") or waste_type["destination"]
            status = payload.get("status") or (
                "review_pending" if destination == "review_bin" else "completed"
            )
            query = """
                INSERT INTO processing_log (
                    external_id, waste_type_id, confidence, action, destination, result,
                    status, review_reason, weight, completed_at,
                    final_waste_type_id, final_destination, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    CASE WHEN %s = 'completed' THEN CURRENT_TIMESTAMP ELSE NULL END,
                    CASE WHEN %s = 'completed' THEN %s ELSE NULL END,
                    CASE WHEN %s = 'completed' THEN %s ELSE NULL END,
                    CURRENT_TIMESTAMP
                )
                ON CONFLICT (external_id)
                DO UPDATE SET external_id = EXCLUDED.external_id
                RETURNING id, status, destination, created_at, completed_at
            """
            row = connection.execute(
                query,
                (
                    payload.get("external_id"),
                    waste_type["id"],
                    payload.get("confidence"),
                    payload.get("action"),
                    destination,
                    payload.get("result", "completed" if status == "completed" else None),
                    status,
                    payload.get("review_reason"),
                    payload.get("net_weight_kg"),
                    status,
                    status,
                    waste_type["id"],
                    status,
                    waste_type["destination"],
                ),
            ).fetchone()
            return dict(row)

    def dashboard(self) -> dict[str, Any]:
        counts = {destination: 0 for destination in DESTINATIONS}
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    COALESCE(pl.final_destination, fwt.destination, wt.destination) AS destination,
                    COUNT(*)::integer AS count
                FROM processing_log AS pl
                LEFT JOIN waste_type AS wt ON wt.id = pl.waste_type_id
                LEFT JOIN waste_type AS fwt ON fwt.id = pl.final_waste_type_id
                WHERE pl.status = 'completed'
                GROUP BY 1
                """
            ).fetchall()
            for row in rows:
                if row["destination"]:
                    counts[row["destination"]] = row["count"]

            pending = connection.execute(
                "SELECT COUNT(*)::integer AS count FROM processing_log WHERE status = 'review_pending'"
            ).fetchone()["count"]
            counts["review_bin"] = pending

            last = connection.execute(
                """
                SELECT
                    COALESCE(fwt.class_name, wt.class_name) AS item,
                    COALESCE(pl.final_destination, fwt.destination, pl.destination) AS dest,
                    pl.weight,
                    pl.review_reason AS reason,
                    pl.completed_at AS ts
                FROM processing_log AS pl
                LEFT JOIN waste_type AS wt ON wt.id = pl.waste_type_id
                LEFT JOIN waste_type AS fwt ON fwt.id = pl.final_waste_type_id
                WHERE pl.status = 'completed'
                ORDER BY pl.completed_at DESC NULLS LAST, pl.id DESC
                LIMIT 1
                """
            ).fetchone()

        last_payload = None
        if last:
            last_payload = {
                "item": last["item"],
                "dest": last["dest"],
                "weight_g": (
                    round(float(last["weight"]) * 1000, 1)
                    if last["weight"] is not None
                    else None
                ),
                "reason": last["reason"],
                "ts": last["ts"],
            }
        return {
            "counts": counts,
            "total": sum(counts[key] for key in DESTINATIONS),
            "last": last_payload,
            "review_pending": pending,
        }

    def statistics(self, period: str, now: datetime | None = None) -> dict[str, Any]:
        now = now or datetime.now()
        start, range_text, title, labels = period_definition(period, now)
        with self.connect() as connection:
            completed_rows = list(
                connection.execute(
                    """
                    SELECT
                        pl.completed_at,
                        COALESCE(fwt.class_name, wt.class_name) AS class_name
                    FROM processing_log AS pl
                    LEFT JOIN waste_type AS wt ON wt.id = pl.waste_type_id
                    LEFT JOIN waste_type AS fwt ON fwt.id = pl.final_waste_type_id
                    WHERE pl.status = 'completed' AND pl.completed_at >= %s
                    ORDER BY pl.completed_at
                    """,
                    (start,),
                ).fetchall()
            )
            status_counts = connection.execute(
                """
                SELECT
                    COUNT(*) FILTER (
                        WHERE status = 'review_pending' AND created_at >= %s
                    )::integer AS pending,
                    COUNT(*) FILTER (
                        WHERE status = 'failed' AND created_at >= %s
                    )::integer AS failed,
                    COUNT(*) FILTER (
                        WHERE status = 'completed' AND reviewed_at >= %s
                    )::integer AS manual
                FROM processing_log
                """,
                (start, start, start),
            ).fetchone()

        values = [0] * len(labels)
        classes = {class_name: 0 for class_name in CLASS_NAMES}
        for row in completed_rows:
            completed_at = row["completed_at"]
            index = period_bucket_index(period, completed_at)
            if 0 <= index < len(values):
                values[index] += 1
            if row["class_name"] in classes:
                classes[row["class_name"]] += 1

        return {
            "period": period,
            "range": range_text,
            "completed": len(completed_rows),
            "pending": status_counts["pending"],
            "failed": status_counts["failed"],
            "manual": status_counts["manual"],
            "timelineTitle": title,
            "labels": labels,
            "values": values,
            "classes": classes,
        }
