"""
PackSight SQLite Database Manager (packsight.db)

Normalized 4-tier relational schema:
  PRODUCT -> SCAN -> SCAN_IMAGE -> DECLARATION

Manages dynamic product registration, historical compliance audits,
multi-image associations, and declaration-level bounding boxes.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_FILE = PROJECT_ROOT / "packsight.db"


def get_connection(db_path: Path = DATABASE_FILE) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path = DATABASE_FILE) -> None:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS product (
                product_id TEXT PRIMARY KEY,
                brand_name TEXT NOT NULL DEFAULT 'Unknown Product',
                category TEXT DEFAULT 'Packaged Commodity',
                created_at TEXT NOT NULL,
                notes TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS scan (
                scan_id TEXT PRIMARY KEY,
                product_id TEXT NOT NULL,
                scanned_at TEXT NOT NULL,
                duration_seconds REAL DEFAULT 0.0,
                image_count INTEGER DEFAULT 1,
                overall_status TEXT NOT NULL,
                compliance_rate_percent REAL DEFAULT 0.0,
                mandatory_fields_present INTEGER DEFAULT 0,
                mandatory_fields_total INTEGER DEFAULT 9,
                missing_fields_json TEXT,
                uncertain_fields_json TEXT,
                FOREIGN KEY (product_id) REFERENCES product(product_id) ON DELETE CASCADE
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_image (
                image_id TEXT PRIMARY KEY,
                scan_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                side_label TEXT,
                orientation_selected TEXT,
                degrees_clockwise INTEGER DEFAULT 0,
                width INTEGER DEFAULT 0,
                height INTEGER DEFAULT 0,
                original_path TEXT,
                highlighted_path TEXT,
                FOREIGN KEY (scan_id) REFERENCES scan(scan_id) ON DELETE CASCADE
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS declaration (
                declaration_id TEXT PRIMARY KEY,
                scan_id TEXT NOT NULL,
                image_id TEXT,
                field_name TEXT NOT NULL,
                extracted_value TEXT,
                ocr_raw_text TEXT,
                confidence REAL DEFAULT 0.0,
                bounding_box_json TEXT,
                source_image TEXT,
                status TEXT NOT NULL,
                detection_status TEXT DEFAULT 'MISSING',
                is_mandatory INTEGER NOT NULL DEFAULT 1,
                detection_reason TEXT,
                needs_review INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (scan_id) REFERENCES scan(scan_id) ON DELETE CASCADE,
                FOREIGN KEY (image_id) REFERENCES scan_image(image_id) ON DELETE SET NULL
            )
            """
        )

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_product ON scan(product_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_date ON scan(scanned_at DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_scan ON scan_image(scan_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_decl_scan ON declaration(scan_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_decl_field ON declaration(field_name)")

        # Migration: ensure detection_status column exists in declaration table
        cursor.execute("PRAGMA table_info(declaration)")
        decl_cols = [row["name"] for row in cursor.fetchall()]
        if "detection_status" not in decl_cols:
            cursor.execute("ALTER TABLE declaration ADD COLUMN detection_status TEXT DEFAULT 'MISSING'")

        conn.commit()


def get_next_product_id(prefix: str = "PS-2026", db_path: Path = DATABASE_FILE) -> str:
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT product_id FROM product WHERE product_id LIKE ? ORDER BY product_id DESC",
            (f"{prefix}-%",),
        )
        rows = cursor.fetchall()
        max_num = 0
        for row in rows:
            pid = row["product_id"]
            try:
                num = int(pid.split("-")[-1])
                if num > max_num:
                    max_num = num
            except (ValueError, IndexError):
                continue

    input_dir = PROJECT_ROOT / "input"
    if input_dir.exists():
        for p in input_dir.iterdir():
            if p.is_dir() and p.name.startswith(f"{prefix}-"):
                try:
                    num = int(p.name.split("-")[-1])
                    if num > max_num:
                        max_num = num
                except (ValueError, IndexError):
                    continue

    next_num = max_num + 1
    return f"{prefix}-{next_num:04d}"


def save_scan_result(result: dict[str, Any], db_path: Path = DATABASE_FILE) -> str:
    init_db(db_path)

    product_id = result.get("product", "Unknown Product")
    brand_name = result.get("brand_name") or product_id
    if brand_name.startswith("PS-") or brand_name.startswith("product-"):
        brand_name = result.get("brand_name", "Unknown Product")

    scan_id = result.get("scan_id") or str(datetime.now(timezone.utc).timestamp())
    scanned_at = result.get("scanned_at") or datetime.now(timezone.utc).isoformat()
    duration_seconds = float(result.get("duration_seconds", 0.0))

    images_dict = result.get("images", {})
    image_count = len(images_dict)

    compliance = result.get("compliance", {})
    overall_status = compliance.get("overall_status", "UNKNOWN")
    compliance_rate = float(compliance.get("compliance_rate_percent", 0.0))
    mandatory_present = int(compliance.get("mandatory_fields_present", 0))
    mandatory_total = int(compliance.get("mandatory_fields_total", 9))
    missing_fields = compliance.get("missing_mandatory_fields", [])
    field_breakdown = compliance.get("field_breakdown", {})

    uncertain_fields = [
        f for f, d in field_breakdown.items()
        if d.get("needs_review") or d.get("status") == "REVIEW"
    ]

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO product (product_id, brand_name, category, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(product_id) DO UPDATE SET
                brand_name = CASE WHEN excluded.brand_name != 'Unknown Product' THEN excluded.brand_name ELSE product.brand_name END
            """,
            (product_id, brand_name, "Packaged Commodity", scanned_at),
        )

        cursor.execute(
            """
            INSERT OR REPLACE INTO scan (
                scan_id, product_id, scanned_at, duration_seconds, image_count,
                overall_status, compliance_rate_percent, mandatory_fields_present,
                mandatory_fields_total, missing_fields_json, uncertain_fields_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scan_id,
                product_id,
                scanned_at,
                duration_seconds,
                image_count,
                overall_status,
                compliance_rate,
                mandatory_present,
                mandatory_total,
                json.dumps(missing_fields),
                json.dumps(uncertain_fields),
            ),
        )

        image_id_map: dict[str, str] = {}
        for idx, (img_name, img_data) in enumerate(images_dict.items(), start=1):
            img_id = f"{scan_id}_img_{idx}"
            image_id_map[img_name] = img_id
            orient = img_data.get("orientation", {})
            orig_dims = img_data.get("source_dimensions", {})
            annot_dims = img_data.get("annotated_dimensions", {})

            cursor.execute(
                """
                INSERT OR REPLACE INTO scan_image (
                    image_id, scan_id, filename, side_label, orientation_selected,
                    degrees_clockwise, width, height, original_path, highlighted_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    img_id,
                    scan_id,
                    img_name,
                    img_name.split(".")[0],
                    orient.get("selected", "0"),
                    int(orient.get("degrees_clockwise", 0)),
                    int(annot_dims.get("width") or orig_dims.get("width") or 0),
                    int(annot_dims.get("height") or orig_dims.get("height") or 0),
                    img_data.get("original_image"),
                    img_data.get("highlighted_image"),
                ),
            )

        for f_idx, (f_name, f_data) in enumerate(field_breakdown.items(), start=1):
            decl_id = f"{scan_id}_decl_{f_idx}"
            source_img = f_data.get("source_image")
            img_id = image_id_map.get(source_img) if source_img else None
            bbox = f_data.get("bounding_box")

            cursor.execute(
                """
                INSERT OR REPLACE INTO declaration (
                    declaration_id, scan_id, image_id, field_name, extracted_value,
                    ocr_raw_text, confidence, bounding_box_json, source_image,
                    status, detection_status, is_mandatory, detection_reason, needs_review
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decl_id,
                    scan_id,
                    img_id,
                    f_name,
                    f_data.get("value"),
                    f_data.get("text"),
                    float(f_data.get("confidence") or 0.0),
                    json.dumps(bbox) if bbox else None,
                    source_img,
                    f_data.get("status", "MISSING"),
                    f_data.get("detection_status", "MISSING"),
                    1 if f_data.get("mandatory", True) else 0,
                    f_data.get("reason"),
                    1 if f_data.get("needs_review") else 0,
                ),
            )

        conn.commit()

    return scan_id


def get_all_scans(db_path: Path = DATABASE_FILE) -> list[dict[str, Any]]:
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                s.scan_id,
                s.product_id,
                p.brand_name,
                s.scanned_at,
                s.duration_seconds,
                s.image_count,
                s.overall_status,
                s.compliance_rate_percent,
                s.mandatory_fields_present,
                s.mandatory_fields_total
            FROM scan s
            LEFT JOIN product p ON s.product_id = p.product_id
            ORDER BY s.scanned_at DESC
            """
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_scan_detail(scan_id: str, db_path: Path = DATABASE_FILE) -> Optional[dict[str, Any]]:
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s.*, p.brand_name, p.category
            FROM scan s
            LEFT JOIN product p ON s.product_id = p.product_id
            WHERE s.scan_id = ?
            """,
            (scan_id,),
        )
        scan_row = cursor.fetchone()
        if not scan_row:
            return None

        scan_dict = dict(scan_row)
        scan_dict["missing_fields"] = json.loads(scan_dict.pop("missing_fields_json") or "[]")
        scan_dict["uncertain_fields"] = json.loads(scan_dict.pop("uncertain_fields_json") or "[]")

        cursor.execute("SELECT * FROM scan_image WHERE scan_id = ? ORDER BY filename", (scan_id,))
        images = [dict(r) for r in cursor.fetchall()]
        scan_dict["images"] = images

        cursor.execute("SELECT * FROM declaration WHERE scan_id = ? ORDER BY is_mandatory DESC, field_name", (scan_id,))
        declarations = []
        for r in cursor.fetchall():
            d = dict(r)
            if d.get("bounding_box_json"):
                d["bounding_box"] = json.loads(d.pop("bounding_box_json"))
            else:
                d.pop("bounding_box_json", None)
                d["bounding_box"] = None
            d["is_mandatory"] = bool(d["is_mandatory"])
            d["needs_review"] = bool(d["needs_review"])
            d["reason"] = d.get("detection_reason")
            d["detection_status"] = d.get("detection_status") or ("DETECTED" if d.get("extracted_value") and d.get("confidence", 0) >= 30.0 else "REVIEW" if d.get("extracted_value") else "MISSING")
            declarations.append(d)
        scan_dict["declarations"] = declarations

        scan_dict["compliance"] = {
            "overall_status": scan_dict.get("overall_status"),
            "compliance_rate_percent": scan_dict.get("compliance_rate_percent"),
            "mandatory_fields_present": scan_dict.get("mandatory_fields_present"),
            "mandatory_fields_total": scan_dict.get("mandatory_fields_total"),
            "missing_mandatory_fields": scan_dict.get("missing_fields", []),
            "uncertain_mandatory_fields": scan_dict.get("uncertain_fields", []),
            "field_breakdown": {
                d["field_name"]: {
                    "field": d["field_name"],
                    "status": d["status"],
                    "detection_status": d.get("detection_status", "MISSING"),
                    "mandatory": d["is_mandatory"],
                    "value": d["extracted_value"],
                    "confidence": d["confidence"],
                    "bounding_box": d.get("bounding_box"),
                    "source_image": d.get("source_image"),
                    "needs_review": d["needs_review"],
                    "reason": d.get("reason"),
                    "text": d.get("ocr_raw_text"),
                }
                for d in declarations
            },
        }

        return scan_dict
