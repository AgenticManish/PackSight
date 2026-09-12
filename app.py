"""
PackSight FastAPI Application
Provides RESTful APIs for packaged commodity compliance scanning,
live camera frame quality assessment, history tracking, and static web UI.
"""

from __future__ import annotations

import base64
import json
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scanner import (
    ALL_SUPPORTED_FIELDS,
    INPUT_FOLDER,
    MANDATORY_FIELDS,
    OPTIONAL_FIELDS,
    OUTPUT_FOLDER,
    SCAN_HISTORY_FILE,
    assess_image_quality,
    configure_tesseract,
    evaluate_compliance,
    record_scan_history,
    scan_product,
)
from database import (
    get_all_scans,
    get_next_product_id,
    get_scan_detail,
    save_scan_result,
)

# Initialize and configure Tesseract
try:
    configure_tesseract()
except Exception as e:
    print(f"Warning: Tesseract initialization warning: {e}")

PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_FOLDER = PROJECT_ROOT / "static"
STATIC_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
INPUT_FOLDER.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="PackSight - Legal Metrology Compliance Scanner",
    description="Automated packaged-commodity compliance verification engine",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QualityCheckRequest(BaseModel):
    image: str  # Base64 encoded image string (data URL or raw base64)


class Base64ScanRequest(BaseModel):
    images: list[str]  # List of base64 encoded images
    product_name: Optional[str] = None


def decode_base64_image(b64_str: str) -> np.ndarray:
    """Decode base64 string to OpenCV BGR image."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from provided base64 data.")
    return img


def enhance_result_with_urls(result: dict[str, Any], product_name: str) -> dict[str, Any]:
    """Add static URL paths to images for convenient frontend consumption."""
    enriched = dict(result)
    for img_name, img_data in enriched.get("images", {}).items():
        if "highlighted_image" in img_data:
            img_data["highlighted_url"] = f"/output/{product_name}/{img_data['highlighted_image']}"
        if "original_image" in img_data:
            img_data["original_url"] = f"/output/{product_name}/{img_data['original_image']}"
    return enriched


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tesseract_configured": True,
    }


@app.get("/api/fields")
async def get_supported_fields() -> dict[str, Any]:
    return {
        "all_fields": ALL_SUPPORTED_FIELDS,
        "mandatory_fields": sorted(MANDATORY_FIELDS),
        "optional_fields": sorted(OPTIONAL_FIELDS),
    }


@app.post("/api/quality-check")
async def quality_check(payload: QualityCheckRequest) -> dict[str, Any]:
    """Evaluate camera frame quality (sharpness, lighting, glare) in real-time."""
    try:
        image = decode_base64_image(payload.image)
        assessment = assess_image_quality(image)
        return assessment
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Quality check failed: {exc}")


@app.get("/api/next-id")
async def get_next_id() -> dict[str, str]:
    """Get the next sequential dynamic product ID (e.g. PS-2026-0001)."""
    return {"next_product_id": get_next_product_id()}


@app.post("/api/scan")
async def scan_uploaded_images(
    files: list[UploadFile] = File(...),
    product_name: Optional[str] = Form(None),
) -> dict[str, Any]:
    """Scan one or more uploaded package photos. Auto-generates PS-2026-XXXX if not specified."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    if not product_name or not product_name.strip() or product_name.strip().lower() in ("new", "auto", "default"):
        scan_tag = get_next_product_id()
    else:
        scan_tag = product_name.strip()

    clean_tag = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in scan_tag)
    target_dir = INPUT_FOLDER / clean_tag
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        saved_files = []
        for file in files:
            file_suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
            save_name = f"image_{len(saved_files) + 1}{file_suffix}"
            save_path = target_dir / save_name
            with save_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(save_path)

        result = scan_product(clean_tag, target_dir)
        return enhance_result_with_urls(result, clean_tag)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scan execution error: {exc}")


@app.post("/api/scan-base64")
async def scan_base64_images(payload: Base64ScanRequest) -> dict[str, Any]:
    """Scan images captured directly from camera via base64. Auto-generates PS-2026-XXXX if not specified."""
    if not payload.images:
        raise HTTPException(status_code=400, detail="No images provided.")

    if not payload.product_name or not payload.product_name.strip() or payload.product_name.strip().lower() in ("new", "auto", "default"):
        scan_tag = get_next_product_id()
    else:
        scan_tag = payload.product_name.strip()

    clean_tag = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in scan_tag)
    target_dir = INPUT_FOLDER / clean_tag
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        for idx, b64_str in enumerate(payload.images, start=1):
            img = decode_base64_image(b64_str)
            cv2.imwrite(str(target_dir / f"capture_{idx}.jpg"), img)

        result = scan_product(clean_tag, target_dir)
        return enhance_result_with_urls(result, clean_tag)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Live scan execution error: {exc}")


@app.post("/api/test-product/{product_id}")
async def run_test_product(product_id: str) -> dict[str, Any]:
    """Instantly run compliance scan on an existing product in input folder."""
    target_dir = INPUT_FOLDER / product_id
    if not target_dir.exists() or not target_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Product folder '{product_id}' not found under input/ directory.",
        )

    try:
        result = scan_product(product_id, target_dir)
        return enhance_result_with_urls(result, product_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Test product scan failed: {exc}")


@app.get("/api/history")
async def get_history() -> list[dict[str, Any]]:
    """Retrieve full history of all compliance scans from SQLite database."""
    try:
        scans = get_all_scans()
        if scans:
            return scans
    except Exception as exc:
        print(f"Error reading SQLite history: {exc}")

    if not SCAN_HISTORY_FILE.exists():
        return []
    try:
        with SCAN_HISTORY_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return list(reversed(data))
            return []
    except Exception:
        return []


@app.get("/api/scan/{scan_id}")
async def get_scan_by_id(scan_id: str) -> dict[str, Any]:
    """Retrieve complete audit record with relational declarations from SQLite database."""
    detail = get_scan_detail(scan_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Scan '{scan_id}' not found in database.")
    return detail


@app.get("/api/product/{product_id}")
async def get_product_data(product_id: str) -> dict[str, Any]:
    """Retrieve compliance JSON for a previously scanned product."""
    data_file = OUTPUT_FOLDER / product_id / "data.json"
    if data_file.exists():
        try:
            with data_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return enhance_result_with_urls(data, product_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error reading product data: {exc}")

    # Fallback to SQLite scan lookup
    try:
        scans = get_all_scans()
        for s in scans:
            if s.get("product_id") == product_id:
                detail = get_scan_detail(s["scan_id"])
                if detail:
                    return detail
    except Exception:
        pass

    raise HTTPException(status_code=404, detail=f"Data for product '{product_id}' not found.")


# Mount static assets and generated output folders
app.mount("/output", StaticFiles(directory=str(OUTPUT_FOLDER)), name="output")
app.mount("/static", StaticFiles(directory=str(STATIC_FOLDER)), name="static")


@app.get("/")
async def root():
    """Serve index.html at root route."""
    index_path = STATIC_FOLDER / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"message": "PackSight API is running. UI index.html not found."})
