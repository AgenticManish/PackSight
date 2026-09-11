"""
PackSight product-label scanner.

The scanner intentionally stays local and lightweight: OpenCV prepares product
photos, Tesseract supplies word-level OCR boxes, and JSON/image artifacts are
saved per product. This makes the output straightforward to move into a
SQLite/FastAPI layer later without changing the scanning workflow.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


# ---------------------------------------------------------------------------
# Project paths and OCR configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_FOLDER = PROJECT_ROOT / "input"
OUTPUT_FOLDER = PROJECT_ROOT / "output"

# A moderate scale is enough for phone photos while keeping scans responsive.
OCR_SCALE = 2.0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------------------
# Packaged-commodity declarations supported by PackSight
# ---------------------------------------------------------------------------

FIELD_PATTERNS: dict[str, tuple[str, ...]] = {
    "MRP": (
        r"\bmrp\b",
        r"\bmaximum\s+retail\s+price\b",
    ),
    "NET QUANTITY": (
        r"\bnet\s*(?:quantity|qty|weight|wt)\b",
        r"\bnetquantity\b",
    ),
    "MANUFACTURER": (
        r"\bmanufactured\s+(?:and\s+)?(?:packed\s+)?by\b",
        r"\b(?:mfg|mfd)\s+by\b",
        r"\bpacked\s+by\b",
    ),
    "MARKETED BY": (
        r"\bmarketed\s+by\b",
        r"\bmkt\s+by\b",
    ),
    "CUSTOMER CARE": (
        r"\b(?:customer|consumer)\s+(?:care|support)\b",
        r"\b(?:helpline|toll\s*free)\b",
        r"\bwecare\b",
    ),
    "MFG / PACKING DATE": (
        r"\b(?:mfg|mfd)\b(?:\s*(?:date|on))?",
        r"\bmanufacturing\s+date\b",
        r"\b(?:packed|packing)\s+(?:on|date)\b",
    ),
    "USE BY / BEST BEFORE": (
        r"\buse\s*by\b",
        r"\bbest\s*before\b",
        r"\b(?:expiry|expires|expiration)\b",
    ),
    "LOT / BATCH": (
        r"\blot\s*(?:no|number)?\b",
        r"\bbatch\s*(?:no|number)?\b",
    ),
    "COUNTRY OF ORIGIN": (
        r"\bcountry\s+of\s+origin\b",
        r"\bmade\s+in\b",
    ),
    "LICENSE": (
        r"\blic(?:ence|ense)?\s*(?:no|number)?\b",
        r"\bfssai\b",
    ),
    "INGREDIENTS": (
        r"\bingredients?\b",
    ),
    "MULTI UNIT PACKAGE": (
        r"\bmulti\s*(?:unit|pack)\b",
    ),
}

COMPILED_FIELD_PATTERNS = {
    field: tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
    for field, patterns in FIELD_PATTERNS.items()
}

DATE_PATTERN = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    r"[\s./_-]*[0-9ilzs]{1,4}\b",
    re.IGNORECASE,
)
NUMERIC_DATE_PATTERN = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+\s*@\s*[\w.-]+\.[a-z]{2,}\b", re.I)
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?\d[\d\s()-]{7,}\d)(?!\d)")
ALPHANUMERIC_CODE_PATTERN = re.compile(
    r"\b(?=[a-z0-9]{6,20}\b)(?=.*\d)[a-z0-9]+\b", re.I
)

FIELD_COLORS: dict[str, tuple[int, int, int]] = {
    "MRP": (0, 196, 255),
    "NET QUANTITY": (0, 220, 110),
    "MANUFACTURER": (255, 150, 0),
    "MARKETED BY": (255, 90, 170),
    "CUSTOMER CARE": (255, 80, 80),
    "MFG / PACKING DATE": (170, 90, 255),
    "USE BY / BEST BEFORE": (190, 0, 255),
    "LOT / BATCH": (0, 235, 235),
    "COUNTRY OF ORIGIN": (130, 190, 0),
    "LICENSE": (60, 180, 255),
    "INGREDIENTS": (60, 255, 200),
    "MULTI UNIT PACKAGE": (255, 210, 0),
}

FIELD_CONTEXT_RADIUS: dict[str, int] = {
    "CUSTOMER CARE": 180,
    "MANUFACTURER": 150,
    "MARKETED BY": 150,
    "MFG / PACKING DATE": 140,
    "USE BY / BEST BEFORE": 140,
    "LOT / BATCH": 140,
    "LICENSE": 100,
    "INGREDIENTS": 120,
}


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def configure_tesseract() -> str:
    """Find Tesseract in a user-friendly, portable way."""

    candidate_strings = [
        os.environ.get("TESSERACT_CMD", ""),
        shutil.which("tesseract") or "",
        os.path.join(os.environ.get("ProgramFiles", ""), "Tesseract-OCR", "tesseract.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    ]

    for candidate in candidate_strings:
        if candidate and Path(candidate).is_file():
            pytesseract.pytesseract.tesseract_cmd = str(Path(candidate))
            return str(Path(candidate))

    raise RuntimeError(
        "Tesseract was not found. Install Tesseract-OCR, add it to PATH, or "
        "set the TESSERACT_CMD environment variable to tesseract.exe."
    )


def natural_key(path: Path) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    ]


def clean_text(text: str) -> str:
    """Normalize OCR punctuation while retaining words and useful numbers."""

    text = text.lower().replace("₹", " rs ").replace("€", " ").replace("£", " ")
    text = text.replace("@", " at ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def safe_confidence(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def box_from_words(words: Iterable[dict[str, Any]]) -> dict[str, int]:
    words = list(words)
    x1 = min(word["x"] for word in words)
    y1 = min(word["y"] for word in words)
    x2 = max(word["x"] + word["w"] for word in words)
    y2 = max(word["y"] + word["h"] for word in words)
    return {
        "x1": int(round(x1)),
        "y1": int(round(y1)),
        "x2": int(round(x2)),
        "y2": int(round(y2)),
    }


def merge_boxes(boxes: Iterable[dict[str, int]]) -> dict[str, int]:
    boxes = list(boxes)
    return {
        "x1": min(box["x1"] for box in boxes),
        "y1": min(box["y1"] for box in boxes),
        "x2": max(box["x2"] for box in boxes),
        "y2": max(box["y2"] for box in boxes),
    }


def clamp_box(box: dict[str, int], width: int, height: int, padding: int = 0) -> dict[str, int]:
    return {
        "x1": max(0, box["x1"] - padding),
        "y1": max(0, box["y1"] - padding),
        "x2": min(width - 1, box["x2"] + padding),
        "y2": min(height - 1, box["y2"] + padding),
    }


def box_center(box: dict[str, int]) -> tuple[float, float]:
    return ((box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2)


def horizontal_gap(first: dict[str, int], second: dict[str, int]) -> int:
    if first["x2"] < second["x1"]:
        return second["x1"] - first["x2"]
    if second["x2"] < first["x1"]:
        return first["x1"] - second["x2"]
    return 0


# ---------------------------------------------------------------------------
# Image orientation and preprocessing
# ---------------------------------------------------------------------------

def rotated_views(image: np.ndarray) -> dict[str, tuple[np.ndarray, int]]:
    """Return all plausible text orientations and their clockwise degrees."""

    return {
        "0": (image, 0),
        "90_CLOCKWISE": (cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 90),
        "90_COUNTERCLOCKWISE": (
            cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
            270,
        ),
        "180": (cv2.rotate(image, cv2.ROTATE_180), 180),
    }


def resized_gray(image: np.ndarray, scale: float) -> np.ndarray:
    scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)


def orientation_score(image: np.ndarray) -> float:
    """Score readable OCR, rewarding label-like words over photographic noise."""

    preview = resized_gray(image, 1.25)
    data = pytesseract.image_to_data(
        preview,
        output_type=Output.DICT,
        config="--oem 3 --psm 11",
    )

    words: list[tuple[str, float]] = []
    for text, raw_confidence in zip(data["text"], data["conf"]):
        text = text.strip()
        confidence = safe_confidence(raw_confidence)
        alphanumeric = sum(char.isalnum() for char in text)
        if not text or alphanumeric == 0:
            continue
        ratio = alphanumeric / len(text)
        if confidence < 5 or ratio < 0.6:
            continue
        words.append((text, confidence))

    readable = sum((confidence + 5) * min(len(text), 18) for text, confidence in words)
    joined = clean_text(" ".join(text for text, _ in words))
    label_hits = sum(
        1
        for patterns in COMPILED_FIELD_PATTERNS.values()
        if any(pattern.search(joined) for pattern in patterns)
    )
    date_bonus = 70 if DATE_PATTERN.search(joined) or NUMERIC_DATE_PATTERN.search(joined) else 0
    long_word_bonus = sum(10 for text, _ in words if len(text) >= 5 and text.isalpha())
    return readable + label_hits * 240 + date_bonus + long_word_bonus


def find_best_orientation(image: np.ndarray) -> tuple[str, np.ndarray, int, dict[str, int]]:
    scores: dict[str, int] = {}
    selected_name = "0"
    selected_image = image
    selected_degrees = 0
    best_score = float("-inf")

    for name, (candidate, degrees) in rotated_views(image).items():
        score = orientation_score(candidate)
        scores[name] = int(round(score))
        if score > best_score:
            best_score = score
            selected_name = name
            selected_image = candidate
            selected_degrees = degrees

    return selected_name, selected_image, selected_degrees, scores


def preprocess_variants(image: np.ndarray, scale: float = OCR_SCALE) -> dict[str, np.ndarray]:
    """Create complementary OCR views for bright, dark, and coloured labels."""

    scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)
    red_channel = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(scaled[:, :, 2])
    adaptive = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )
    return {
        "clahe": clahe,
        "adaptive": adaptive,
        "red_channel": red_channel,
    }


# ---------------------------------------------------------------------------
# OCR line construction
# ---------------------------------------------------------------------------

def ocr_lines(
    source_image: np.ndarray,
    *,
    variant_name: str,
    psm: int,
    region_name: str,
    offset_x: int = 0,
    offset_y: int = 0,
) -> list[dict[str, Any]]:
    """Run word-box OCR and reconstruct Tesseract lines in source coordinates."""

    data = pytesseract.image_to_data(
        source_image,
        output_type=Output.DICT,
        config=f"--oem 3 --psm {psm}",
    )
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)

    for index, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        if not text:
            continue

        confidence = safe_confidence(data["conf"][index])
        if confidence < 0:
            continue

        word = {
            "text": text,
            "confidence": confidence,
            "x": data["left"][index] / OCR_SCALE + offset_x,
            "y": data["top"][index] / OCR_SCALE + offset_y,
            "w": data["width"][index] / OCR_SCALE,
            "h": data["height"][index] / OCR_SCALE,
        }
        key = (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )
        grouped[key].append(word)

    lines: list[dict[str, Any]] = []
    for words in grouped.values():
        words.sort(key=lambda word: word["x"])
        line_text = " ".join(word["text"] for word in words)
        if not any(character.isalnum() for character in line_text):
            continue
        confidence = sum(word["confidence"] for word in words) / len(words)
        lines.append(
            {
                "text": line_text,
                "confidence": round(confidence, 2),
                "box": box_from_words(words),
                "variant": variant_name,
                "psm": psm,
                "region": region_name,
            }
        )

    lines.sort(key=lambda line: (line["box"]["y1"], line["box"]["x1"]))
    return lines


def ocr_passes(image: np.ndarray) -> list[tuple[str, np.ndarray, int, str, int, int]]:
    """Return full-image and focused passes without changing the public layout."""

    height, width = image.shape[:2]
    passes: list[tuple[str, np.ndarray, int, str, int, int]] = []

    variants = preprocess_variants(image)
    for variant_name, variant in variants.items():
        passes.append((variant_name, variant, 6, "full", 0, 0))

    # Sparse-text mode catches isolated address, price, and phone fields.
    passes.append(("clahe", variants["clahe"], 11, "full", 0, 0))

    # Product declarations are commonly printed in narrow upper/lower panels.
    regions = {
        "top_panel": (0, 0, width, max(1, int(height * 0.58))),
        "bottom_panel": (0, int(height * 0.38), width, height),
    }
    for region_name, (x1, y1, x2, y2) in regions.items():
        crop = image[y1:y2, x1:x2]
        crop_variants = preprocess_variants(crop)
        for variant_name in ("clahe", "red_channel"):
            passes.append(
                (
                    variant_name,
                    crop_variants[variant_name],
                    6,
                    region_name,
                    x1,
                    y1,
                )
            )
    return passes


# ---------------------------------------------------------------------------
# Declaration matching and value extraction
# ---------------------------------------------------------------------------

def matched_fields(text: str) -> list[str]:
    normalized = clean_text(text)
    return [
        field
        for field, patterns in COMPILED_FIELD_PATTERNS.items()
        if any(pattern.search(normalized) for pattern in patterns)
    ]


def canonical_month_date(raw_value: str) -> str:
    """Turn tolerant OCR such as JUNI26 into a compact human-readable date."""

    clean = clean_text(raw_value)
    match = re.search(
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*([0-9ilzs]{1,4})",
        clean,
        re.I,
    )
    if not match:
        return raw_value.strip()
    replacement_table = str.maketrans({"i": "1", "l": "1", "z": "2", "s": "5"})
    year = match.group(2).lower().translate(replacement_table)
    return f"{match.group(1).upper()}/{year}"


def extract_value(field: str, text: str) -> str | None:
    raw = " ".join(text.split())
    normalized = clean_text(raw)

    if field == "MRP":
        match = re.search(
            r"(?:\bmrp\b|maximum retail price)[^0-9]{0,20}"
            r"(?:rs\s*)?([0-9]{1,5}(?:[.,][0-9]{1,2})?)",
            normalized,
            re.I,
        )
        return f"₹{match.group(1).replace(',', '.')}" if match else None

    if field == "NET QUANTITY":
        match = re.search(
            r"(?:net\s*(?:quantity|qty|weight|wt)?)[^0-9]{0,24}"
            r"([0-9]{1,4}(?:[.,][0-9]+)?\s*(?:kg|g|gm|grams?|ml|l))\b",
            normalized,
            re.I,
        )
        return match.group(1).replace(",", ".") if match else None

    if field in {"MFG / PACKING DATE", "USE BY / BEST BEFORE"}:
        dates = DATE_PATTERN.findall(raw)
        if dates:
            return canonical_month_date(dates[0])
        numeric = NUMERIC_DATE_PATTERN.search(raw)
        return numeric.group(0) if numeric else None

    if field == "LOT / BATCH":
        match = re.search(
            r"(?:lot|batch)\s*(?:no|number)?\s*[:#-]?\s*([a-z0-9-]{5,})",
            normalized,
            re.I,
        )
        return match.group(1).upper() if match else None

    if field == "LICENSE":
        match = re.search(
            r"(?:lic(?:ence|ense)?|fssai)\s*(?:no|number)?\s*"
            r"([a-z0-9-]{7,})",
            normalized,
            re.I,
        )
        return match.group(1).upper() if match else None

    if field == "CUSTOMER CARE":
        email = EMAIL_PATTERN.search(raw)
        phone = PHONE_PATTERN.search(raw)
        values = []
        if email:
            values.append(re.sub(r"\s+", "", email.group(0)))
        if phone:
            values.append(re.sub(r"\s+", " ", phone.group(0)).strip())
        return " | ".join(values) if values else None

    return None


def nearby_context(
    anchor: dict[str, Any],
    lines: list[dict[str, Any]],
    field: str,
    image_width: int,
) -> tuple[str, dict[str, int], float]:
    """Expand a label into nearby text/value lines while keeping boxes focused."""

    anchor_box = anchor["box"]
    anchor_center_y = box_center(anchor_box)[1]
    radius = FIELD_CONTEXT_RADIUS.get(field, 75)
    selected = [anchor]
    candidates: list[tuple[float, dict[str, Any]]] = []

    for line in lines:
        if line is anchor:
            continue
        line_box = line["box"]
        vertical_distance = abs(box_center(line_box)[1] - anchor_center_y)
        if vertical_distance > radius:
            continue
        if horizontal_gap(anchor_box, line_box) > max(80, int(image_width * 0.16)):
            continue
        candidates.append((vertical_distance, line))

    for _, line in sorted(candidates, key=lambda item: item[0])[:5]:
        selected.append(line)

    selected.sort(key=lambda line: (line["box"]["y1"], line["box"]["x1"]))
    context_text = " ".join(line["text"] for line in selected)
    context_box = merge_boxes(line["box"] for line in selected)
    confidence = sum(line["confidence"] for line in selected) / len(selected)
    return context_text, context_box, confidence


def candidate_score(
    field: str,
    confidence: float,
    value: str | None,
    source_text: str,
    region: str,
) -> float:
    score = max(0.0, confidence)
    if value:
        score += 42
    if field in matched_fields(source_text):
        score += 28
    if region != "full":
        score += 5
    return score + min(len(clean_text(source_text)), 80) * 0.18


def make_candidate(
    field: str,
    line: dict[str, Any],
    pass_lines: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    *,
    text_override: str | None = None,
    value_override: str | None = None,
    reason: str,
) -> dict[str, Any]:
    context_text, context_box, context_confidence = nearby_context(
        line, pass_lines, field, image_width
    )
    text = text_override or context_text
    value = value_override if value_override is not None else extract_value(field, text)
    box = clamp_box(context_box, image_width, image_height, padding=7)
    return {
        "field": field,
        "value": value,
        "text": text,
        "confidence": round(context_confidence, 2),
        "bounding_box": box,
        "ocr": {
            "variant": line["variant"],
            "psm": line["psm"],
            "region": line["region"],
            "reason": reason,
        },
        "_score": candidate_score(field, context_confidence, value, text, line["region"]),
    }


def date_candidates(
    lines: list[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for line in lines:
        raw_matches = DATE_PATTERN.findall(line["text"])
        numeric_matches = NUMERIC_DATE_PATTERN.findall(line["text"])
        values = [canonical_month_date(match) for match in raw_matches] + numeric_matches
        if not values:
            continue
        candidates.append(
            make_candidate(
                "MFG / PACKING DATE",
                line,
                lines,
                image_width,
                image_height,
                value_override=values[0],
                reason="date_pattern",
            )
        )
        if len(values) >= 2:
            candidates.append(
                make_candidate(
                    "USE BY / BEST BEFORE",
                    line,
                    lines,
                    image_width,
                    image_height,
                    value_override=values[1],
                    reason="date_range_pattern",
                )
            )
    return candidates


def code_candidates_near_dates(
    lines: list[dict[str, Any]],
    date_lines: list[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    """Treat a nearby alphanumeric production code as a lot/batch candidate."""

    if not date_lines:
        return []
    candidates: list[dict[str, Any]] = []
    for line in lines:
        normalized = clean_text(line["text"])
        codes = ALPHANUMERIC_CODE_PATTERN.findall(normalized)
        if not codes:
            continue
        line_x, line_y = box_center(line["box"])
        closest = min(
            abs(line_y - box_center(date_line["box"])[1])
            + 0.35 * abs(line_x - box_center(date_line["box"])[0])
            for date_line in date_lines
        )
        if closest > max(260, image_height * 0.32):
            continue
        # Do not turn JUN/26 (which OCR may emit as JUNI26) into a batch ID.
        usable_codes = [
            code
            for code in codes
            if not DATE_PATTERN.fullmatch(code)
            and not re.match(
                r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[0-9]+$",
                code,
                re.IGNORECASE,
            )
        ]
        if not usable_codes:
            continue
        code = max(usable_codes, key=len).upper()
        if code.isdigit() and len(code) < 7:
            continue
        candidates.append(
            make_candidate(
                "LOT / BATCH",
                line,
                lines,
                image_width,
                image_height,
                value_override=code,
                reason="production_code_near_date",
            )
        )
    return candidates


def customer_contact_candidates(
    lines: list[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for line in lines:
        # A bare run of digits can just as easily be a lot number, date, or
        # barcode.  A standalone email is strong evidence of customer care;
        # phone numbers are associated through the nearby CUSTOMER CARE label.
        if not EMAIL_PATTERN.search(line["text"]):
            continue
        candidates.append(
            make_candidate(
                "CUSTOMER CARE",
                line,
                lines,
                image_width,
                image_height,
                reason="contact_pattern",
            )
        )
    return candidates


def deduplicate_candidates(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Choose the best evidence for each field without discarding text details."""

    best: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        field = candidate["field"]
        existing = best.get(field)
        if existing is None or candidate["_score"] > existing["_score"]:
            best[field] = candidate
    for candidate in best.values():
        candidate.pop("_score", None)
    return best


# ---------------------------------------------------------------------------
# Image annotation and product scanning
# ---------------------------------------------------------------------------

def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(origin[0], image.shape[1] - text_width - 8))
    y = max(text_height + 8, origin[1])
    cv2.rectangle(
        image,
        (x - 4, y - text_height - 6),
        (x + text_width + 4, y + baseline + 4),
        color,
        thickness=cv2.FILLED,
    )
    cv2.putText(image, text, (x, y), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)


def annotate_image(image: np.ndarray, fields: dict[str, dict[str, Any]]) -> np.ndarray:
    """Draw high-contrast boxes on a full-colour, upright source image."""

    annotated = image.copy()
    overlay = annotated.copy()
    height, width = annotated.shape[:2]
    for field, detection in fields.items():
        box = clamp_box(detection["bounding_box"], width, height, padding=0)
        color = FIELD_COLORS.get(field, (0, 220, 255))
        cv2.rectangle(overlay, (box["x1"], box["y1"]), (box["x2"], box["y2"]), color, -1)
    cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0, annotated)

    for field, detection in fields.items():
        box = clamp_box(detection["bounding_box"], width, height, padding=0)
        color = FIELD_COLORS.get(field, (0, 220, 255))
        cv2.rectangle(annotated, (box["x1"], box["y1"]), (box["x2"], box["y2"]), color, 3)
        draw_label(annotated, field, (box["x1"], max(24, box["y1"] - 8)), color)
    return annotated


def scan_image(image_path: Path, output_path: Path) -> dict[str, Any]:
    print(f"\nScanning: {image_path.name}")
    original = cv2.imread(str(image_path))
    if original is None:
        raise ValueError(f"Could not open image: {image_path}")

    orientation_name, oriented, degrees, orientation_scores = find_best_orientation(original)
    print(f"  orientation: {orientation_name} ({degrees}° clockwise)")

    all_lines: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    date_source_lines: list[dict[str, Any]] = []
    height, width = oriented.shape[:2]

    for variant_name, prepared, psm, region_name, offset_x, offset_y in ocr_passes(oriented):
        pass_lines = ocr_lines(
            prepared,
            variant_name=variant_name,
            psm=psm,
            region_name=region_name,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        all_lines.extend(pass_lines)
        for line in pass_lines:
            for field in matched_fields(line["text"]):
                all_candidates.append(
                    make_candidate(
                        field,
                        line,
                        pass_lines,
                        width,
                        height,
                        reason="declaration_keyword",
                    )
                )

        pass_date_candidates = date_candidates(pass_lines, width, height)
        all_candidates.extend(pass_date_candidates)
        if pass_date_candidates:
            date_source_lines.extend(
                [
                    line
                    for line in pass_lines
                    if DATE_PATTERN.search(line["text"]) or NUMERIC_DATE_PATTERN.search(line["text"])
                ]
            )
        all_candidates.extend(customer_contact_candidates(pass_lines, width, height))

    all_candidates.extend(code_candidates_near_dates(all_lines, date_source_lines, width, height))
    fields = deduplicate_candidates(all_candidates)
    annotated = annotate_image(oriented, fields)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        raise IOError(f"Could not save highlighted image: {output_path}")

    for field, detection in fields.items():
        value = f" — {detection['value']}" if detection["value"] else ""
        # Windows consoles using cp1252 cannot print the rupee glyph.
        print(f"  [FOUND] {field}{value}".replace("₹", "Rs."))

    return {
        "orientation": {
            "selected": orientation_name,
            "degrees_clockwise": degrees,
            "scores": orientation_scores,
        },
        "source_dimensions": {
            "width": int(original.shape[1]),
            "height": int(original.shape[0]),
        },
        "annotated_dimensions": {"width": width, "height": height},
        "highlighted_image": output_path.name,
        "fields": fields,
    }


def aggregate_fields(images: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Pick the strongest per-product declaration while retaining source image."""

    candidates: list[tuple[str, dict[str, Any]]] = []
    for filename, image_result in images.items():
        for field, detection in image_result.get("fields", {}).items():
            copy = json.loads(json.dumps(detection))
            copy["source_image"] = filename
            candidates.append((field, copy))

    best: dict[str, dict[str, Any]] = {}
    for field, candidate in candidates:
        existing = best.get(field)
        candidate_rank = candidate["confidence"] + (35 if candidate.get("value") else 0)
        existing_rank = (
            existing["confidence"] + (35 if existing.get("value") else 0)
            if existing
            else float("-inf")
        )
        if candidate_rank > existing_rank:
            best[field] = candidate
    return best


def scan_product(product_name: str, product_input_folder: Path) -> dict[str, Any]:
    product_output_folder = OUTPUT_FOLDER / product_name
    product_output_folder.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        (
            path
            for path in product_input_folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_key,
    )
    if not image_paths:
        raise ValueError(f"No supported images found in {product_input_folder}")

    image_results: dict[str, dict[str, Any]] = {}
    for image_path in image_paths:
        image_results[image_path.name] = scan_image(
            image_path,
            product_output_folder / f"highlighted_{image_path.name}",
        )

    fields = aggregate_fields(image_results)
    result = {
        "product": product_name,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "images": image_results,
        "fields": fields,
        "missing_fields": sorted(set(FIELD_PATTERNS) - set(fields)),
    }
    json_path = product_output_folder / "data.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(f"  saved: {json_path}")
    return result


def main() -> None:
    print("=" * 60)
    print("PACKSIGHT SCANNER")
    print("=" * 60)
    print(f"Tesseract: {configure_tesseract()}")
    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_FOLDER}")

    products = sorted(
        (path for path in INPUT_FOLDER.iterdir() if path.is_dir()),
        key=natural_key,
    )
    if not products:
        print("No product folders found.")
        return

    for product_folder in products:
        print(f"\n{'#' * 60}\nPRODUCT: {product_folder.name}\n{'#' * 60}")
        scan_product(product_folder.name, product_folder)
    print(f"\n{'=' * 60}\nSCAN COMPLETE\n{'=' * 60}")


if __name__ == "__main__":
    main()
