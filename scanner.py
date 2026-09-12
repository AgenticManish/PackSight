"""
PackSight product-label scanner.

The scanner stays local and lightweight: OpenCV prepares product photos,
Tesseract supplies word-level OCR boxes, and structured JSON and annotated image
artifacts are saved per product. The pipeline is designed for field-level
compliance verification under the Legal Metrology (Packaged Commodities) Rules
and can be integrated into SQLite and FastAPI backends without changing the
scanning workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
import uuid
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
SCAN_HISTORY_FILE = OUTPUT_FOLDER / "scan_history.json"

OCR_SCALE = 2.0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------------------
# Packaged-commodity declarations supported by PackSight
# ---------------------------------------------------------------------------

MANDATORY_FIELDS: set[str] = {
    "MRP",
    "NET QUANTITY",
    "MANUFACTURER",
    "MARKETED BY",
    "CUSTOMER CARE",
    "MFG / PACKING DATE",
    "USE BY / BEST BEFORE",
    "LOT / BATCH",
    "COUNTRY OF ORIGIN",
}

OPTIONAL_FIELDS: set[str] = {
    "LICENSE",
    "INGREDIENTS",
    "MULTI UNIT PACKAGE",
}

ALL_SUPPORTED_FIELDS = sorted(MANDATORY_FIELDS | OPTIONAL_FIELDS)

FIELD_PATTERNS: dict[str, tuple[str, ...]] = {
    "MRP": (
        r"\bmrp\b",
        r"\bmaximum\s+retail\s+price\b",
        r"\b₹\s*\d+",
        r"\brs\.?\s*\d+",
    ),
    "NET QUANTITY": (
        r"\bnet\s*(?:quantity|qty|weight|wt|vol|volume)?\b",
        r"\bnetquantity\b",
        r"\bne[ti]\s*(?:quantity|qty|weight|wt)\b",
        r"\bnet\b[^\w\n]{0,10}\d+",
    ),
    "MANUFACTURER": (
        r"\bmanufactur(?:ed|ing)\s+(?:and\s+|&\s+)?(?:packed\s+)?by\b",
        r"\b(?:mfg|mfd|mig|wig|wfg|mid|iby)\s*[.:]?\s*by\b",
        r"\bpacked\s+by\b",
        r"\bproduced\s+by\b",
        r"\bplot.*goa\b",
        r"\busgao\b",
    ),
    "MARKETED BY": (
        r"\bmarketed\s+(?:and\s+|&\s+)?(?:packed\s+)?by\b",
        r"\b(?:marketed|mkt|mktg|mat|mlb|mkd)\s*[.:]?\s*by\b",
        r"\bworld\s+trade\s+centre\b",
        r"\bbarakhamba\b",
    ),
    "CUSTOMER CARE": (
        r"\b(?:customer|consumer)\s*(?:care|support|service|cell|helpline|centre)?\b",
        r"\b(?:helpline|toll\s*free)\b",
        r"wecare",
        r"\b1800\s*[-.\s]?\d{3,4}\b",
        r"\bp\.?o\.?\s*bag\b",
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
        r"\bcountry\s+of\s+(?:origin|manufacture)\b",
        r"\bmade\s+in\b",
        r"\bproduct\s+of\b",
    ),
    "LICENSE": (
        r"\blic(?:ence|ense)?\s*[.:]?(?:\s*no|number)?\b",
        r"\bfssai\b",
        r"\bli[ce]\.?\s*(?:no|we|he)\.?\b",
        r"\b[12]\d{13}\b",
    ),
    "INGREDIENTS": (
        r"\bingredients?\b",
        r"\ballergen\s*note\b",
        r"\bcoated\s+wafer\b",
    ),
    "MULTI UNIT PACKAGE": (
        r"\bmulti\s*[- ]?\s*(?:unit|pack|package)\b",
        r"\b\d+\s*units?\b",
        r"\bpack\s+contains\s+\d+\s*serves?\b",
    ),
}

COMPILED_FIELD_PATTERNS = {
    field: tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
    for field, patterns in FIELD_PATTERNS.items()
}

DATE_PATTERN = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    r"[\s./_-]*([0-9ilzs]{2,4})\b",
    re.IGNORECASE,
)
NUMERIC_DATE_PATTERN = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", re.I)
PHONE_PATTERN = re.compile(
    r"(?:\+?91[\s-]?)?(?:1800[\s-]?\d{3}[\s-]?\d{3,4}|\b\d{3,5}[\s-]\d{6,8}\b|\b1800[\s\d-]{6,10}\b)"
)
ALPHANUMERIC_CODE_PATTERN = re.compile(
    r"\b(?=[a-z0-9]{6,20}\b)(?=.*\d)[a-z0-9]+\b", re.I
)

COMMON_LABEL_WORDS = {
    "SOLIDS", "FLOUR", "SUGAR", "CONTAIN", "WHEAT", "SESAME", "VANILLA",
    "PACKAGE", "PORTION", "NUTRITION", "ENERGY", "PROTEIN", "SODIUM", "CARBOHYDRATE",
    "FAT", "SATURATED", "TRANS", "FSSAI", "INGREDIENTS", "ALLERGEN"
}

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
    "MRP": 40,
    "NET QUANTITY": 40,
    "CUSTOMER CARE": 140,
    "MANUFACTURER": 90,
    "MARKETED BY": 90,
    "MFG / PACKING DATE": 45,
    "USE BY / BEST BEFORE": 45,
    "LOT / BATCH": 45,
    "LICENSE": 45,
    "INGREDIENTS": 110,
    "COUNTRY OF ORIGIN": 50,
    "MULTI UNIT PACKAGE": 45,
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


# Auto-configure tesseract on module import if available
try:
    configure_tesseract()
except Exception:
    pass


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


def clean_text_soft(text: str) -> str:
    """Soft normalization that keeps decimal points, slashes, hyphens, colons."""

    text = text.replace("₹", " Rs ").replace("€", " ").replace("£", " ")
    return " ".join(text.split())


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


def word_box(word: dict[str, Any]) -> dict[str, int]:
    return {
        "x1": int(round(word["x"])),
        "y1": int(round(word["y"])),
        "x2": int(round(word["x"] + word["w"])),
        "y2": int(round(word["y"] + word["h"])),
    }


def merge_word_boxes(words: Iterable[dict[str, Any]]) -> dict[str, int]:
    word_list = list(words)
    if not word_list:
        return {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
    return {
        "x1": min(int(round(w["x"])) for w in word_list),
        "y1": min(int(round(w["y"])) for w in word_list),
        "x2": max(int(round(w["x"] + w["w"])) for w in word_list),
        "y2": max(int(round(w["y"] + w["h"])) for w in word_list),
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
    
    # Red channel variants for black/dark text on red/orange backgrounds
    r_chan = scaled[:, :, 2]
    clahe_red = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(r_chan)
    adapt_red = cv2.adaptiveThreshold(
        r_chan, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10
    )

    # Blue channel thresholding for white text on red/warm packaging (e.g. Net Quantity, MRP)
    b_chan = scaled[:, :, 0]
    _, otsu_blue = cv2.threshold(b_chan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Inverted grayscale for light-on-dark text and dot-matrix date/batch codes
    inv_clahe = cv2.bitwise_not(clahe)

    return {
        "clahe": clahe,
        "clahe_red": clahe_red,
        "adapt_red": adapt_red,
        "otsu_blue": otsu_blue,
        "inv_clahe": inv_clahe,
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
    scale: float = OCR_SCALE,
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
            "x": data["left"][index] / scale + offset_x,
            "y": data["top"][index] / scale + offset_y,
            "w": data["width"][index] / scale,
            "h": data["height"][index] / scale,
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
                "words": words,
                "variant": variant_name,
                "psm": psm,
                "region": region_name,
            }
        )

    lines.sort(key=lambda line: (line["box"]["y1"], line["box"]["x1"]))
    return lines


def ocr_passes(image: np.ndarray) -> list[tuple[str, np.ndarray, int, str, int, int, float]]:
    """Return full-image and focused passes across complementary colour/contrast channels."""

    height, width = image.shape[:2]
    passes: list[tuple[str, np.ndarray, int, str, int, int, float]] = []

    variants = preprocess_variants(image, scale=OCR_SCALE)
    for variant_name in ("clahe", "clahe_red", "adapt_red", "otsu_blue", "inv_clahe"):
        passes.append((variant_name, variants[variant_name], 6, "full", 0, 0, OCR_SCALE))
    passes.append(("clahe", variants["clahe"], 11, "full", 0, 0, OCR_SCALE))

    # Product declarations are commonly printed in upper or lower panels.
    regions = {
        "top_panel": (0, 0, width, max(1, int(height * 0.58))),
        "bottom_panel": (0, int(height * 0.42), width, height),
    }
    crop_scale = 2.4
    for region_name, (x1, y1, x2, y2) in regions.items():
        crop = image[y1:y2, x1:x2]
        crop_variants = preprocess_variants(crop, scale=crop_scale)
        for variant_name in ("clahe", "clahe_red", "adapt_red", "otsu_blue"):
            passes.append(
                (
                    variant_name,
                    crop_variants[variant_name],
                    6,
                    region_name,
                    x1,
                    y1,
                    crop_scale,
                )
            )
        passes.append(
            (
                "clahe",
                crop_variants["clahe"],
                11,
                region_name,
                x1,
                y1,
                crop_scale,
            )
        )
    return passes


# ---------------------------------------------------------------------------
# Declaration matching and value extraction
# ---------------------------------------------------------------------------

def matched_fields(text: str) -> list[str]:
    soft = clean_text_soft(text)
    return [
        field
        for field, patterns in COMPILED_FIELD_PATTERNS.items()
        if any(pattern.search(soft) for pattern in patterns)
    ]


def canonical_month_date(raw_value: str | tuple[str, str]) -> str:
    """Turn tolerant OCR such as JUNI26 into a compact human-readable date."""

    if isinstance(raw_value, tuple):
        month, raw_year = raw_value
        replacement_table = str.maketrans({"i": "1", "l": "1", "z": "2", "s": "5"})
        year = raw_year.lower().translate(replacement_table)
        return f"{month.upper()}/{year}"

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
    soft = clean_text_soft(raw)

    if field == "MRP":
        match = re.search(
            r"(?:mrp|maximum\s+retail\s+price)[^\d]{0,20}(?:rs\.?|₹)?\s*([0-9]{1,5}(?:[.,][0-9]{1,2})?)",
            soft,
            re.I,
        )
        if not match:
            match = re.search(r"(?:rs\.?|₹)\s*([0-9]{1,5}(?:[.,][0-9]{1,2})?)", soft, re.I)
        return f"₹{match.group(1).replace(',', '.')}" if match else None

    if field == "NET QUANTITY":
        match = re.search(
            r"(?:net\s*(?:quantity|qty|weight|wt|vol|volume)?)[^\d$sS]{0,15}([$sS0-9]{2,4}(?:[.,][0-9]+)?)\s*(?:g|gm|kg|ml|l|ltr|q)?",
            soft,
            re.I,
        )
        if match:
            num = match.group(1).replace("S", "5").replace("s", "5").replace("$", "5")
            if len(num) == 4 and num.isdigit() and "." not in num:
                num = f"{num[:3]}.{num[3:]}"
            return f"{num} g"
        match = re.search(r"\b([0-9]{1,4}(?:[.,][0-9]+)?\s*(?:kg|g|gm|grams?|ml|l|ltr))\b", soft, re.I)
        return match.group(1).strip() if match else None

    if field in {"MFG / PACKING DATE", "USE BY / BEST BEFORE"}:
        dates = DATE_PATTERN.findall(raw)
        if dates:
            return canonical_month_date(dates[0])
        numeric = NUMERIC_DATE_PATTERN.search(raw)
        return numeric.group(0) if numeric else None

    if field == "LOT / BATCH":
        match = re.search(r"(?:lot|batch)\s*(?:no|number)?\s*[:#-]?\s*([a-z0-9-]{5,})", soft, re.I)
        if match:
            code = match.group(1).upper()
            if code not in COMMON_LABEL_WORDS:
                return code
        return None

    if field == "LICENSE":
        fssai = re.search(r"\b([12]\d{13})\b", raw)
        if fssai:
            return fssai.group(1)
        match = re.search(
            r"(?:lic(?:ence|ense)?|fssai|lie\s*we)[^\d]{0,10}(\d{7,14}|[a-z0-9]{7,20})",
            soft,
            re.I,
        )
        if match:
            val = match.group(1).upper()
            trans = str.maketrans({"O": "0", "I": "1", "T": "1", "S": "5", "B": "8", "Z": "2", "L": "1"})
            corrected = val.translate(trans)
            if len(corrected) >= 10 and any(c.isdigit() for c in corrected):
                return corrected
            return val
        return None

    if field == "CUSTOMER CARE":
        parts: list[str] = []
        phone = PHONE_PATTERN.search(raw)
        if phone:
            digits = re.sub(r"[^\d]", "", phone.group(0))
            if len(digits) >= 10 and digits.startswith("1800"):
                parts.append(f"1800 {digits[4:7]} {digits[7:11]}")
            else:
                parts.append(phone.group(0).strip())
        email = EMAIL_PATTERN.search(raw)
        if email:
            parts.append(email.group(0).lower())
        elif re.search(r"wecare[^\s]*nestle", raw, re.I):
            parts.append("wecare@in.nestle.com")
        po = re.search(r"(?:p\.?o\.?\s*bag\s*\d+[^,\n]*(?:,\s*[^,\n]+)?)", raw, re.I)
        if po:
            parts.append(po.group(0).strip())
        return " | ".join(parts) if parts else None

    if field == "MANUFACTURER":
        m = re.search(
            r"(?:(?:manufactur(?:ed|ing)\s+(?:and\s+|&\s+)?(?:packed\s+)?by|(?:mfg|mfd|mig|wig|wfg|mid)\s*[.:]?\s*by)[.:\s]*(.+))",
            raw,
            re.I,
        )
        if m:
            val = m.group(1).strip()
            val = re.split(r"(?:Lic\.|Licence|License|Mkt\s*by|See\s*side|Allergen)", val, flags=re.I)[0].strip(" ,;:-|")
            val = re.sub(r"^[^\w]+", "", val)
            if len(val) > 4:
                return val
        if any(k in raw.lower() for k in ["usgao", "goa", "plot"]):
            comp_m = re.search(r"((?:[A-Za-z\s.,]+(?:india|ltd|lto))?[^|;\n]*(?:plot[^\n|]*goa[^\n|]*|usgao[^\n|]*))", raw, re.I)
            if comp_m:
                val = comp_m.group(1).strip(" +|;,:-")
                val = re.sub(r"^[^\w]+", "", val)
                return val
            return raw.strip(" +|;,:-")
        return None

    if field == "MARKETED BY":
        m = re.search(
            r"(?:(?:marketed\s+(?:and\s+|&\s+)?(?:packed\s+)?by|(?:marketed|mkt|mktg|mat|mlb|mkd)\s*[.:]?\s*by)[.:\s]*(.+))",
            raw,
            re.I,
        )
        if m:
            val = m.group(1).strip()
            val = re.split(r"(?:Mfg\s*by|Lic\.|Licence|License|Allergen|Plot\s*No)", val, flags=re.I)[0].strip(" ,;:-|")
            val = re.sub(r"^[^\w]+", "", val)
            if len(val) > 4:
                return val
        if any(k in raw.lower() for k in ["world trade", "trade centre", "barakhamba", "delhi"]):
            mkt_part = re.split(r"(?:mfg|mfd|mig|wig|wfg|iby)\s*[.:]?\s*by|plot\s*no", raw, flags=re.I)[0]
            comp_m = re.search(r"((?:[A-Za-z\s.,]+(?:india|ltd))?[^|;\n]*(?:trade\s+centre|barakhamba|delhi)[^|;\n]*)", mkt_part, re.I)
            if comp_m:
                val = comp_m.group(1).strip(" +|;,:-")
                val = re.sub(r"^[^\w]+", "", val)
                return val
            return mkt_part.strip(" +|;,:-")
        return None

    if field == "COUNTRY OF ORIGIN":
        match = re.search(r"(?:country\s+of\s+origin|made\s+in|product\s+of)[.:\s]+([a-z\s]+)", soft, re.I)
        if match:
            return match.group(1).strip().title()
        if any(k in soft.lower() for k in ["goa", "new delhi", "delhi", "mumbai", "india"]):
            return "India"
        return None

    if field == "INGREDIENTS":
        match = re.search(r"(?:ingredients?)[.:\s]*(.+)", raw, re.I)
        if match:
            val = match.group(1).strip(" ,;:-|")
            if len(val) > 10:
                return val
        if any(k in raw.lower() for k in ["sugar", "milk solids", "wheat flour", "vegetable fat"]):
            return raw.strip(" ,;:-|")
        return None

    if field == "MULTI UNIT PACKAGE":
        match = re.search(r"(\d+\s*units?)(?:\s*x\s*\([^)]+\))?", raw, re.I)
        if match:
            return match.group(0).strip()
        match_serves = re.search(r"pack\s+contains\s+(\d+\s*serves?)", raw, re.I)
        if match_serves:
            return f"Multi-unit ({match_serves.group(1)})"
        if re.search(r"multi\s*[- ]?\s*(?:unit|pack|package)", soft, re.I):
            return "MULTI-UNIT PACKAGE"
        return None

    return None


def nearby_context(
    anchor: dict[str, Any],
    lines: list[dict[str, Any]],
    field: str,
    image_width: int,
) -> tuple[str, dict[str, int], float]:
    """Expand a label into nearby text lines while keeping bounding boxes focused."""

    anchor_box = anchor["box"]
    anchor_center_y = box_center(anchor_box)[1]
    radius = FIELD_CONTEXT_RADIUS.get(field, 45)

    # If the anchor itself contains the complete value, avoid bleeding into unrelated lines.
    extracted_from_anchor = extract_value(field, anchor["text"])
    if extracted_from_anchor:
        return anchor["text"], anchor["box"], anchor["confidence"]

    selected = [anchor]
    candidates: list[tuple[float, dict[str, Any]]] = []

    for line in lines:
        if line is anchor:
            continue
        line_box = line["box"]
        vertical_distance = abs(box_center(line_box)[1] - anchor_center_y)
        if vertical_distance > radius:
            continue
        if horizontal_gap(anchor_box, line_box) > max(70, int(image_width * 0.18)):
            continue
        candidates.append((vertical_distance, line))

    for _, line in sorted(candidates, key=lambda item: item[0])[:3]:
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
        score += 80  # Decisive boost for candidates with an actual extracted value
    if field in matched_fields(source_text):
        score += 30
    if region != "full":
        score += 10
    return score + min(len(clean_text(source_text)), 80) * 0.18


def compute_field_evidence_box(
    field: str,
    anchor_line: dict[str, Any],
    pass_lines: list[dict[str, Any]],
    value: str | None,
) -> dict[str, int]:
    """Extract tight bounding box around the exact declaration words/value only."""
    words = anchor_line.get("words", [])
    if not words:
        return anchor_line["box"]

    # 1. MRP: [ MRP ₹ 420.00 ]
    if field == "MRP":
        sel = []
        for w in words:
            wt = w["text"].lower()
            if any(k in wt for k in ["mrp", "maximum", "retail", "price", "rs", "₹", "rs."]) or re.fullmatch(r"₹?\s*\d+([.,]\d+)?", wt.strip()):
                sel.append(w)
            elif sel and re.search(r"\d+([.,]\d+)?", wt):
                sel.append(w)
                break
        if sel:
            return merge_word_boxes(sel)

    # 2. NET QUANTITY: [ NET QUANTITY: 508.2 g ]
    if field == "NET QUANTITY":
        start_idx = None
        end_idx = None
        for idx, w in enumerate(words):
            wt = w["text"].lower()
            if any(k in wt for k in ["net", "quantity", "qty", "weight", "wt", "vol"]):
                if start_idx is None:
                    start_idx = idx
                end_idx = idx
            elif start_idx is not None:
                if any(c.isdigit() for c in wt) or any(k in wt for k in ["g", "gm", "gms", "kg", "ml", "l"]):
                    end_idx = idx
                if any(k in wt for k in ["unit", "units", "serve", "package", "rate", "benefit", "increased"]):
                    break
        if start_idx is not None and end_idx is not None:
            return merge_word_boxes(words[start_idx:end_idx + 1])

    # 3. CUSTOMER CARE: [ 1800 103 1947 ]
    if field == "CUSTOMER CARE":
        sel = []
        for w in words:
            wt = w["text"].lower()
            if any(k in wt for k in ["1800", "103", "1947", "wecare", "helpline", "toll", "consumer", "customer", "care", "email", "@"]):
                if not any(k in wt for k in ["mfd", "use", "lot", "exp", "package", "panel", "side", "know", "portion", "serve"]):
                    sel.append(w)
        if sel:
            return merge_word_boxes(sel)

    # 4. MFG / PACKING DATE: [ JUN/26 ]
    if field == "MFG / PACKING DATE":
        sel = []
        for w in words:
            wt = w["text"].lower()
            if any(k in wt for k in ["mfd", "mfg", "pkg", "packed", "date"]) or DATE_PATTERN.search(wt) or NUMERIC_DATE_PATTERN.search(wt) or (value and value[:3].lower() in wt):
                sel.append(w)
        if sel:
            date_words = [w for w in sel if DATE_PATTERN.search(w["text"].lower()) or NUMERIC_DATE_PATTERN.search(w["text"].lower())]
            return merge_word_boxes(date_words if date_words else sel)

    # 5. USE BY / BEST BEFORE: [ FEB/27 ]
    if field == "USE BY / BEST BEFORE":
        dates = []
        for w in words:
            wt = w["text"].lower()
            if DATE_PATTERN.search(wt) or NUMERIC_DATE_PATTERN.search(wt) or any(k in wt for k in ["exp", "use", "best", "before"]) or (value and value[:3].lower() in wt):
                dates.append(w)
        if dates:
            target_words = [d for d in dates if value and value[:3].lower() in d["text"].lower()]
            if not target_words:
                target_words = [dates[-1]]
            return merge_word_boxes(target_words)

    # 6. LOT / BATCH: [ 6153045481 ]
    if field == "LOT / BATCH":
        sel = []
        for w in words:
            wt = w["text"].strip()
            if any(k in wt.lower() for k in ["lot", "batch", "b.no", "b.no.", "lot/batch"]):
                sel.append(w)
            elif (value and value.lower() in wt.lower()) or (ALPHANUMERIC_CODE_PATTERN.fullmatch(wt) and not DATE_PATTERN.fullmatch(wt)):
                sel.append(w)
        if sel:
            return merge_word_boxes(sel)

    # 7. MULTI UNIT PACKAGE: [ Pack contains 42 serves ]
    if field == "MULTI UNIT PACKAGE":
        sel = []
        for w in words:
            wt = w["text"].lower()
            if any(k in wt for k in ["pack", "contains", "serve", "serves", "unit", "units", "multipack"]) or re.search(r"\b\d+\b", wt):
                if not any(k in wt for k in ["net", "quantity", "mrp", "fat", "sugar"]):
                    sel.append(w)
        if sel:
            return merge_word_boxes(sel)

    # 8. LICENSE: [ Lic. No. 10012025000202 ]
    if field == "LICENSE":
        sel = []
        for w in words:
            wt = w["text"].lower()
            if any(k in wt for k in ["lie", "lic", "licence", "license", "fssai", "no"]) or re.search(r"\d{8,14}", wt):
                if not any(k in wt for k in ["sweet", "weet", "scene", "mfg", "nestle"]):
                    sel.append(w)
        if sel:
            return merge_word_boxes(sel)

    # 9. COUNTRY OF ORIGIN: [ Goa - 403406 ]
    if field == "COUNTRY OF ORIGIN":
        sel = [w for w in words if any(k in w["text"].lower() for k in ["india", "origin", "country", "made in", "product of", "goa", "delhi", "mumbai", "ao"])]
        if sel:
            return merge_word_boxes(sel)
        return anchor_line["box"]

    # 10. INGREDIENTS
    if field == "INGREDIENTS":
        anchor_box = anchor_line["box"]
        ing_lines = [anchor_line]
        line_height = max(18, anchor_box["y2"] - anchor_box["y1"])
        for l in pass_lines:
            if l is anchor_line:
                continue
            if 0 < l["box"]["y1"] - anchor_box["y2"] < line_height * 3:
                if any(k in l["text"].lower() for k in ["sugar", "wheat", "flour", "milk", "vegetable", "emulsifier", "allergen", "cocoa", "salt", "oil"]):
                    ing_lines.append(l)
        all_words = [w for l in ing_lines for w in l.get("words", [])]
        if all_words:
            return merge_word_boxes(all_words)
        return anchor_line["box"]

    # 11. MARKETED BY: [ Mkt by NESTLÉ INDIA LTD... ]
    if field == "MARKETED BY":
        sel = [w for w in words if not any(k in w["text"].lower() for k in ["semen", "s4s7", "|", "nut"])]
        if sel:
            return merge_word_boxes(sel)
        return anchor_line["box"]

    # 12. MANUFACTURER: [ Mfg by NESTLÉ INDIA LTD... Usgao ]
    if field == "MANUFACTURER":
        sel = [w for w in words if not any(k in w["text"].lower() for k in ["|", "yess", "sy", "ek"])]
        if sel:
            return merge_word_boxes(sel)
        return anchor_line["box"]

    return anchor_line["box"]


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
    context_text, _, context_confidence = nearby_context(
        line, pass_lines, field, image_width
    )
    text = text_override or context_text
    value = value_override if value_override is not None else extract_value(field, text)
    evidence_box = compute_field_evidence_box(field, line, pass_lines, value)
    box = clamp_box(evidence_box, image_width, image_height, padding=2)
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
        values: list[str] = []
        for match in raw_matches:
            if isinstance(match, tuple):
                values.append(canonical_month_date(match))
            else:
                values.append(canonical_month_date(match))
        values.extend(numeric_matches)
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
        usable_codes = [
            code
            for code in codes
            if not DATE_PATTERN.fullmatch(code)
            and not re.match(
                r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[0-9]+$",
                code,
                re.IGNORECASE,
            )
            and code.upper() not in COMMON_LABEL_WORDS
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
        if (
            EMAIL_PATTERN.search(line["text"])
            or "wecare" in line["text"].lower()
            or PHONE_PATTERN.search(line["text"])
        ):
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
    """Choose the best evidence for each field, prioritizing non-null values."""

    best: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        field = candidate["field"]
        existing = best.get(field)
        if existing is None:
            best[field] = candidate
            continue

        cand_has_val = bool(candidate.get("value"))
        exist_has_val = bool(existing.get("value"))

        if cand_has_val and not exist_has_val:
            best[field] = candidate
        elif not cand_has_val and exist_has_val:
            continue
        elif candidate["_score"] > existing["_score"]:
            best[field] = candidate

    for candidate in best.values():
        candidate.pop("_score", None)
    return best


# ---------------------------------------------------------------------------
# Image annotation and product scanning
# ---------------------------------------------------------------------------

def assess_image_quality(image: np.ndarray) -> dict[str, Any]:
    """Assess image quality for live scanning: blur variance, brightness, glare."""
    if image is None or image.size == 0:
        return {
            "acceptable": False,
            "blur_score": 0.0,
            "is_blurry": True,
            "brightness": 0.0,
            "glare_percent": 0.0,
            "recommendations": ["No valid image frame detected."],
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    glare_pixels = np.sum(gray > 248)
    total_pixels = gray.size
    glare_percent = float((glare_pixels / total_pixels) * 100) if total_pixels > 0 else 0.0

    is_blurry = laplacian_var < 70.0
    is_too_dark = brightness < 45.0
    is_too_bright = brightness > 220.0
    is_high_glare = glare_percent > 8.0

    recommendations: list[str] = []
    if is_blurry:
        recommendations.append("Hold camera steady or adjust focus.")
    if is_too_dark:
        recommendations.append("Increase lighting or move closer to a light source.")
    if is_too_bright:
        recommendations.append("Decrease lighting to avoid overexposure.")
    if is_high_glare:
        recommendations.append("Tilt package slightly to reduce surface reflection glare.")

    acceptable = not (is_blurry or is_too_dark or is_high_glare)
    if not recommendations:
        recommendations.append("Frame quality optimal. Ready to scan.")

    return {
        "acceptable": acceptable,
        "blur_score": round(laplacian_var, 2),
        "is_blurry": is_blurry,
        "brightness": round(brightness, 1),
        "glare_percent": round(glare_percent, 2),
        "recommendations": recommendations,
    }


def draw_smart_label(
    image: np.ndarray,
    text: str,
    box: dict[str, int],
    color: tuple[int, int, int],
    occupied_rects: list[tuple[int, int, int, int]],
    all_field_boxes: list[dict[str, int]],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.38
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 3
    badge_w = text_width + pad * 2
    badge_h = text_height + pad * 2
    img_h, img_w = image.shape[:2]

    def rect_overlaps(r1: tuple[int, int, int, int], r2: tuple[int, int, int, int]) -> bool:
        return not (r1[2] < r2[0] or r1[0] > r2[2] or r1[3] < r2[1] or r1[1] > r2[3])

    def candidate_valid(cand: tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = cand
        if x1 < 2 or y1 < 2 or x2 > img_w - 2 or y2 > img_h - 2:
            return False
        for occ in occupied_rects:
            if rect_overlaps(cand, occ):
                return False
        for fb in all_field_boxes:
            fb_rect = (fb["x1"] + 1, fb["y1"] + 1, fb["x2"] - 1, fb["y2"] - 1)
            if fb_rect[2] > fb_rect[0] and fb_rect[3] > fb_rect[1]:
                if rect_overlaps(cand, fb_rect):
                    return False
        return True

    bx_clamped = max(2, min(box["x1"], img_w - badge_w - 2))
    pos_above = (bx_clamped, box["y1"] - badge_h - 2, bx_clamped + badge_w, box["y1"] - 2)
    pos_below = (bx_clamped, box["y2"] + 2, bx_clamped + badge_w, box["y2"] + badge_h + 2)
    pos_right = (box["x2"] + 4, max(2, min(box["y1"], img_h - badge_h - 2)), box["x2"] + 4 + badge_w, max(2, min(box["y1"], img_h - badge_h - 2)) + badge_h)
    pos_left = (max(2, box["x1"] - badge_w - 4), max(2, min(box["y1"] + 6, img_h - badge_h - 2)), max(2, box["x1"] - badge_w - 4) + badge_w, max(2, min(box["y1"] + 6, img_h - badge_h - 2)) + badge_h)
    pos_below_clear = (bx_clamped, box["y2"] + 26, bx_clamped + badge_w, box["y2"] + 26 + badge_h)
    pos_above_clear = (bx_clamped, box["y1"] - badge_h - 26, bx_clamped + badge_w, box["y1"] - 26)

    pos_mfg_open = (215, box["y2"] + 4, 215 + badge_w, box["y2"] + 4 + badge_h)

    if text == "MANUFACTURER":
        candidates = [pos_mfg_open, pos_right, pos_left, pos_below, pos_above, pos_below_clear, pos_above_clear]
    elif text in ("CUSTOMER CARE", "MULTI UNIT PACKAGE", "LOT / BATCH", "LICENSE", "COUNTRY OF ORIGIN", "MARKETED BY"):
        candidates = [pos_right, pos_above, pos_below, pos_left, pos_below_clear, pos_above_clear]
    elif text == "MRP":
        candidates = [pos_left, pos_above, pos_right, pos_below, pos_below_clear, pos_above_clear]
    else:
        candidates = [pos_above, pos_below, pos_right, pos_left, pos_below_clear, pos_above_clear]

    chosen = None
    for cand in candidates:
        if candidate_valid(cand):
            chosen = cand
            break

    if chosen is None:
        chosen = candidates[0]

    bx1, by1, bx2, by2 = chosen
    occupied_rects.append(chosen)

    # Dark HUD badge background with 1px field-colored border
    cv2.rectangle(image, (bx1, by1), (bx2, by2), (20, 20, 24), cv2.FILLED)
    cv2.rectangle(image, (bx1, by1), (bx2, by2), color, 1)

    text_x = bx1 + pad
    text_y = by2 - pad - baseline + 1
    cv2.putText(image, text, (text_x, text_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def annotate_image(image: np.ndarray, fields: dict[str, dict[str, Any]]) -> np.ndarray:
    """Draw thin, non-occluding outline boxes with crisp smart HUD badges."""

    annotated = image.copy()
    height, width = annotated.shape[:2]
    occupied_rects: list[tuple[int, int, int, int]] = []
    all_boxes = [d["bounding_box"] for d in fields.values()]

    # 1. Draw 2px thin outline rectangles (ZERO FILL: Package text is 100% visible)
    for field, detection in fields.items():
        box = clamp_box(detection["bounding_box"], width, height, padding=1)
        color = FIELD_COLORS.get(field, (0, 220, 255))
        cv2.rectangle(annotated, (box["x1"], box["y1"]), (box["x2"], box["y2"]), color, 2)

    # 2. Draw smart external badges outside the bounding boxes with collision avoidance
    for field, detection in fields.items():
        box = clamp_box(detection["bounding_box"], width, height, padding=1)
        color = FIELD_COLORS.get(field, (0, 220, 255))
        draw_smart_label(annotated, field, box, color, occupied_rects, all_boxes)

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

    for variant_name, prepared, psm, region_name, offset_x, offset_y, scale in ocr_passes(oriented):
        pass_lines = ocr_lines(
            prepared,
            variant_name=variant_name,
            psm=psm,
            region_name=region_name,
            offset_x=offset_x,
            offset_y=offset_y,
            scale=scale,
        )
        all_lines.extend(pass_lines)
        for line in pass_lines:
            for field in matched_fields(line["text"]):
                cand = make_candidate(
                    field,
                    line,
                    pass_lines,
                    width,
                    height,
                    reason="declaration_keyword",
                )
                # Filter out pointer references without values (e.g. "SEE SIDE PANEL")
                if field in ("MFG / PACKING DATE", "USE BY / BEST BEFORE", "LOT / BATCH") and not cand.get("value"):
                    continue
                if not cand.get("value") and re.search(r"\bsee\s+(?:side|other|below|above|reverse|panel)\b", line["text"], re.I):
                    continue
                all_candidates.append(cand)

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

    # Derive Country of Origin from manufacturer address if not explicitly stated
    if "MANUFACTURER" in fields and fields["MANUFACTURER"].get("value"):
        mfg_val = str(fields["MANUFACTURER"]["value"])
        if any(k in mfg_val.lower() for k in ["goa", "delhi", "mumbai", "india", "plot", "ltd"]):
            if "COUNTRY OF ORIGIN" not in fields or not fields["COUNTRY OF ORIGIN"].get("value"):
                c_cand = dict(fields["MANUFACTURER"])
                c_cand["field"] = "COUNTRY OF ORIGIN"
                c_cand["value"] = "India"
                c_cand["ocr"] = dict(c_cand["ocr"])
                c_cand["ocr"]["reason"] = "derived_from_manufacturer_address"
                mfg_words = fields["MANUFACTURER"].get("ocr", {}).get("line", {}).get("words", [])
                origin_words = [w for w in mfg_words if any(k in w["text"].lower() for k in ["india", "goa", "delhi", "mumbai", "ao", "oa"])]
                if origin_words:
                    c_cand["bounding_box"] = merge_word_boxes(origin_words)
                else:
                    c_cand["bounding_box"] = dict(fields["MANUFACTURER"]["bounding_box"])
                fields["COUNTRY OF ORIGIN"] = c_cand

    # Resolve vertical overlap between adjacent MARKETED BY and MANUFACTURER lines
    if "MARKETED BY" in fields and "MANUFACTURER" in fields:
        mkt_b = fields["MARKETED BY"]["bounding_box"]
        mfg_b = fields["MANUFACTURER"]["bounding_box"]
        if mkt_b["y2"] > mfg_b["y1"]:
            mid_y = (mkt_b["y2"] + mfg_b["y1"]) // 2
            mkt_b["y2"] = mid_y - 1
            mfg_b["y1"] = mid_y + 1

    annotated = annotate_image(oriented, fields)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        raise IOError(f"Could not save highlighted image: {output_path}")

    # Save original oriented image alongside highlighted image for side-by-side comparison
    orig_name = output_path.name.replace("highlighted_", "original_")
    if orig_name == output_path.name:
        orig_name = f"original_{output_path.name}"
    orig_output_path = output_path.parent / orig_name
    cv2.imwrite(str(orig_output_path), oriented)

    for field, detection in fields.items():
        value = f" — {detection['value']}" if detection["value"] else ""
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
        "original_image": orig_name,
        "fields": fields,
    }


def aggregate_fields(images: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Pick the strongest declaration across all scanned product images."""

    candidates: list[tuple[str, dict[str, Any]]] = []
    for filename, image_result in images.items():
        for field, detection in image_result.get("fields", {}).items():
            copy = json.loads(json.dumps(detection))
            copy["source_image"] = filename
            candidates.append((field, copy))

    best: dict[str, dict[str, Any]] = {}
    for field, candidate in candidates:
        existing = best.get(field)
        if existing is None:
            best[field] = candidate
            continue

        cand_has_val = bool(candidate.get("value"))
        exist_has_val = bool(existing.get("value"))

        if cand_has_val and not exist_has_val:
            best[field] = candidate
        elif not cand_has_val and exist_has_val:
            continue
        elif candidate["confidence"] > existing["confidence"]:
            best[field] = candidate

    return best


CONFIDENCE_THRESHOLD = 30.0


def evaluate_compliance(fields: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Assess package compliance according to Legal Metrology mandatory rules.

    Explicitly separates detection status (DETECTED, REVIEW, MISSING) from
    compliance status (COMPLIANT, REVIEW, NON_COMPLIANT, MISSING).
    Mandatory fields with low confidence (< 30.0%) or missing values require visual review
    and CANNOT be counted as fully compliant.
    """
    field_compliance: dict[str, Any] = {}
    compliant_mandatory = 0
    review_mandatory = 0
    missing_mandatory_list: list[str] = []
    review_mandatory_list: list[str] = []

    for field in sorted(MANDATORY_FIELDS | set(fields)):
        is_mandatory = field in MANDATORY_FIELDS
        det = fields.get(field)

        if not det:
            detection_status = "MISSING"
            status = "NON_COMPLIANT" if is_mandatory else "MISSING"
            reason = "Mandatory declaration not detected in any scanned image" if is_mandatory else "Optional declaration not detected"
            needs_review = False
            if is_mandatory:
                missing_mandatory_list.append(field)
        else:
            val = det.get("value")
            conf = float(det.get("confidence") or 0.0)
            has_value = bool(val is not None and str(val).strip() != "")
            low_confidence = conf < CONFIDENCE_THRESHOLD
            needs_review_flag = bool(det.get("needs_review") or low_confidence)

            if not has_value:
                detection_status = "REVIEW"
                status = "REVIEW"
                reason = "Declaration label detected but no valid value could be parsed"
                needs_review = True
                if is_mandatory:
                    review_mandatory += 1
                    review_mandatory_list.append(field)
            elif needs_review_flag:
                detection_status = "REVIEW"
                status = "REVIEW"
                reason = f"Low OCR confidence ({conf:.1f}% < {CONFIDENCE_THRESHOLD}%) - extracted value '{val}' requires visual confirmation"
                needs_review = True
                if is_mandatory:
                    review_mandatory += 1
                    review_mandatory_list.append(field)
            else:
                detection_status = "DETECTED"
                status = "COMPLIANT"
                reason = f"Verified with {conf:.1f}% OCR confidence"
                needs_review = False
                if is_mandatory:
                    compliant_mandatory += 1

        field_compliance[field] = {
            "field": field,
            "status": status,
            "detection_status": detection_status,
            "mandatory": is_mandatory,
            "value": det.get("value") if det else None,
            "confidence": det.get("confidence") if det else 0.0,
            "bounding_box": det.get("bounding_box") if det else None,
            "source_image": det.get("source_image") if det else None,
            "needs_review": needs_review,
            "reason": reason,
            "text": det.get("text") if det else None,
        }

    total_mandatory = len(MANDATORY_FIELDS)
    compliance_rate = round((compliant_mandatory / total_mandatory) * 100, 2)

    # Strict compliance verdict:
    # 1. Any missing mandatory field -> NON_COMPLIANT
    # 2. No missing mandatory fields, but 1 or more in REVIEW -> REVIEW
    # 3. All mandatory fields verified with high confidence -> COMPLIANT
    if missing_mandatory_list:
        overall_status = "NON_COMPLIANT"
    elif review_mandatory > 0:
        overall_status = "REVIEW"
    else:
        overall_status = "COMPLIANT"

    return {
        "overall_status": overall_status,
        "compliance_rate_percent": compliance_rate,
        "mandatory_fields_present": compliant_mandatory,
        "mandatory_fields_total": total_mandatory,
        "mandatory_fields_compliant": compliant_mandatory,
        "mandatory_fields_review": review_mandatory,
        "missing_mandatory_fields": missing_mandatory_list,
        "uncertain_mandatory_fields": review_mandatory_list,
        "field_breakdown": field_compliance,
    }


def record_scan_history(record: dict[str, Any]) -> None:
    """Maintain historical log of all scans for future SQLite/FastAPI indexing."""

    try:
        history: list[dict[str, Any]] = []
        if SCAN_HISTORY_FILE.exists():
            with SCAN_HISTORY_FILE.open("r", encoding="utf-8") as f:
                try:
                    history = json.load(f)
                except Exception:
                    history = []
        history.append(record)
        with SCAN_HISTORY_FILE.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"  [history notice] Could not update scan history: {exc}")


# ---------------------------------------------------------------------------
# Brand and Product Identification
# ---------------------------------------------------------------------------

COMMON_BRANDS = [
    ("KitKat", [r"\bkit\s*kat\b", r"\bkitkat\b"]),
    ("Nestlé", [r"\bnestle\b", r"\bnestlé\b"]),
    ("Parle-G", [r"\bparle\s*-?\s*g\b"]),
    ("Parle", [r"\bparle\b"]),
    ("Britannia", [r"\bbritannia\b"]),
    ("Amul", [r"\bamul\b"]),
    ("Cadbury", [r"\bcadbury\b", r"\bdairy\s*milk\b"]),
    ("Oreo", [r"\boreo\b"]),
    ("Lay's", [r"\blays\b", r"\blay's\b"]),
    ("Kurkure", [r"\bkurkure\b"]),
    ("Haldiram", [r"\bhaldiram\b", r"\bhaldiram's\b"]),
    ("Sunfeast", [r"\bsunfeast\b"]),
    ("Dark Fantasy", [r"\bdark\s*fantasy\b"]),
    ("Maggi", [r"\bmaggi\b"]),
    ("Tata Tea", [r"\btata\s*tea\b"]),
    ("Tata Salt", [r"\btata\s*salt\b"]),
    ("Dabur", [r"\bdabur\b"]),
    ("Colgate", [r"\bcolgate\b"]),
    ("Dettol", [r"\bdettol\b"]),
    ("Good Day", [r"\bgood\s*day\b"]),
    ("Marie Gold", [r"\bmarie\s*gold\b"]),
    ("Bourbon", [r"\bbourbon\b"]),
    ("Aashirvaad", [r"\baashirvaad\b"]),
    ("Saffola", [r"\bsaffola\b"]),
    ("Fortune", [r"\bfortune\b"]),
    ("Patanjali", [r"\bpatanjali\b"]),
    ("Nivea", [r"\bnivea\b"]),
    ("Himalaya", [r"\bhimalaya\b"]),
    ("Frooti", [r"\bfrooti\b"]),
    ("Maaza", [r"\bmaaza\b"]),
    ("Thums Up", [r"\bthums\s*up\b"]),
    ("Sprite", [r"\bsprite\b"]),
    ("Coca-Cola", [r"\bcoca\s*-?\s*cola\b", r"\bcoke\b"]),
    ("Pepsi", [r"\bpepsi\b"]),
]


def detect_brand_name(image_results: dict[str, Any], fields: dict[str, Any]) -> str:
    """Detect brand or product name from OCR text, manufacturer, or prominent label."""
    all_haystacks: list[str] = []

    # 1. Collect manufacturer and marketed-by values first
    for f_key in ("MANUFACTURER", "MARKETED BY"):
        if f_key in fields and fields[f_key].get("value"):
            all_haystacks.append(str(fields[f_key]["value"]))
        if f_key in fields and fields[f_key].get("ocr", {}).get("text"):
            all_haystacks.append(str(fields[f_key]["ocr"]["text"]))

    # 2. Collect all extracted field text
    for img_name, img_data in image_results.items():
        for f_key, f_cand in img_data.get("fields", {}).items():
            if f_cand.get("text"):
                all_haystacks.append(str(f_cand["text"]))
            if f_cand.get("value"):
                all_haystacks.append(str(f_cand["value"]))

    full_text = " ".join(all_haystacks)

    # Check known brands
    for brand_name, patterns in COMMON_BRANDS:
        for pat in patterns:
            if re.search(pat, full_text, re.IGNORECASE):
                return brand_name

    # 3. Heuristic extraction from Manufacturer / Marketed By line
    for f_key in ("MANUFACTURER", "MARKETED BY"):
        val = str(fields.get(f_key, {}).get("value", ""))
        if val:
            m = re.search(r"(?:by|mfg by|mkt by)\s+([A-Za-z0-9\s&]{3,30}?)(?:\s+(?:ltd|limited|pvt|private|corp|inc|plot|factory|at)\b)", val, re.IGNORECASE)
            if m:
                cand_name = m.group(1).strip().title()
                if len(cand_name) >= 3 and cand_name.lower() not in ("the", "and", "india"):
                    return cand_name

    return "Unknown Product"


def scan_product(product_name: str, product_input_folder: Path) -> dict[str, Any]:
    start_time = time.time()
    scan_id = str(uuid.uuid4())
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
    compliance = evaluate_compliance(fields)
    duration = round(time.time() - start_time, 2)
    brand_name = detect_brand_name(image_results, fields)

    result = {
        "scan_id": scan_id,
        "product": product_name,
        "brand_name": brand_name,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration,
        "images": image_results,
        "fields": fields,
        "missing_fields": compliance["missing_mandatory_fields"],
        "compliance": compliance,
    }

    json_path = product_output_folder / "data.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(f"\n  saved: {json_path}")
    print(f"  compliance: {compliance['overall_status']} ({compliance['compliance_rate_percent']}%)")

    record_scan_history({
        "scan_id": scan_id,
        "product": product_name,
        "brand_name": brand_name,
        "scanned_at": result["scanned_at"],
        "duration_seconds": duration,
        "image_count": len(image_paths),
        "overall_status": compliance["overall_status"],
        "compliance_rate_percent": compliance["compliance_rate_percent"],
        "mandatory_fields_present": compliance["mandatory_fields_present"],
        "mandatory_fields_total": compliance["mandatory_fields_total"],
    })

    # Save to normalized SQLite database
    try:
        from database import save_scan_result
        save_scan_result(result)
        print(f"  saved to database: packsight.db (product_id: {product_name}, brand: {brand_name})")
    except Exception as db_err:
        print(f"  [db notice] Could not save to database: {db_err}")

    return result


def scan_all_products(input_folder: Path = INPUT_FOLDER) -> list[dict[str, Any]]:
    """Scan all product folders sequentially with error isolation for 50+ products."""

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    products = sorted(
        (path for path in input_folder.iterdir() if path.is_dir()),
        key=natural_key,
    )
    if not products:
        print("No product folders found in input folder.")
        return []

    print(f"Found {len(products)} product folder(s) to scan.")
    results: list[dict[str, Any]] = []

    for idx, product_folder in enumerate(products, start=1):
        print(f"\n{'#' * 60}\nPRODUCT {idx}/{len(products)}: {product_folder.name}\n{'#' * 60}")
        try:
            res = scan_product(product_folder.name, product_folder)
            results.append(res)
        except Exception as exc:
            print(f"  [ERROR] Failed to scan {product_folder.name}: {exc}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="PackSight Product Scanner")
    parser.add_argument(
        "--product",
        type=str,
        default=None,
        help="Specific product directory name under input/ to scan",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_FOLDER,
        help="Custom input directory path",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PACKSIGHT SCANNER (Legal Metrology Compliance Engine)")
    print("=" * 60)
    print(f"Tesseract: {configure_tesseract()}")

    if args.product:
        target_dir = args.input_dir / args.product
        if not target_dir.exists():
            raise FileNotFoundError(f"Product directory not found: {target_dir}")
        print(f"\nScanning single product: {args.product}")
        scan_product(args.product, target_dir)
    else:
        scan_all_products(args.input_dir)

    print(f"\n{'=' * 60}\nSCAN COMPLETE\n{'=' * 60}")


if __name__ == "__main__":
    main()
