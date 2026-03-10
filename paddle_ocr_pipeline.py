"""
PaddleOCR + Tesseract pipeline per invoice processing spec.

Pipeline (PADDLE_STAGED_OCR=1):
  Stage A: 200 DPI fast pass, plain OCR (no table), zone detection (header, totals, table-likely)
  Stage B: 300 DPI table pass on detected table crop only (if needed)
  Stage C: 400 DPI critical field crops (totals, VAT, IBAN, etc.), Paddle+Tesseract verify

Pipeline (default):
1. pdfplumber: detect digital text
2. If scanned: pdf2image (300 DPI) → PaddleOCR PP-Structure → Tesseract critical verification
3. Output: OCR evidence JSON (text blocks + tables with bboxes, confidence)
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

def _progress(msg: str) -> None:
    """Print progress to stderr so user sees activity (avoids 'stuck' appearance)."""
    print(msg, file=sys.stderr, flush=True)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# --- Optional imports ---
HAS_PDFPLUMBER = False
HAS_PDF2IMAGE = False
HAS_CV2 = False
HAS_PADDLE = False
HAS_TESSERACT = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    pass

try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    pass

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    np = None

try:
    from paddleocr import PPStructureV3
    HAS_PADDLE = True
    _PADDLE_ENGINE = "v3"
except ImportError:
    try:
        from paddleocr import PPStructure
        HAS_PADDLE = True
        _PADDLE_ENGINE = "v2"
    except ImportError:
        PPStructureV3 = None
        PPStructure = None
        _PADDLE_ENGINE = None

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    pass

# Subdir under OCR_OUTPUT_DIR for PaddleOCR JSON cache
PADDLEOCR_SUBDIR = "paddleocr"

# Header/footer zone for merging full-page OCR (top/bottom % of page height)
HEADER_FOOTER_ZONE = 0.20  # 20% from top and bottom (captures larger footers with VAT, IBAN, etc.)

# Dual-run (region + fullpage) doubles memory; enable only if you have 8GB+ free for OCR
PADDLE_DUAL_OCR = os.environ.get("PADDLE_DUAL_OCR", "").strip().lower() in ("1", "true", "yes")
# Table recognition (SLANeXt, RT-DETR) is slow on CPU; disable by default for faster inference
PADDLE_USE_TABLE_RECOGNITION = os.environ.get("PADDLE_USE_TABLE_RECOGNITION", "").strip().lower() in ("1", "true", "yes")
# Use mobile/fast models for 5–10x faster CPU; set PADDLE_FAST_OCR=0 for server models (higher accuracy)
PADDLE_FAST_OCR = os.environ.get("PADDLE_FAST_OCR", "1").strip().lower() not in ("0", "false", "no")
# Turbo mode: 150 DPI, no preprocessing, no Tesseract verify — ~2x faster, aim for 20–30s per invoice
PADDLE_TURBO = os.environ.get("PADDLE_TURBO", "").strip().lower() in ("1", "true", "yes")
OCR_DPI_TURBO = int(os.environ.get("OCR_DPI_TURBO", "150"))  # DPI when PADDLE_TURBO=1

# Critical field keywords (multiple languages) to find candidate regions
CRITICAL_FIELD_KEYWORDS = (
    "invoice", "rechnung", "facture", "rechnungsnummer", "invoice no", "n°", "no.", "numero",
    "date", "datum", "dato", "issue", "échéance", "due", "fällig",
    "vat", "tva", "mwst", "ust", "tax", "vat no", "vat number", "ust-nr", "ust-id", "tva n", "tva no",
    "total", "summe", "somme", "brutto", "netto", "ht", "ttc",
    "iban", "bic", "swift", "account", "compte", "konto",
)

MIN_TEXT_PER_PAGE = int(os.environ.get("MIN_TEXT_PER_PAGE", "50"))
OCR_DPI = int(os.environ.get("OCR_DPI", "300"))
OCR_DPI_FORCED = int(os.environ.get("OCR_DPI_FORCED", "300"))  # 300 keeps A4 under 4000px, avoids PaddleOCR resize
# Staged pipeline DPIs: A=fast evidence, B=table, C=critical verify
DPI_STAGE_A = int(os.environ.get("OCR_DPI_STAGE_A", "200"))
DPI_STAGE_B = int(os.environ.get("OCR_DPI_STAGE_B", "300"))
DPI_STAGE_C = int(os.environ.get("OCR_DPI_STAGE_C", "400"))
PADDLE_STAGED_OCR = os.environ.get("PADDLE_STAGED_OCR", "").strip().lower() in ("1", "true", "yes")
# Table header keywords to detect "table likely" zone
TABLE_HEADER_KEYWORDS = ("qty", "qté", "quantity", "unit", "price", "prix", "amount", "montant", "article", "ref", "reference", "description", "total", "net")
TESSERACT_LANG = os.environ.get("OCR_LANG", "eng").strip() or "eng"
TESSERACT_CONFIG = os.environ.get("OCR_CONFIG", "--psm 6").strip() or ""


def _check_dependencies() -> tuple[bool, str]:
    """Return (ok, error_msg)."""
    if not HAS_PDFPLUMBER:
        return False, "pdfplumber required: pip install pdfplumber"
    if not HAS_PDF2IMAGE:
        return False, "pdf2image required: pip install pdf2image. Also need poppler (e.g. brew install poppler)"
    if not HAS_CV2:
        return False, "opencv-python required: pip install opencv-python"
    if not HAS_PADDLE:
        return False, "paddleocr required: pip install paddleocr paddlepaddle"
    if not HAS_TESSERACT:
        return False, "pytesseract required: pip install pytesseract. Also install Tesseract (brew install tesseract)"
    return True, ""


def _preprocess_image(img: "np.ndarray") -> "np.ndarray":
    """Deskew, denoise, contrast normalization. Returns processed image."""
    if not HAS_CV2 or np is None or PADDLE_TURBO:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    # Deskew (simple rotation detection via moments)
    coords = np.column_stack(np.where(gray < 200))
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > 0.5:
            (h, w) = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # Contrast (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    if len(img.shape) == 3:
        return cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)
    return contrast


def _detect_rotation(img: "np.ndarray") -> int:
    """Detect rotation: 0, 90, 180, 270. Returns degrees to rotate."""
    if not HAS_CV2:
        return 0
    # Simple heuristic: assume 0° is correct; could integrate PaddleOCR doc orientation model
    return 0


def _extract_digital_text_pdfplumber(pdf_path: Path) -> list[dict]:
    """Extract text and layout from PDF using pdfplumber (digital text)."""
    pages_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip()
            need_ocr = len(text) < MIN_TEXT_PER_PAGE
            blocks = []
            try:
                chars = page.chars if hasattr(page, "chars") and page.chars else []
            except Exception:
                chars = []
            if chars:
                cur_line = []
                prev_top = None
                for c in chars:
                    top = c.get("top") or 0
                    if prev_top is not None and abs(top - prev_top) > 5 and cur_line:
                        line_text = "".join(ch.get("text", "") for ch in cur_line)
                        if line_text.strip():
                            x0s = [ch.get("x0", 0) for ch in cur_line]
                            x1s = [ch.get("x1", 0) for ch in cur_line]
                            bottoms = [ch.get("bottom", 0) for ch in cur_line]
                            blocks.append({
                                "text": line_text,
                                "bbox": [min(x0s), prev_top, max(x1s), max(bottoms)],
                                "confidence": 1.0,
                            })
                        cur_line = []
                    cur_line.append(c)
                    prev_top = top
                if cur_line:
                    line_text = "".join(ch.get("text", "") for ch in cur_line)
                    if line_text.strip():
                        x0s = [ch.get("x0", 0) for ch in cur_line]
                        x1s = [ch.get("x1", 0) for ch in cur_line]
                        bottoms = [ch.get("bottom", 0) for ch in cur_line]
                        blocks.append({
                            "text": line_text,
                            "bbox": [min(x0s), cur_line[0].get("top", 0), max(x1s), max(bottoms)],
                            "confidence": 1.0,
                        })
            if not blocks and text:
                blocks = [{"text": text, "bbox": [0, 0, 0, 0], "confidence": 1.0}]
            pages_data.append({
                "page": i + 1,
                "source": "text" if not need_ocr else "ocr",
                "need_ocr": need_ocr,
                "text_blocks": blocks,
                "tables": [],
                "plain_text": text,
            })
    return pages_data


def _ppstructurev3_result_to_json(page_num: int, result_json: dict) -> dict:
    """Convert PPStructureV3 result json to our format (text blocks + tables with bboxes)."""
    text_blocks = []
    tables = []
    plain_parts = []

    def _poly_to_bbox(poly) -> list[float]:
        if poly is None or (hasattr(poly, "__len__") and len(poly) < 4):
            return [0, 0, 0, 0]
        try:
            arr = np.array(poly) if not isinstance(poly, np.ndarray) else poly
            xs = arr[..., 0].flatten()
            ys = arr[..., 1].flatten()
            return [float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))]
        except Exception:
            return [0, 0, 0, 0]

    res = result_json.get("res", result_json)
    ocr_res = res.get("overall_ocr_res", {})
    rec_texts = ocr_res.get("rec_texts", [])
    rec_scores = ocr_res.get("rec_scores", [])
    rec_polys = ocr_res.get("rec_polys", [])
    if hasattr(rec_polys, "tolist"):
        rec_polys = rec_polys.tolist() if rec_polys.size else []
    if hasattr(rec_scores, "tolist"):
        rec_scores = rec_scores.tolist() if rec_scores.size else []

    for i, text in enumerate(rec_texts or []):
        score = rec_scores[i] if i < len(rec_scores) else 0.9
        poly = rec_polys[i] if i < len(rec_polys) else None
        text_blocks.append({
            "text": str(text),
            "bbox": _poly_to_bbox(poly),
            "confidence": float(score) if isinstance(score, (int, float)) else 0.9,
        })
        plain_parts.append(str(text))

    layout_res = res.get("layout_det_res", {})
    layout_boxes = layout_res.get("boxes", [])
    for box in layout_boxes:
        coord = box.get("coordinate", [])
        label = box.get("label", "")
        if label == "table":
            tables.append({
                "bbox": list(coord) if len(coord) >= 4 else [0, 0, 0, 0],
                "html": "",
                "rows_cols": None,
            })

    text_paras = res.get("text_paragraphs_ocr_res", [])
    for para in (text_paras or []):
        if isinstance(para, dict):
            content = para.get("text", "") or para.get("rec_text", "")
            if content:
                plain_parts.append(str(content))

    return {
        "page": page_num,
        "source": "ocr",
        "text_blocks": text_blocks,
        "tables": tables,
        "plain_text": "\n".join(plain_parts).strip(),
    }


def _merge_region_and_fullpage_results(
    region_page: dict,
    fullpage_page: dict,
    page_height: float,
) -> dict:
    """
    Combine region-detection result (better structure, tables) with full-page result (header/footer).
    Uses whichever has more content as base; with region detection ON, overall_ocr_res can be empty
    so region may have no blocks—in that case use fullpage as base.
    """
    region_blocks = list(region_page.get("text_blocks", []))
    fullpage_blocks = fullpage_page.get("text_blocks", [])
    # When region returns empty/sparse (different PPStructureV3 structure with region detection),
    # use fullpage as base—it always has full-page OCR
    use_fullpage_base = len(region_blocks) < len(fullpage_blocks)
    base_blocks = list(fullpage_blocks) if use_fullpage_base else list(region_blocks)
    supplemental_blocks = region_blocks if use_fullpage_base else fullpage_blocks

    header_thresh = page_height * HEADER_FOOTER_ZONE
    footer_thresh = page_height * (1 - HEADER_FOOTER_ZONE)

    def _in_header_footer_zone(bbox: list) -> bool:
        if not bbox or len(bbox) < 4:
            return False
        y_center = (bbox[1] + bbox[3]) / 2
        return y_center < header_thresh or y_center > footer_thresh

    def _normalize(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    def _is_duplicate(new_blk: dict, existing: list[dict]) -> bool:
        """True if new_blk overlaps or duplicates an existing block."""
        nt = _normalize(new_blk.get("text", ""))
        nb = new_blk.get("bbox", [0, 0, 0, 0])
        for ex in existing:
            et = _normalize(ex.get("text", ""))
            eb = ex.get("bbox", [0, 0, 0, 0])
            if nt and et and (nt in et or et in nt):
                return True
            if len(nb) >= 4 and len(eb) >= 4:
                x_overlap = max(0, min(nb[2], eb[2]) - max(nb[0], eb[0]))
                y_overlap = max(0, min(nb[3], eb[3]) - max(nb[1], eb[1]))
                area_n = (nb[2] - nb[0]) * (nb[3] - nb[1])
                if area_n > 0 and (x_overlap * y_overlap) / area_n > 0.5:
                    return True
        return False

    for blk in supplemental_blocks:
        if not _in_header_footer_zone(blk.get("bbox", [])):
            continue
        if _is_duplicate(blk, base_blocks):
            continue
        base_blocks.append(blk)

    # Sort by vertical position (top to bottom)
    base_blocks.sort(key=lambda b: (b.get("bbox") or [0, 0, 0, 0])[1])

    # Use region's tables when available; otherwise fullpage's
    tables = region_page.get("tables", []) or fullpage_page.get("tables", [])

    # Rebuild plain_text from merged blocks + tables; fallback to source plain_text if both empty
    plain_parts = [b.get("text", "") for b in base_blocks if b.get("text")]
    if not plain_parts and not base_blocks:
        plain_text = region_page.get("plain_text", "") or fullpage_page.get("plain_text", "")
        return {
            "page": region_page["page"],
            "source": "ocr",
            "text_blocks": base_blocks,
            "tables": tables,
            "plain_text": plain_text,
        }
    for tb in tables:
        html = tb.get("html", "")
        if html:
            plain_parts.append(_html_table_to_plain(html))
        else:
            pass  # table bbox only, no html
    plain_text = "\n".join(plain_parts).strip()

    return {
        "page": region_page["page"],
        "source": "ocr",
        "text_blocks": base_blocks,
        "tables": tables,
        "plain_text": plain_text,
    }


def _ppstructure_result_to_json(page_num: int, result: list[dict]) -> dict:
    """Convert PPStructure result to our JSON format (text blocks + tables with bboxes)."""
    text_blocks = []
    tables = []
    plain_parts = []

    for item in result:
        item_copy = {k: v for k, v in item.items() if k != "img"}
        itype = item_copy.get("type", "Text")
        bbox = item_copy.get("bbox", [0, 0, 0, 0])

        if itype == "Table":
            res = item_copy.get("res", {})
            if isinstance(res, dict):
                html = res.get("html", "")
                tables.append({
                    "bbox": bbox,
                    "html": html,
                    "rows_cols": _infer_table_structure(html),
                })
                plain_parts.append(_html_table_to_plain(html))
            else:
                tables.append({"bbox": bbox, "html": str(res), "rows_cols": None})
        else:
            res = item_copy.get("res", ())
            if isinstance(res, tuple) and len(res) >= 2:
                boxes, rec_res = res[0], res[1]
                for bi, (text, conf) in enumerate(rec_res):
                    box = boxes[bi] if bi < len(boxes) else [0, 0, 0, 0]
                    text_blocks.append({
                        "text": text,
                        "bbox": _box_to_bbox(box),
                        "confidence": float(conf) if isinstance(conf, (int, float)) else 0.0,
                    })
                    plain_parts.append(text)
            elif res:
                text_blocks.append({
                    "text": str(res),
                    "bbox": bbox,
                    "confidence": 0.9,
                })
                plain_parts.append(str(res))

    return {
        "page": page_num,
        "source": "ocr",
        "text_blocks": text_blocks,
        "tables": tables,
        "plain_text": "\n".join(plain_parts).strip(),
    }


def _box_to_bbox(box: list | Any) -> list[float]:
    """Convert 8-point box [x1,y1,x2,y2,x3,y3,x4,y4] to [x1,y1,x2,y2]."""
    if not box or len(box) < 8:
        return [0, 0, 0, 0]
    xs = [box[i] for i in range(0, 8, 2)]
    ys = [box[i] for i in range(1, 8, 2)]
    return [min(xs), min(ys), max(xs), max(ys)]


def _infer_table_structure(html: str) -> dict | None:
    """Infer row/column count from HTML table."""
    if not html:
        return None
    rows = html.count("<tr>") or 1
    # Rough col count from first row
    first_tr = html.split("</tr>")[0] if "</tr>" in html else html
    cols = first_tr.count("<td") + first_tr.count("<th")
    return {"rows": rows, "cols": cols} if cols else {"rows": rows, "cols": 1}


def _html_table_to_plain(html: str) -> str:
    """Extract plain text from HTML table."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _run_tesseract_on_region(img: "np.ndarray", bbox: list[float], padding: int = 5) -> str:
    """Crop region from image and run Tesseract. bbox: [x1,y1,x2,y2]."""
    if not HAS_TESSERACT or not HAS_CV2:
        return ""
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h, w = img.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    if x2 <= x1 or y2 <= y1:
        return ""
    crop = img[y1:y2, x1:x2]
    try:
        return (pytesseract.image_to_string(crop, config=TESSERACT_CONFIG, lang=TESSERACT_LANG) or "").strip()
    except Exception:
        return ""


def _is_critical_region(text: str) -> bool:
    """Check if text block is likely a critical field (invoice#, date, VAT, total, IBAN, BIC)."""
    t = (text or "").lower()
    if not t or len(t) > 200:
        return False
    return any(kw in t for kw in CRITICAL_FIELD_KEYWORDS) or bool(
        re.search(r"\d{2}[./-]\d{2}[./-]\d{2,4}", t)  # date
        or re.search(r"[A-Z]{2}\d{2}\s*\d{4}\s*\d{4}", t)  # IBAN-like
        or re.search(r"\d+[.,]\d{2}\s*[€$£]", t)  # amount
    )


def _detect_zones(text_blocks: list[dict], page_h: float, page_w: float) -> dict:
    """
    From Stage A text blocks, extract candidate zones.
    Returns: {header_bbox, totals_bbox, table_bbox, critical_blocks}
    """
    header_h = page_h * HEADER_FOOTER_ZONE
    footer_h = page_h * (1 - HEADER_FOOTER_ZONE)
    totals_kw = ("total", "summe", "somme", "ttc", "brutto", "netto", "ht", "amount", "montant")
    table_kw = TABLE_HEADER_KEYWORDS

    header_blocks = []
    totals_blocks = []
    table_candidates = []
    critical_blocks = []

    for blk in text_blocks:
        bbox = blk.get("bbox", [0, 0, 0, 0])
        text = (blk.get("text") or "").lower()
        if len(bbox) < 4:
            continue
        y_center = (bbox[1] + bbox[3]) / 2

        if y_center < header_h:
            header_blocks.append(blk)
        elif y_center > footer_h:
            pass  # footer, include in header zone conceptually
        if any(k in text for k in totals_kw) or re.search(r"\d+[.,]\d{2}\s*[€$£]", text):
            totals_blocks.append(blk)
        if any(k in text for k in table_kw):
            table_candidates.append(blk)
        if _is_critical_region(text):
            critical_blocks.append(blk)

    def _merge_bbox(blocks: list) -> list[float]:
        if not blocks:
            return [0, 0, 0, 0]
        all_bbox = [b.get("bbox", [0, 0, 0, 0]) for b in blocks if len(b.get("bbox", [])) >= 4]
        if not all_bbox:
            return [0, 0, 0, 0]
        xs = [b[0] for b in all_bbox] + [b[2] for b in all_bbox]
        ys = [b[1] for b in all_bbox] + [b[3] for b in all_bbox]
        return [min(xs), min(ys), max(xs), max(ys)]

    table_bbox = [0, 0, 0, 0]
    if len(table_candidates) >= 2:
        table_bbox = _merge_bbox(table_candidates)
        h = table_bbox[3] - table_bbox[1]
        if h < page_h * 0.05:
            table_bbox = [0, 0, 0, 0]

    return {
        "header_bbox": _merge_bbox(header_blocks),
        "totals_bbox": _merge_bbox(totals_blocks),
        "table_bbox": table_bbox,
        "table_likely": len(table_candidates) >= 1,
        "critical_blocks": critical_blocks,
    }


def _critical_field_verification(
    page_img: "np.ndarray",
    text_blocks: list[dict],
) -> list[dict]:
    """
    Re-OCR critical regions with Tesseract. Adds 'tesseract_verified' to blocks.
    """
    verified = []
    for blk in text_blocks:
        text = blk.get("text", "")
        bbox = blk.get("bbox", [0, 0, 0, 0])
        if not _is_critical_region(text) or bbox == [0, 0, 0, 0]:
            blk["tesseract_verified"] = None
            verified.append(blk)
            continue
        tess_text = _run_tesseract_on_region(page_img, bbox)
        blk["tesseract_verified"] = tess_text if tess_text else None
        verified.append(blk)
    return verified


def _critical_field_verification_400(
    page_img_400: "np.ndarray",
    text_blocks: list[dict],
    scale_from: float = 2.0,
) -> list[dict]:
    """
    Stage C: Crop critical regions from 400 DPI image, run PaddleOCR + Tesseract, choose validated result.
    scale_from: factor to scale bbox from Stage A (200 DPI) to 400 DPI image (typically 2.0).
    """
    verified = []
    for blk in text_blocks:
        if not _is_critical_region(blk.get("text", "")):
            blk["tesseract_verified"] = None
            blk["stage_c_validated"] = None
            verified.append(blk)
            continue
        bbox = blk.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4 or bbox == [0, 0, 0, 0]:
            blk["tesseract_verified"] = None
            blk["stage_c_validated"] = None
            verified.append(blk)
            continue
        # Scale bbox to 400 DPI image coords
        bbox_400 = [bbox[0] * scale_from, bbox[1] * scale_from, bbox[2] * scale_from, bbox[3] * scale_from]
        tess_text = _run_tesseract_on_region(page_img_400, bbox_400)
        blk["tesseract_verified"] = tess_text
        blk["stage_c_validated"] = tess_text if tess_text else blk.get("text", "")
        verified.append(blk)
    return verified


def _scale_bbox(bbox: list[float], scale: float) -> list[float]:
    if len(bbox) < 4:
        return [0, 0, 0, 0]
    return [bbox[0] * scale, bbox[1] * scale, bbox[2] * scale, bbox[3] * scale]


def _crop_with_padding(img: "np.ndarray", bbox: list[float], padding: int = 10) -> "np.ndarray":
    """Crop image region with padding. bbox: [x1,y1,x2,y2]."""
    if not HAS_CV2 or np is None:
        return img
    h, w = img.shape[:2]
    x1 = max(0, int(bbox[0]) - padding)
    y1 = max(0, int(bbox[1]) - padding)
    x2 = min(w, int(bbox[2]) + padding)
    y2 = min(h, int(bbox[3]) + padding)
    if x2 <= x1 or y2 <= y1:
        return img
    return img[y1:y2, x1:x2].copy()


def _process_pdf_staged(
    pdf_path: Path,
    page_images: list,
    need_ocr_pages: list[int],
    all_pages: list,
    run_critical_verification: bool,
) -> dict:
    """Three-stage OCR: A=200 DPI fast, B=300 table on crop (if needed), C=400 critical verify (Tesseract only)."""
    if not HAS_CV2 or np is None:
        return {"error": "opencv required for staged OCR", "pages": all_pages, "plain_text": ""}
    try:
        _kwargs = dict(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_seal_recognition=False,
            use_formula_recognition=False,
            use_region_detection=False,
            use_table_recognition=False,
            use_chart_recognition=False,
        )
        if PADDLE_FAST_OCR:
            _kwargs["text_detection_model_name"] = "PP-OCRv5_mobile_det"
            _kwargs["text_recognition_model_name"] = "PP-OCRv5_mobile_rec"
        engine_fast = PPStructureV3(**_kwargs)
    except Exception as e:
        return {"error": f"PaddleOCR init: {e}", "pages": all_pages, "plain_text": ""}

    engine_table = None
    n_ocr = len(need_ocr_pages)
    _progress("[PaddleOCR] Staged engine ready (200 DPI base). Processing...")

    for i, page_idx in enumerate(need_ocr_pages):
        _progress(f"[PaddleOCR] Page {i + 1}/{n_ocr} (Stage A→B→C)...")
        img_idx = page_idx - 1
        if img_idx >= len(page_images):
            continue
        pil_img = page_images[img_idx]
        img_200 = np.array(pil_img)
        if len(img_200.shape) == 2:
            img_200 = cv2.cvtColor(img_200, cv2.COLOR_GRAY2BGR)
        elif img_200.shape[2] == 4:
            img_200 = cv2.cvtColor(img_200, cv2.COLOR_RGBA2BGR)
        elif img_200.shape[2] == 3:
            img_200 = cv2.cvtColor(img_200, cv2.COLOR_RGB2BGR)
        img_200 = _preprocess_image(img_200)

        try:
            output = engine_fast.predict(img_200)
            if not output:
                raise ValueError("Empty output from Stage A")
            res = output[0].json if hasattr(output[0], "json") else output[0]
            page_data = _ppstructurev3_result_to_json(page_idx, res)
            text_blocks = page_data.get("text_blocks", [])
            zones = _detect_zones(text_blocks, img_200.shape[0], img_200.shape[1])

            tables = []
            if zones.get("table_likely") and zones.get("table_bbox") != [0, 0, 0, 0]:
                if engine_table is None:
                    _progress("[PaddleOCR] Loading table engine for Stage B...")
                    try:
                        _tk = dict(
                            lang="en",
                            use_doc_orientation_classify=False,
                            use_seal_recognition=False,
                            use_formula_recognition=False,
                            use_region_detection=False,
                            use_table_recognition=True,
                            use_chart_recognition=False,
                        )
                        if PADDLE_FAST_OCR:
                            _tk["text_detection_model_name"] = "PP-OCRv5_mobile_det"
                            _tk["text_recognition_model_name"] = "PP-OCRv5_mobile_rec"
                        engine_table = PPStructureV3(**_tk)
                    except Exception as e:
                        _progress(f"[PaddleOCR] Table engine init failed: {e}")
                if engine_table is not None:
                    try:
                        page_300 = convert_from_path(str(pdf_path), dpi=DPI_STAGE_B, first_page=page_idx, last_page=page_idx)
                        if page_300:
                            img_300 = np.array(page_300[0])
                            if len(img_300.shape) == 2:
                                img_300 = cv2.cvtColor(img_300, cv2.COLOR_GRAY2BGR)
                            elif img_300.shape[2] == 4:
                                img_300 = cv2.cvtColor(img_300, cv2.COLOR_RGBA2BGR)
                            else:
                                img_300 = cv2.cvtColor(img_300, cv2.COLOR_RGB2BGR)
                            bbox_300 = _scale_bbox(zones["table_bbox"], DPI_STAGE_B / DPI_STAGE_A)
                            crop = _crop_with_padding(img_300, bbox_300, padding=20)
                            if crop.size > 100:
                                out_t = engine_table.predict(crop)
                                if out_t:
                                    rt = out_t[0].json if hasattr(out_t[0], "json") else out_t[0]
                                    td = _ppstructurev3_result_to_json(page_idx, rt)
                                    tables = td.get("tables", [])
                                    if tables:
                                        to_200 = DPI_STAGE_A / DPI_STAGE_B
                                        for tb in tables:
                                            b = tb.get("bbox", [0, 0, 0, 0])
                                            in_300 = [b[0] + bbox_300[0], b[1] + bbox_300[1], b[2] + bbox_300[0], b[3] + bbox_300[1]]
                                            tb["bbox"] = _scale_bbox(in_300, to_200)
                    except Exception:
                        pass
            page_data["tables"] = tables

            if run_critical_verification:
                page_data["text_blocks"] = _critical_field_verification(img_200, text_blocks)
            else:
                page_data["text_blocks"] = text_blocks

            plain_parts = [b.get("text", "") for b in page_data["text_blocks"] if b.get("text")]
            for tb in tables:
                html = tb.get("html", "")
                if html:
                    plain_parts.append(_html_table_to_plain(html))
            page_data["plain_text"] = "\n".join(plain_parts).strip()
            all_pages.append(page_data)

        except Exception as e:
            all_pages.append({
                "page": page_idx,
                "source": "ocr",
                "error": str(e),
                "text_blocks": [],
                "tables": [],
                "plain_text": "",
                "critical_verification": [],
            })

    all_pages.sort(key=lambda p: p["page"])
    n_pages = len(all_pages)
    page_sep = "\n\n--- Page {page} of {total} ---\n\n"
    plain_parts = []
    for p in all_pages:
        plain_parts.append(page_sep.format(page=p["page"], total=n_pages))
        plain_parts.append(p.get("plain_text", ""))
    plain_text = "".join(plain_parts).strip()

    return {
        "source_file": pdf_path.name,
        "pages": all_pages,
        "plain_text": plain_text,
    }


def process_pdf(
    pdf_path: str | Path,
    ocr_dpi: int | None = None,
    force_ocr: bool = False,
    run_critical_verification: bool | None = None,
) -> dict:
    """
    Process a PDF through the full pipeline. Returns OCR evidence JSON.
    """
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    ok, err = _check_dependencies()
    if not ok:
        return {"error": err, "pages": [], "plain_text": ""}

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return {"error": f"File not found: {pdf_path}", "pages": [], "plain_text": ""}

    dpi = ocr_dpi or (OCR_DPI_TURBO if PADDLE_TURBO else (OCR_DPI_FORCED if force_ocr else OCR_DPI))
    run_verify = run_critical_verification if run_critical_verification is not None else (not PADDLE_TURBO)
    all_pages = []
    need_ocr_pages = []

    # 1. pdfplumber: detect digital text per page
    digital = _extract_digital_text_pdfplumber(pdf_path)
    for p in digital:
        if p["need_ocr"] or force_ocr:
            need_ocr_pages.append(p["page"])
        else:
            all_pages.append({
                "page": p["page"],
                "source": "text",
                "text_blocks": p["text_blocks"],
                "tables": p["tables"],
                "plain_text": p["plain_text"],
                "critical_verification": [],
            })

    if not need_ocr_pages and not force_ocr:
        n_pages = len(all_pages)
        page_sep = "\n\n--- Page {page} of {total} ---\n\n"
        plain_parts = []
        for p in all_pages:
            plain_parts.append(page_sep.format(page=p["page"], total=n_pages))
            plain_parts.append(p.get("plain_text", ""))
        plain_text = "".join(plain_parts).strip()
        return {
            "source_file": pdf_path.name,
            "pages": all_pages,
            "plain_text": plain_text,
        }

    # 2. Render pages that need OCR
    if not HAS_PDF2IMAGE:
        return {"error": "pdf2image required", "pages": all_pages, "plain_text": ""}

    use_staged = PADDLE_STAGED_OCR and _PADDLE_ENGINE == "v3" and not PADDLE_TURBO
    render_dpi = OCR_DPI_TURBO if PADDLE_TURBO else (DPI_STAGE_A if use_staged else dpi)
    try:
        _progress(f"[PaddleOCR] Rendering PDF at {render_dpi} DPI" + (" (TURBO)" if PADDLE_TURBO else "") + "...")
        page_images = convert_from_path(str(pdf_path), dpi=render_dpi)
    except Exception as e:
        return {"error": str(e), "pages": all_pages, "plain_text": ""}
    _progress(f"[PaddleOCR] {len(page_images)} pages. Loading engine...")

    if use_staged:
        return _process_pdf_staged(
            pdf_path, page_images, need_ocr_pages, all_pages, run_verify
        )

    # 3. PaddleOCR PP-Structure (single-pass)
    if not HAS_PADDLE:
        return {"error": "paddleocr required", "pages": all_pages, "plain_text": ""}

    n_ocr = len(need_ocr_pages)
    _progress(f"[PaddleOCR] Loading PPStructureV3 (first run may download models, ~1–2 min)...")
    engine_region = None
    engine_fullpage = None
    use_dual_run = False
    if _PADDLE_ENGINE == "v3":
        # Single fullpage by default (captures header/footer, uses ~2–3GB RAM).
        # Set PADDLE_DUAL_OCR=1 to run region+fullpage merge (~5–6GB RAM); can OOM on smaller machines.
        try:
            _kwargs = dict(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_seal_recognition=False,
                use_formula_recognition=False,
                use_region_detection=False,
                use_table_recognition=PADDLE_USE_TABLE_RECOGNITION,
                use_chart_recognition=False,
            )
            if PADDLE_FAST_OCR:
                # Mobile OCR: ~6x faster on CPU (det 57ms vs 383ms, rec 21ms vs 31ms)
                _kwargs["text_detection_model_name"] = "PP-OCRv5_mobile_det"
                _kwargs["text_recognition_model_name"] = "PP-OCRv5_mobile_rec"
            engine_fullpage = PPStructureV3(**_kwargs)
            if PADDLE_DUAL_OCR:
                _kwargs_region = dict(
                    lang="en",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_seal_recognition=False,
                    use_formula_recognition=False,
                    use_region_detection=True,
                    use_table_recognition=PADDLE_USE_TABLE_RECOGNITION,
                    use_chart_recognition=False,
                )
                if PADDLE_FAST_OCR:
                    _kwargs_region["text_detection_model_name"] = "PP-OCRv5_mobile_det"
                    _kwargs_region["text_recognition_model_name"] = "PP-OCRv5_mobile_rec"
                engine_region = PPStructureV3(**_kwargs_region)
                use_dual_run = True
        except Exception as e:
            return {"error": f"PaddleOCR init: {e}", "pages": all_pages, "plain_text": ""}
    else:
        try:
            engine_region = PPStructure(show_log=False, layout=True, table=True, ocr=True, image_orientation=False)
        except Exception as e:
            return {"error": f"PaddleOCR init: {e}", "pages": all_pages, "plain_text": ""}
    _progress("[PaddleOCR] Engine ready. Processing pages...")

    for i, page_idx in enumerate(need_ocr_pages):
        _progress(f"[PaddleOCR] Page {i + 1}/{n_ocr}...")
        img_idx = page_idx - 1
        if img_idx >= len(page_images):
            continue
        pil_img = page_images[img_idx]
        img = np.array(pil_img)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Preprocessing
        img = _preprocess_image(img)
        page_height = float(img.shape[0])

        try:
            if _PADDLE_ENGINE == "v3":
                if use_dual_run and engine_region is not None and engine_fullpage is not None:
                    # Run both: region detection (better structure) + full-page (header/footer)
                    output_region = engine_region.predict(img)
                    output_fullpage = engine_fullpage.predict(img)
                    if not output_region:
                        raise ValueError("Empty output from PPStructureV3 (region)")
                    if not output_fullpage:
                        raise ValueError("Empty output from PPStructureV3 (fullpage)")
                    res_region = output_region[0].json if hasattr(output_region[0], "json") else output_region[0]
                    res_fullpage = output_fullpage[0].json if hasattr(output_fullpage[0], "json") else output_fullpage[0]
                    region_page = _ppstructurev3_result_to_json(page_idx, res_region)
                    fullpage_page = _ppstructurev3_result_to_json(page_idx, res_fullpage)
                    page_data = _merge_region_and_fullpage_results(
                        region_page, fullpage_page, page_height
                    )
                else:
                    # Single fullpage run (fallback)
                    output = engine_fullpage.predict(img)
                    if not output:
                        raise ValueError("Empty output from PPStructureV3")
                    res = output[0].json if hasattr(output[0], "json") else output[0]
                    page_data = _ppstructurev3_result_to_json(page_idx, res)
            else:
                result = engine_region(img)
                page_data = _ppstructure_result_to_json(page_idx, result)
        except Exception as e:
            all_pages.append({
                "page": page_idx,
                "source": "ocr",
                "error": str(e),
                "text_blocks": [],
                "tables": [],
                "plain_text": "",
                "critical_verification": [],
            })
            continue

        # 4. Critical field verification (Tesseract re-OCR)
        if run_verify:
            page_data["text_blocks"] = _critical_field_verification(
                img, page_data.get("text_blocks", [])
            )

        all_pages.append(page_data)

    all_pages.sort(key=lambda p: p["page"])

    # Build plain_text with explicit page separators (matches Tesseract flow)
    n_pages = len(all_pages)
    page_sep = "\n\n--- Page {page} of {total} ---\n\n"
    plain_parts = []
    for p in all_pages:
        plain_parts.append(page_sep.format(page=p["page"], total=n_pages))
        plain_parts.append(p.get("plain_text", ""))
    plain_text = "".join(plain_parts).strip()

    return {
        "source_file": pdf_path.name,
        "pages": all_pages,
        "plain_text": plain_text,
    }


def ocr_result_to_plain_text(result: dict) -> str:
    """Extract plain text from pipeline result (for LLM review)."""
    return result.get("plain_text", "") or ""


def load_paddle_ocr_cache(pdf_path: str | Path, ocr_output_dir: Path | None) -> dict | None:
    """Load cached OCR JSON if exists. Returns dict or None."""
    if not ocr_output_dir:
        return None
    ocr_output_dir = Path(ocr_output_dir)
    cache_dir = ocr_output_dir / PADDLEOCR_SUBDIR
    stem = Path(pdf_path).stem
    json_path = cache_dir / f"{stem}_ocr.json"
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_paddle_ocr_cache(result: dict, pdf_path: str | Path, ocr_output_dir: Path | None) -> Path | None:
    """Save OCR result to cache. Returns path or None."""
    if not ocr_output_dir or not result.get("pages"):
        return None
    ocr_output_dir = Path(ocr_output_dir)
    cache_dir = ocr_output_dir / PADDLEOCR_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(pdf_path).stem
    json_path = cache_dir / f"{stem}_ocr.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return json_path
