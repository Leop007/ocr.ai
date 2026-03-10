#!/usr/bin/env python3
"""
Run PaddleOCR + Tesseract pipeline on PDFs (per invoice processing spec).
Output: OCR evidence JSON (text blocks + tables with bboxes).
Usage: python run_paddle_ocr.py file.pdf [file2.pdf ...]
       python run_paddle_ocr.py --force-ocr invoice.pdf
"""
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# Add project root for imports
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from paddle_ocr_pipeline import (
    load_paddle_ocr_cache,
    ocr_result_to_plain_text,
    process_pdf,
    save_paddle_ocr_cache,
    PADDLEOCR_SUBDIR,
)

# Resolve OCR output dir from env (same pattern as main.py)
def _get_ocr_output_dir() -> Path | None:
    import os
    d = os.environ.get("OCR_OUTPUT_DIR", "ocr_output")
    if not d or not d.strip():
        return None
    p = Path(d.strip())
    if not p.is_absolute():
        p = _script_dir / p
    return p


def main() -> None:
    argv = sys.argv[1:]
    force_ocr = "--force-ocr" in argv
    argv = [a for a in argv if a != "--force-ocr"]

    if not argv or "--help" in argv or "-h" in argv:
        print(__doc__)
        print("\nOptions:")
        print("  --force-ocr   OCR all pages (ignore digital text)")
        sys.exit(0)

    ocr_output_dir = _get_ocr_output_dir()
    if ocr_output_dir:
        cache_dir = ocr_output_dir / PADDLEOCR_SUBDIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output: {cache_dir}/\n")

    base = _script_dir
    for arg in argv:
        path = Path(arg)
        if not path.is_absolute():
            path = base / path
        if not path.exists():
            print(f"Skip (not found): {path}")
            continue
        if path.suffix.lower() != ".pdf":
            print(f"Skip (not PDF): {path}")
            continue

        print(f"Processing: {path.name}")
        cached = load_paddle_ocr_cache(path, ocr_output_dir) if ocr_output_dir else None
        if cached and not force_ocr:
            print(f"  Using cached: {ocr_output_dir / PADDLEOCR_SUBDIR / path.stem}_ocr.json")
            result = cached
        else:
            result = process_pdf(path, force_ocr=force_ocr)
            if result.get("error"):
                print(f"  Error: {result['error']}")
                continue
            if ocr_output_dir:
                out_path = save_paddle_ocr_cache(result, path, ocr_output_dir)
                if out_path:
                    print(f"  Saved: {out_path}")

        plain = ocr_result_to_plain_text(result)
        print(f"  Extracted {len(plain)} chars, {len(result.get('pages', []))} pages")
        if plain:
            print("  --- Preview (first 400 chars) ---")
            print("  ", plain[:400].replace("\n", "\n  ") + ("..." if len(plain) > 400 else ""))
        print()


if __name__ == "__main__":
    main()
