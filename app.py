"""
Simple CLI for OCR invoice extraction and AI review.
Run with: python app.py
"""
import re
import sys
from pathlib import Path

from main import (
    GEMINI_API_KEY,
    OCR_METHOD_LLM,
    OCR_METHOD_TESSERACT,
    SUPPORTED_OFFICE_EXTENSIONS,
    _check_tesseract_installed,
    run_files,
)

_BASE = Path(__file__).resolve().parent


def prompt_model() -> tuple[str, bool, str]:
    """Ask user: OCR only (Tesseract or LLM), or OCR + LLM review. Returns (llm_provider, ask_review, ocr_method)."""
    print("\nMode:")
    print("  1. OCR only with Tesseract (no LLM)")
    print("  2. OCR only with LLM (vision; no Tesseract, no review)")
    print("  3. OCR + Ollama (Tesseract + local review)")
    print("  4. OCR + llama.cpp / LM Studio (Tesseract + local review)")
    print("  5. OCR + Gemini (Tesseract + remote review)")
    while True:
        choice = input("Choose [1/2/3/4/5]: ").strip()
        if choice == "1":
            return "auto", False, OCR_METHOD_TESSERACT
        if choice == "2":
            print("  Which LLM for vision?")
            print("    a) Ollama   b) LM Studio / llama.cpp   c) Gemini")
            while True:
                sub = input("  Choose [a/b/c]: ").strip().lower()
                if sub == "a":
                    return "ollama", False, OCR_METHOD_LLM
                if sub == "b":
                    return "local", False, OCR_METHOD_LLM
                if sub == "c":
                    if not GEMINI_API_KEY.strip():
                        print("    GEMINI_API_KEY not set. Use a or b, or add the key to .env.")
                        continue
                    return "gemini", False, OCR_METHOD_LLM
                print("  Enter a, b, or c")
        if choice == "3":
            return "ollama", True, OCR_METHOD_TESSERACT
        if choice == "4":
            return "local", True, OCR_METHOD_TESSERACT
        if choice == "5":
            if not GEMINI_API_KEY.strip():
                print("  GEMINI_API_KEY not set in .env. Use 1–4 or add the key.")
                continue
            return "gemini", True, OCR_METHOD_TESSERACT
        print("  Enter 1, 2, 3, 4, or 5")


def main() -> None:
    print("=== Invoice OCR & AI Review ===\n")
    llm, ask_review, ocr_method = prompt_model()
    if ocr_method == OCR_METHOD_TESSERACT:
        ok, err = _check_tesseract_installed()
        if not ok:
            print(f"Error: {err}", file=sys.stderr)
            sys.exit(1)
    path_input = input("File or folder path (PDF, Word, Excel): ").strip()
    if not path_input:
        print("No path given. Exiting.")
        return

    p = Path(path_input)
    if not p.is_absolute():
        p = _BASE / p
    allowed_suffixes = (".pdf",) + SUPPORTED_OFFICE_EXTENSIONS
    if p.is_file() and p.suffix.lower() in allowed_suffixes:
        paths = [p]
    elif p.is_dir():
        paths = []
        for ext in ("*.pdf", "*.doc", "*.docx", "*.xls", "*.xlsx"):
            paths.extend(p.glob(ext))
        paths = sorted(paths)
        if not paths:
            print(f"No supported files (PDF, doc/docx, xls/xlsx) found in {p}")
            return
        pattern_input = input("File pattern (regex, optional; Enter = all): ").strip()
        if pattern_input:
            try:
                pat = re.compile(pattern_input)
                paths = [f for f in paths if pat.search(f.name)]
            except re.error:
                print(f"Invalid regex: {pattern_input}. Using all files.")
            if not paths:
                print("No files match that pattern. Exiting.")
                return
        print(f"Found {len(paths)} file(s)")
    else:
        print(f"Not found or unsupported type: {p} (use PDF, .doc, .docx, .xls, .xlsx)")
        return

    if ask_review or ocr_method == OCR_METHOD_LLM:
        from main import _check_llm_server
        ok, err = _check_llm_server(llm)
        if not ok:
            print(err)
            sys.exit(1)
    llm_names = {"gemini": "Gemini", "ollama": "Ollama", "local": "llama.cpp", "auto": "—"}
    if not ask_review:
        mode = "OCR only (LLM vision)" if ocr_method == OCR_METHOD_LLM else "OCR only (Tesseract)"
    else:
        mode = f"OCR (Tesseract) + {llm_names.get(llm, llm)}"
    print(f"\nProcessing: {mode}\n")

    run_files(
        paths,
        ask_review=ask_review,
        base_dir=_BASE,
        llm=llm,
        overlap_extract_llm=ask_review,
        ocr_method=ocr_method,
    )
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
