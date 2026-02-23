"""
Simple CLI for OCR invoice extraction and AI review.
Run with: python app.py
"""
import sys
from pathlib import Path

from main import (
    GEMINI_API_KEY,
    _check_tesseract_installed,
    run_files,
)

_BASE = Path(__file__).resolve().parent


def prompt_model() -> str:
    """Ask user to choose local or remote model."""
    print("\nModel:")
    print("  1. Ollama (local, default localhost:11434)")
    print("  2. llama.cpp (local, OpenAI-compatible)")
    print("  3. Remote (Google Gemini)")
    while True:
        choice = input("Choose [1/2/3]: ").strip()
        if choice == "1":
            return "ollama"
        if choice == "2":
            return "local"
        if choice == "3":
            if not GEMINI_API_KEY.strip():
                print("  GEMINI_API_KEY not set in .env. Use Ollama/llama.cpp or add the key.")
                continue
            return "gemini"
        print("  Enter 1, 2, or 3")


def main() -> None:
    print("=== Invoice OCR & AI Review ===\n")
    ok, err = _check_tesseract_installed()
    if not ok:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
    llm = prompt_model()
    path_input = input("PDF file or folder path: ").strip()
    if not path_input:
        print("No path given. Exiting.")
        return

    p = Path(path_input)
    if not p.is_absolute():
        p = _BASE / p
    if p.is_file() and p.suffix.lower() == ".pdf":
        paths = [p]
    elif p.is_dir():
        paths = sorted(p.glob("*.pdf"))
        if not paths:
            print(f"No PDFs found in {p}")
            return
        print(f"Found {len(paths)} PDF(s)")
    else:
        print(f"Not found or not a PDF: {p}")
        return

    llm_names = {"gemini": "Gemini", "ollama": "Ollama", "local": "llama.cpp"}
    print(f"\nProcessing with {llm_names.get(llm, llm)} (extract + LLM in parallel)...\n")

    run_files(
        paths,
        ask_review=True,
        base_dir=_BASE,
        llm=llm,
        overlap_extract_llm=True,
    )
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
