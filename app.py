"""
Simple CLI for OCR invoice extraction and AI review.
Run with: python app.py
"""
from pathlib import Path

from main import (
    GEMINI_API_KEY,
    run_files,
)

_BASE = Path(__file__).resolve().parent


def prompt_model() -> str:
    """Ask user to choose local or remote model."""
    print("\nModel:")
    print("  1. Local (Ollama/llama.cpp)")
    print("  2. Remote (Google Gemini)")
    while True:
        choice = input("Choose [1/2]: ").strip()
        if choice == "1":
            return "local"
        if choice == "2":
            if not GEMINI_API_KEY.strip():
                print("  GEMINI_API_KEY not set in .env. Use Local or add the key.")
                continue
            return "gemini"
        print("  Enter 1 or 2")


def main() -> None:
    print("=== Invoice OCR & AI Review ===\n")
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

    print(f"\nProcessing with {'Gemini' if llm == 'gemini' else 'Ollama'} (extract + LLM in parallel)...\n")

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
