"""
Extract text from PDF, structure it, and optionally send to llama.cpp for invoice review.
"""
import json
import os
import re
from pathlib import Path

import PyPDF2
import requests
from dotenv import load_dotenv

# Load .env from the ocr directory (script location)
load_dotenv(Path(__file__).resolve().parent / ".env")

# Optional: for scanned PDFs (pip install pdf2image; system: poppler-utils)
try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False
try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False


def normalize_text(text: str) -> str:
    """Collapse whitespace, trim, keep paragraph breaks."""
    if not text or not text.strip():
        return ""
    # Normalize line endings and collapse multiple blanks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)  # at most one blank line
    return text.strip()


def extract_text_from_page(page) -> str:
    """Get text from a PyPDF2 page (digital text only)."""
    try:
        return page.extract_text() or ""
    except Exception:
        return ""


def extract_text_via_ocr(image) -> str:
    """Get text from a page image using Tesseract."""
    if not HAS_OCR:
        return ""
    try:
        return pytesseract.image_to_string(image) or ""
    except Exception:
        return ""


def pdf_to_structured_text(
    pdf_path: str | Path,
    use_ocr_fallback: bool = True,
    min_text_per_page: int = 50,
    ocr_dpi: int = 200,
) -> list[dict]:
    """
    Extract text from each page. Structure: list of {"page": 1-based index, "text": "...", "source": "text"|"ocr"}.

    - use_ocr_fallback: if True, run OCR on pages where extracted text is short (e.g. scanned).
    - min_text_per_page: if extracted text has fewer than this many chars, treat as scanned and use OCR.
    - ocr_dpi: DPI when rendering pages to images for OCR (higher = better quality, slower).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    result = []
    reader = PyPDF2.PdfReader(str(pdf_path))
    n_pages = len(reader.pages)

    # Optional: render all pages to images once (for OCR fallback)
    page_images = None
    if use_ocr_fallback and HAS_PDF2IMAGE and HAS_OCR:
        try:
            page_images = convert_from_path(str(pdf_path), dpi=ocr_dpi)
        except Exception:
            page_images = None

    for page_num in range(n_pages):
        page = reader.pages[page_num]
        text = extract_text_from_page(page)
        source = "text"

        if use_ocr_fallback and len(text.strip()) < min_text_per_page:
            if page_images is not None and page_num < len(page_images):
                text = extract_text_via_ocr(page_images[page_num])
                source = "ocr"

        text = normalize_text(text)
        result.append({
            "page": page_num + 1,
            "text": text,
            "source": source,
        })

    return result


def structured_to_plain_text(structured: list[dict], page_sep: str = "\n\n--- Page {page} ---\n\n") -> str:
    """Turn structured page list into one plain string with optional page markers."""
    parts = []
    for p in structured:
        if page_sep:
            parts.append(page_sep.format(page=p["page"]))
        parts.append(p["text"])
    return "".join(parts).strip()


def structured_to_paragraphs(structured: list[dict]) -> list[dict]:
    """Split each page's text into paragraphs. Returns list of {"page", "paragraph_index", "text"}."""
    paragraphs = []
    for p in structured:
        blocks = [b.strip() for b in p["text"].split("\n\n") if b.strip()]
        for i, block in enumerate(blocks):
            paragraphs.append({
                "page": p["page"],
                "paragraph_index": i + 1,
                "text": block,
            })
    return paragraphs


# --- Config from environment ---

_OCR_DIR = Path(__file__).resolve().parent


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip() or default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except (TypeError, ValueError):
        return default


DEFAULT_LLAMA_URL = _env("LLAMA_SERVER_URL", "http://localhost:8080")
OLLAMA_MODEL = _env("OLLAMA_MODEL", "llama")
LLM_TIMEOUT = _env_int("LLM_TIMEOUT", 120)
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 1024)
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.2)

_prompt_file = _env("PROMPT_FILE", "prompt.txt")
DEFAULT_PROMPT_FILE = Path(_prompt_file) if Path(_prompt_file).is_absolute() else _OCR_DIR / _prompt_file

_invoices_summary = _env("INVOICES_SUMMARY_DIR", "invoices_summary")
INVOICES_SUMMARY_DIR = _OCR_DIR / _invoices_summary if not Path(_invoices_summary).is_absolute() else Path(_invoices_summary)

DEFAULT_OCR_DPI = _env_int("OCR_DPI", 200)
DEFAULT_MIN_TEXT_PER_PAGE = _env_int("MIN_TEXT_PER_PAGE", 50)

# Fallback if no prompt file is found
INVOICE_REVIEW_SYSTEM = """You are an expert at reviewing documents. Reply with a single JSON object with keys: "answer" (string), "has_issues" (boolean), "issues" (array of strings)."""


def load_prompt_file(path: str | Path | None = None) -> str:
    """Load system prompt from a text file. Returns fallback INVOICE_REVIEW_SYSTEM if file missing."""
    p = Path(path) if path else DEFAULT_PROMPT_FILE
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    try:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return INVOICE_REVIEW_SYSTEM


def _parse_review_json(content: str) -> tuple[str, dict]:
    """Extract original answer and parsed JSON from model content. Returns (answer_text, json_dict)."""
    raw = content.strip()
    # Try to find JSON in the response (model might wrap in ```json ... ```)
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if json_match:
        raw = json_match.group(1).strip()
    try:
        data = json.loads(raw)
        answer = data.get("answer") or raw
        if not isinstance(data.get("issues"), list):
            data["issues"] = data.get("issues") or []
        if "has_issues" not in data:
            data["has_issues"] = bool(data.get("issues"))
        return answer, data
    except json.JSONDecodeError:
        return raw, {"answer": raw, "has_issues": None, "issues": []}


def ask_llama_invoice_review(
    invoice_text: str,
    base_url: str | None = None,
    timeout: int | None = None,
    max_tokens: int | None = None,
    system_prompt: str | None = None,
) -> dict:
    """
    Send document text to llama.cpp server and return the model's review.

    system_prompt: override for the system message (e.g. from load_prompt_file()). Uses default if None.
    Returns dict with "answer", "json", and optionally "error".
    """
    prompt = (system_prompt or "").strip() or INVOICE_REVIEW_SYSTEM
    base_url = (base_url or DEFAULT_LLAMA_URL).rstrip("/")
    timeout = timeout if timeout is not None else LLM_TIMEOUT
    max_tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Review the following document and reply with the JSON object only.\n\n{invoice_text}"},
        ],
        "max_tokens": max_tokens,
        "temperature": LLM_TEMPERATURE,
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        choice = data.get("choices", [{}])[0]
        content = (choice.get("message", {}) or {}).get("content", "").strip()
        answer, review_json = _parse_review_json(content)
        return {"answer": answer, "json": review_json}
    except requests.exceptions.RequestException as e:
        return {"answer": "", "json": {}, "error": str(e)}
    except (KeyError, IndexError) as e:
        return {"answer": "", "json": {}, "error": f"Unexpected response: {e}"}


def save_run_summary(pdf_path: str | Path, review: dict, out_dir: Path | None = None) -> Path:
    """Save the review summary for a run into invoices_summary directory. Returns path to written file."""
    out_dir = Path(out_dir) if out_dir else INVOICES_SUMMARY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(pdf_path)
    stem = path.stem
    out_file = out_dir / f"{stem}_summary.json"
    data = {
        "source_file": path.name,
        "answer": review.get("answer", ""),
        **review.get("json", {}),
    }
    out_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_file


def extract_and_review_invoice(
    pdf_path: str | Path,
    llama_url: str | None = None,
    use_ocr_fallback: bool = True,
    system_prompt: str | None = None,
) -> tuple[str, dict]:
    """
    Extract text from PDF, send to llama.cpp, return (plain_invoice_text, review_result).

    system_prompt: optional override (e.g. from load_prompt_file()). If None, uses default.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_absolute():
        pdf_path = Path(__file__).resolve().parent / pdf_path
    structured = pdf_to_structured_text(
        pdf_path,
        use_ocr_fallback=use_ocr_fallback and HAS_PDF2IMAGE and HAS_OCR,
        min_text_per_page=DEFAULT_MIN_TEXT_PER_PAGE,
        ocr_dpi=DEFAULT_OCR_DPI,
    )
    plain = structured_to_plain_text(structured)
    if not plain.strip():
        return plain, {"answer": "No text could be extracted from the PDF.", "json": {"answer": "No text.", "has_issues": None, "issues": []}}
    review = ask_llama_invoice_review(plain, base_url=llama_url, system_prompt=system_prompt)
    return plain, review


def run_example(pdf_path: str | Path | None = None, ask_review: bool = True, prompt_file: str | Path | None = None):
    """Extract text from one PDF and optionally send to llama for invoice review."""
    base = Path(__file__).resolve().parent
    path = Path(pdf_path) if pdf_path else base / "Invoice_Summary_17308199.pdf"
    if not path.is_absolute():
        path = base / path

    print(f"Processing: {path.name}\n")
    structured = pdf_to_structured_text(
        path,
        use_ocr_fallback=HAS_PDF2IMAGE and HAS_OCR,
        min_text_per_page=DEFAULT_MIN_TEXT_PER_PAGE,
        ocr_dpi=DEFAULT_OCR_DPI,
    )
    print(f"Pages: {len(structured)}\n")

    plain = structured_to_plain_text(structured)
    print("--- Extracted text (first 1500 chars) ---\n")
    print(plain[:1500] + ("..." if len(plain) > 1500 else ""))
    print("\n---\n")

    if ask_review and plain.strip():
        system_prompt = load_prompt_file(prompt_file)
        if prompt_file:
            print(f"Using prompt file: {prompt_file}\n")
        print("Asking llama.cpp for review...\n")
        review = ask_llama_invoice_review(plain, system_prompt=system_prompt)
        if review.get("error"):
            print("--- Error ---\n", review["error"])
        else:
            print("--- Invoice review (original answer) ---\n")
            print(review["answer"])
            print("\n--- Invoice review (JSON) ---\n")
            print(json.dumps(review["json"], indent=2, ensure_ascii=False))
            summary_path = save_run_summary(path, review)
            print(f"\nSummary saved to: {summary_path}")
        return structured, plain, review
    return structured, plain, None


def run_files(
    file_names: list[str | Path],
    ask_review: bool = True,
    base_dir: Path | None = None,
    prompt_file: str | Path | None = None,
):
    """
    Run extract + optional review on one or more PDF files.
    file_names: list of paths (relative to base_dir or cwd, or absolute).
    prompt_file: path to prompt text file (relative to script dir or absolute). Uses prompt.txt in script dir if None.
    """
    base = base_dir or Path(__file__).resolve().parent
    system_prompt = load_prompt_file(prompt_file) if ask_review else None
    if ask_review and prompt_file:
        print(f"Prompt file: {prompt_file}\n")
    results = []
    for i, name in enumerate(file_names):
        path = Path(name)
        if not path.is_absolute():
            path = base / path
        sep = "\n" + "=" * 60 + "\n"
        if i > 0:
            print(sep)
        print(f"File [{i + 1}/{len(file_names)}]: {path.name}")
        if not path.exists():
            print(f"  Skipped: file not found: {path}")
            results.append({"file": str(path), "error": "file not found"})
            continue
        try:
            plain, review = extract_and_review_invoice(
                path, use_ocr_fallback=HAS_PDF2IMAGE and HAS_OCR, system_prompt=system_prompt
            )
            n_chars = len(plain)
            print(f"  Extracted {n_chars} characters")
            if not ask_review:
                print("  --- Extracted text (first 500 chars) ---")
                print((plain[:500] + ("..." if n_chars > 500 else "")))
                results.append({"file": path.name, "plain_preview": plain[:500], "review": None})
                continue
            if review.get("error"):
                print("  --- Error ---", review["error"])
                results.append({"file": path.name, "error": review["error"], "json": None})
                continue
            print("  --- Review (answer) ---")
            print("  ", review["answer"].replace("\n", "\n  "))
            print("  --- Review (JSON) ---")
            print("  ", json.dumps(review["json"], indent=2, ensure_ascii=False).replace("\n", "\n  "))
            summary_path = save_run_summary(path, review)
            print(f"  Summary saved to: {summary_path}")
            results.append({"file": path.name, "answer": review["answer"], "json": review["json"]})
        except Exception as e:
            print(f"  Skipped: {e}")
            results.append({"file": path.name, "error": str(e)})
    return results


if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    ask_review = "--no-llama" not in argv
    # Parse --prompt <path> (optional)
    prompt_path = None
    args = []
    i = 0
    while i < len(argv):
        if argv[i] == "--prompt" and i + 1 < len(argv):
            prompt_path = argv[i + 1]
            i += 2
            continue
        if argv[i] != "--no-llama":
            args.append(argv[i])
        i += 1
    if not args:
        run_example(ask_review=ask_review, prompt_file=prompt_path)
    else:
        run_files(args, ask_review=ask_review, prompt_file=prompt_path)
