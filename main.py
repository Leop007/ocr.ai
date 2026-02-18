"""
Extract text from PDF, structure it, and optionally send to llama.cpp for invoice review.
"""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def extract_text_via_ocr(image, config: str | None = None) -> str:
    """Get text from a page image using Tesseract. config: optional Tesseract args (e.g. '--psm 6')."""
    if not HAS_OCR:
        return ""
    try:
        return pytesseract.image_to_string(image, config=config or "") or ""
    except Exception:
        return ""


def pdf_to_structured_text(
    pdf_path: str | Path,
    use_ocr_fallback: bool = True,
    min_text_per_page: int = 50,
    ocr_dpi: int = 200,
    force_ocr: bool = False,
    ocr_config: str | None = None,
    ocr_workers: int | None = None,
) -> list[dict]:
    """
    Extract text from each page. Structure: list of {"page": 1-based index, "text": "...", "source": "text"|"ocr"}.

    - use_ocr_fallback: if True, run OCR on pages where extracted text is short (e.g. scanned).
    - min_text_per_page: if extracted text has fewer than this many chars, treat as scanned and use OCR.
    - ocr_dpi: DPI when rendering pages to images for OCR (higher = better quality, slower).
    - force_ocr: if True, always use OCR on every page (for scanned-only PDFs). Ignores digital text.
    - ocr_config: optional Tesseract config (e.g. "--psm 6" for block text).
    - ocr_workers: max workers for parallel page OCR (default: min(4, pages)). 1 = sequential.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if force_ocr:
        min_text_per_page = 999999
    tesseract_config = ocr_config or DEFAULT_OCR_CONFIG

    reader = PyPDF2.PdfReader(str(pdf_path))
    n_pages = len(reader.pages)

    # Optional: render all pages to images once (for OCR fallback)
    page_images = None
    if use_ocr_fallback and HAS_PDF2IMAGE and HAS_OCR:
        try:
            page_images = convert_from_path(str(pdf_path), dpi=ocr_dpi)
        except Exception:
            page_images = None

    # First pass: get digital text, identify pages needing OCR
    prelim = []
    ocr_jobs = []
    for page_num in range(n_pages):
        page = reader.pages[page_num]
        text = extract_text_from_page(page)
        needs_ocr = use_ocr_fallback and len(text.strip()) < min_text_per_page
        if needs_ocr and page_images is not None and page_num < len(page_images):
            ocr_jobs.append((page_num + 1, page_images[page_num], text))
        else:
            prelim.append((page_num + 1, normalize_text(text), "ocr" if needs_ocr else "text"))

    # Run OCR on needed pages (parallel if multiple)
    if not ocr_jobs:
        return [{"page": p, "text": t, "source": s} for p, t, s in sorted(prelim, key=lambda x: x[0])]
    if len(ocr_jobs) == 1 or (ocr_workers or 1) <= 1:
        result = [(p, normalize_text(extract_text_via_ocr(img, config=tesseract_config)), "ocr") for p, img, _ in ocr_jobs]
    else:
        workers = ocr_workers or min(4, len(ocr_jobs))
        result = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut = {ex.submit(extract_text_via_ocr, img, tesseract_config): (p,) for p, img, _ in ocr_jobs}
            for f in as_completed(fut):
                p = fut[f][0]
                text = normalize_text(f.result())
                result.append((p, text, "ocr"))
    # Merge and sort by page number
    merged = {p: (p, t, s) for p, t, s in prelim}
    for p, text, src in result:
        merged[p] = (p, text, src)
    return [{"page": p, "text": t, "source": s} for p, t, s in sorted(merged.values(), key=lambda x: x[0])]


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


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off", ""):
        return False
    return default


GEMINI_API_KEY = _env("GEMINI_API_KEY", "")
GEMINI_BETA = _env_bool("GEMINI_BETA", False)
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-2.5-flash")

DEFAULT_LLAMA_URL = _env("LLAMA_SERVER_URL", "http://localhost:8080")
OLLAMA_MODEL = _env("OLLAMA_MODEL", "llama")
LLM_TIMEOUT = _env_int("LLM_TIMEOUT", 120)
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 2048)
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.2)

GEMINI_MAX_TOKENS = _env_int("GEMINI_MAX_TOKENS", 4096)
GEMINI_TEMPERATURE = _env_float("GEMINI_TEMPERATURE", LLM_TEMPERATURE)

_prompt_file = _env("PROMPT_FILE", "prompt.txt")
DEFAULT_PROMPT_FILE = Path(_prompt_file) if Path(_prompt_file).is_absolute() else _OCR_DIR / _prompt_file

_invoices_summary = _env("INVOICES_SUMMARY_DIR", "invoices_summary")
INVOICES_SUMMARY_DIR = _OCR_DIR / _invoices_summary if not Path(_invoices_summary).is_absolute() else Path(_invoices_summary)

DEFAULT_OCR_DPI = _env_int("OCR_DPI", 200)
DEFAULT_OCR_DPI_FORCED = _env_int("OCR_DPI_FORCED", 300)
DEFAULT_MIN_TEXT_PER_PAGE = _env_int("MIN_TEXT_PER_PAGE", 50)
DEFAULT_OCR_CONFIG = _env("OCR_CONFIG", "--psm 6")

_invoices_ocr_dir = _env("INVOICES_OCR_DIR", "")
INVOICES_OCR_DIR = None
if _invoices_ocr_dir.strip():
    p = Path(_invoices_ocr_dir.strip())
    INVOICES_OCR_DIR = p if p.is_absolute() else _OCR_DIR / p

_ocr_output_dir = _env("OCR_OUTPUT_DIR", "ocr_output")
OCR_OUTPUT_DIR = None
if _ocr_output_dir.strip():
    p = Path(_ocr_output_dir.strip())
    OCR_OUTPUT_DIR = p if p.is_absolute() else _OCR_DIR / p

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


def _fill_from_truncated_inner(answer_str: str, data: dict) -> str | None:
    """When inner JSON fails to parse (truncated/malformed), extract fields via regex. Returns extracted answer or None."""
    if not answer_str or not isinstance(answer_str, str):
        return None
    extracted_answer = None
    # Extract has_issues: true/false
    m = re.search(r'"has_issues"\s*:\s*(true|false)', answer_str, re.I)
    if m:
        data["has_issues"] = m.group(1).lower() == "true"
    # Extract inner answer text ("answer": "..." - handles escaped quotes)
    m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', answer_str)
    if m:
        inner_answer = m.group(1).replace('\\"', '"').replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
        extracted_answer = _clean_answer(inner_answer)
    # Extract issues array (simplified: look for "issues":[ ... ])
    m = re.search(r'"issues"\s*:\s*\[(.*?)\]', answer_str, re.DOTALL)
    if m:
        try:
            arr = json.loads("[" + m.group(1) + "]")
            if isinstance(arr, list):
                data["issues"] = [_clean_answer(str(i)) for i in arr]
        except json.JSONDecodeError:
            pass
    # If has_issues still None, default to False when we have at least partial parse
    if data.get("has_issues") is None and ("has_issues" in answer_str or '"answer"' in answer_str):
        data["has_issues"] = bool(data.get("issues"))
    return extracted_answer

def _normalize_flags(flags) -> str:
    """Convert flags to a single string (green/orange/red). Handles emoji and duplicate arrays."""
    if flags is None:
        return "green"
    if isinstance(flags, list):
        # Take worst: red > orange > green. Dedupe and pick one.
        s = set(str(f).strip().lower() for f in flags if f)
        if "red" in s:
            return "red"
        if "orange" in s:
            return "orange"
        if "green" in s:
            return "green"
        return flags[0] if flags else "green"
    s = str(flags).strip().lower()
    # Map emoji to words (🟢🟠🔴 and similar)
    if "🟢" in str(flags) or "🟩" in str(flags) or s == "green":
        return "green"
    if "🟠" in str(flags) or "🟧" in str(flags) or s == "orange":
        return "orange"
    if "🔴" in str(flags) or "🟥" in str(flags) or s == "red":
        return "red"
    return s if s in ("green", "orange", "red") else "green"


def _clean_answer(text: str) -> str:
    """Remove escape sequences and normalize whitespace for a clean answer string."""
    if not text or not isinstance(text, str):
        return str(text) if text else ""
    # Replace literal escape sequences (backslash-n, backslash-t, etc.) with spaces
    s = text.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    s = s.replace('\\"', '"')
    # Replace actual newlines and tabs with space
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # Collapse multiple spaces and strip
    s = re.sub(r" +", " ", s).strip()
    return s


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
        # Handle double-encoded JSON (Gemini sometimes puts JSON string inside "answer")
        if isinstance(answer, str) and answer.strip().startswith("{"):
            try:
                inner = json.loads(answer)
                answer = inner.get("answer", answer)
                if isinstance(inner.get("issues"), list):
                    data["issues"] = inner["issues"]
                if "has_issues" in inner:
                    data["has_issues"] = inner["has_issues"]
                if "flags" in inner:
                    data["flags"] = inner["flags"]
            except json.JSONDecodeError:
                # Inner JSON truncated/malformed - try regex to extract has_issues and answer text
                extracted = _fill_from_truncated_inner(answer, data)
                if extracted is not None:
                    answer = extracted
        if not isinstance(data.get("issues"), list):
            data["issues"] = data.get("issues") or []
        if data.get("has_issues") is None:
            data["has_issues"] = bool(data.get("issues"))
        data["answer"] = _clean_answer(answer)
        data["issues"] = [_clean_answer(str(i)) for i in data["issues"]]
        data["flags"] = _normalize_flags(data.get("flags"))
        return data["answer"], data
    except json.JSONDecodeError:
        return _clean_answer(raw), {"answer": _clean_answer(raw), "has_issues": None, "issues": [], "flags": "green"}


def ask_gemini_invoice_review(
    invoice_text: str,
    api_key: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
    max_tokens: int | None = None,
    system_prompt: str | None = None,
) -> dict:
    """
    Send document text to Google Gemini API and return the model's review.

    api_key: Gemini API key (uses GEMINI_API_KEY from env if not provided).
    Returns dict with "answer", "json", and optionally "error".
    """
    key = (api_key or GEMINI_API_KEY).strip()
    if not key:
        return {"answer": "", "json": {}, "error": "GEMINI_API_KEY is not set"}

    prompt = (system_prompt or "").strip() or INVOICE_REVIEW_SYSTEM
    model = model or GEMINI_MODEL
    timeout = timeout if timeout is not None else LLM_TIMEOUT
    max_tokens = max_tokens if max_tokens is not None else GEMINI_MAX_TOKENS

    api_version = "v1beta" if GEMINI_BETA else "v1"
    url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:generateContent"
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    user_message = (
        f"{prompt}\n\n"
        f"Review the following document. Reply with ONLY a single JSON object, no other text before or after.\n\n"
        f"{invoice_text}"
    )
    generation_config = {
        "maxOutputTokens": max_tokens,
        "temperature": GEMINI_TEMPERATURE,
    }
    # Enable Deep Think reasoning for Gemini 3 models
    if "gemini-3" in (model or "").lower():
        generation_config["thinkingConfig"] = {"thinkingLevel": "HIGH"}
    payload = {
        "contents": [{"parts": [{"text": user_message}]}],
        "generationConfig": generation_config,
    }
    if GEMINI_BETA:
        payload["systemInstruction"] = {"parts": [{"text": prompt}]}

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        parts = (data.get("candidates", [{}])[0].get("content", {}).get("parts", []))
        content = (parts[0].get("text", "") if parts else "").strip()
        answer, review_json = _parse_review_json(content)
        return {"answer": answer, "json": review_json, "llm_provider": "gemini", "llm_model": model}
    except requests.exceptions.RequestException as e:
        return {"answer": "", "json": {}, "error": str(e), "llm_provider": "gemini", "llm_model": model}
    except (KeyError, IndexError) as e:
        return {"answer": "", "json": {}, "error": f"Unexpected response: {e}", "llm_provider": "gemini", "llm_model": model}


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
        return {"answer": answer, "json": review_json, "llm_provider": "local", "llm_model": OLLAMA_MODEL}
    except requests.exceptions.RequestException as e:
        return {"answer": "", "json": {}, "error": str(e), "llm_provider": "local", "llm_model": OLLAMA_MODEL}
    except (KeyError, IndexError) as e:
        return {"answer": "", "json": {}, "error": f"Unexpected response: {e}", "llm_provider": "local", "llm_model": OLLAMA_MODEL}


LLM_PROVIDERS = ("auto", "local", "gemini")


DEFAULT_RETRY = 2
DEFAULT_RETRY_INTERVAL = 2


def ask_invoice_review(
    invoice_text: str,
    base_url: str | None = None,
    timeout: int | None = None,
    max_tokens: int | None = None,
    system_prompt: str | None = None,
    llm: str = "auto",
    retry: int = DEFAULT_RETRY,
    retry_interval: int = DEFAULT_RETRY_INTERVAL,
) -> dict:
    """
    Send document text to AI for invoice review.

    llm: "auto" (use Gemini if GEMINI_API_KEY set, else local), "local" (Ollama/llama.cpp),
         "gemini" (Google Gemini).
    retry: number of retries on timeout/server error/empty response (default 2).
    retry_interval: seconds to wait between retries (default 2).
    """
    use_gemini = (
        (llm == "gemini")
        or (llm == "auto" and GEMINI_API_KEY.strip())
    )
    if use_gemini:
        if not GEMINI_API_KEY.strip():
            return {"answer": "", "json": {}, "error": "GEMINI_API_KEY is not set. Add it to .env or use --llm=local", "llm_provider": "gemini", "llm_model": GEMINI_MODEL}

    def _call() -> dict:
        if use_gemini:
            return ask_gemini_invoice_review(invoice_text, timeout=timeout, max_tokens=max_tokens, system_prompt=system_prompt)
        return ask_llama_invoice_review(invoice_text, base_url=base_url, timeout=timeout, max_tokens=max_tokens, system_prompt=system_prompt)

    last_review = None
    for attempt in range(1 + retry):
        last_review = _call()
        has_error = last_review.get("error")
        has_answer = last_review.get("answer") or (last_review.get("json") or {}).get("answer")
        if not has_error and has_answer:
            return last_review
        if attempt < retry:
            time.sleep(retry_interval)
    return last_review or {"answer": "", "json": {}, "error": "All retries exhausted", "llm_provider": "local", "llm_model": ""}


def save_run_summary(pdf_path: str | Path, review: dict, out_dir: Path | None = None) -> Path:
    """Save the review summary for a run into invoices_summary directory. Returns path to written file."""
    out_dir = Path(out_dir) if out_dir else INVOICES_SUMMARY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(pdf_path)
    stem = path.stem
    out_file = out_dir / f"{stem}_summary.json"
    data = {
        "source_file": path.name,
        "llm_provider": review.get("llm_provider", ""),
        "llm_model": review.get("llm_model", ""),
        "processing_time_seconds": review.get("processing_time_seconds"),
        "answer": review.get("answer", ""),
        **review.get("json", {}),
    }
    out_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_file


def save_extracted_text(pdf_path: str | Path, plain_text: str, structured: list[dict] | None = None, out_dir: Path | None = None) -> Path | None:
    """Save extracted/OCR text to OCR_OUTPUT_DIR. Returns path or None if disabled."""
    if not OCR_OUTPUT_DIR:
        return None
    out = Path(out_dir) if out_dir else OCR_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(pdf_path).stem
    txt_path = out / f"{stem}_extracted.txt"
    header = f"# Extracted from {Path(pdf_path).name}\n"
    if structured:
        header += "# Per-page: " + ", ".join(f"p{p['page']}={p['source']}" for p in structured) + "\n"
    txt_path.write_text(header + "\n" + plain_text, encoding="utf-8")
    return txt_path


def _is_in_ocr_dir(pdf_path: Path) -> bool:
    """True if pdf_path is under INVOICES_OCR_DIR (scanned docs, force OCR)."""
    if not INVOICES_OCR_DIR:
        return False
    try:
        pdf_resolved = Path(pdf_path).resolve()
        ocr_resolved = INVOICES_OCR_DIR.resolve()
        pdf_resolved.relative_to(ocr_resolved)
        return True
    except ValueError:
        return False


def extract_and_review_invoice(
    pdf_path: str | Path,
    llama_url: str | None = None,
    use_ocr_fallback: bool = True,
    system_prompt: str | None = None,
    llm: str = "auto",
    retry: int = DEFAULT_RETRY,
    retry_interval: int = DEFAULT_RETRY_INTERVAL,
    force_ocr: bool = False,
) -> tuple[str, dict]:
    """
    Extract text from PDF, send to AI, return (plain_invoice_text, review_result).

    llm: "auto", "local", or "gemini".
    force_ocr: always use OCR on every page (for scanned PDFs).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_absolute():
        pdf_path = Path(__file__).resolve().parent / pdf_path
    force = force_ocr or _is_in_ocr_dir(pdf_path)
    ocr_dpi = DEFAULT_OCR_DPI_FORCED if force else DEFAULT_OCR_DPI
    structured = pdf_to_structured_text(
        pdf_path,
        use_ocr_fallback=use_ocr_fallback and HAS_PDF2IMAGE and HAS_OCR,
        min_text_per_page=DEFAULT_MIN_TEXT_PER_PAGE,
        ocr_dpi=ocr_dpi,
        force_ocr=force,
    )
    plain = structured_to_plain_text(structured)
    if OCR_OUTPUT_DIR:
        save_extracted_text(pdf_path, plain, structured=structured)
    if not plain.strip():
        return plain, {"answer": "No text could be extracted from the PDF.", "json": {"answer": "No text.", "has_issues": None, "issues": []}, "llm_provider": "", "llm_model": "", "processing_time_seconds": None}
    t0 = time.perf_counter()
    review = ask_invoice_review(plain, base_url=llama_url, system_prompt=system_prompt, llm=llm, retry=retry, retry_interval=retry_interval)
    review["processing_time_seconds"] = round(time.perf_counter() - t0, 2)
    return plain, review


def run_example(pdf_path: str | Path | None = None, ask_review: bool = True, prompt_file: str | Path | None = None, llm: str = "auto", retry: int = DEFAULT_RETRY, retry_interval: int = DEFAULT_RETRY_INTERVAL, force_ocr: bool = False):
    """Extract text from one PDF and optionally send to llama for invoice review."""
    base = Path(__file__).resolve().parent
    path = Path(pdf_path) if pdf_path else base / "Invoice_Summary_17308199.pdf"
    if not path.is_absolute():
        path = base / path

    print(f"Processing: {path.name}\n")
    force = force_ocr or _is_in_ocr_dir(path)
    ocr_dpi = DEFAULT_OCR_DPI_FORCED if force else DEFAULT_OCR_DPI
    structured = pdf_to_structured_text(
        path,
        use_ocr_fallback=HAS_PDF2IMAGE and HAS_OCR,
        min_text_per_page=DEFAULT_MIN_TEXT_PER_PAGE,
        ocr_dpi=ocr_dpi,
        force_ocr=force,
    )
    print(f"Pages: {len(structured)}\n")

    plain = structured_to_plain_text(structured)
    if OCR_OUTPUT_DIR:
        save_extracted_text(path, plain, structured=structured)
        print(f"Extracted text saved to {OCR_OUTPUT_DIR}/\n")
    print("--- Extracted text (first 1500 chars) ---\n")
    print(plain[:1500] + ("..." if len(plain) > 1500 else ""))
    print("\n---\n")

    if ask_review and plain.strip():
        system_prompt = load_prompt_file(prompt_file)
        if prompt_file:
            print(f"Using prompt file: {prompt_file}\n")
        print(f"Asking AI for review (llm={llm})...\n")
        t0 = time.perf_counter()
        review = ask_invoice_review(plain, system_prompt=system_prompt, llm=llm, retry=retry, retry_interval=retry_interval)
        review["processing_time_seconds"] = round(time.perf_counter() - t0, 2)
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


def _extract_pdf_only(
    path: Path,
    use_ocr_fallback: bool = True,
    force_ocr: bool = False,
) -> tuple[str, list[dict]]:
    """Extract text from PDF (no LLM). Returns (plain_text, structured)."""
    force = force_ocr or _is_in_ocr_dir(path)
    ocr_dpi = DEFAULT_OCR_DPI_FORCED if force else DEFAULT_OCR_DPI
    structured = pdf_to_structured_text(
        path,
        use_ocr_fallback=use_ocr_fallback,
        min_text_per_page=DEFAULT_MIN_TEXT_PER_PAGE,
        ocr_dpi=ocr_dpi,
        force_ocr=force,
    )
    return structured_to_plain_text(structured), structured


def run_files(
    file_names: list[str | Path],
    ask_review: bool = True,
    base_dir: Path | None = None,
    prompt_file: str | Path | None = None,
    llm: str = "auto",
    retry: int = DEFAULT_RETRY,
    retry_interval: int = DEFAULT_RETRY_INTERVAL,
    force_ocr: bool = False,
    overlap_extract_llm: bool = True,
):
    """
    Run extract + optional review on one or more PDF files.
    file_names: list of paths (relative to base_dir or cwd, or absolute).
    prompt_file: path to prompt text file (relative to script dir or absolute). Uses prompt.txt in script dir if None.
    overlap_extract_llm: when True and processing multiple files, extract next PDF while LLM reviews current.
    """
    base = base_dir or Path(__file__).resolve().parent
    system_prompt = load_prompt_file(prompt_file) if ask_review else None
    use_ocr = HAS_PDF2IMAGE and HAS_OCR
    if ask_review and prompt_file:
        print(f"Prompt file: {prompt_file}\n")
    results = []
    next_extract_future = None
    with ThreadPoolExecutor(max_workers=2) as ex:
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
                # Get extraction (from previous overlap or do now)
                if next_extract_future is not None:
                    plain, structured = next_extract_future.result()
                else:
                    plain, structured = _extract_pdf_only(path, use_ocr_fallback=use_ocr, force_ocr=force_ocr)
                next_extract_future = None
                # Kick off extraction for next file while we run LLM (if multiple files, ask_review)
                if overlap_extract_llm and ask_review and i + 1 < len(file_names):
                    next_path = Path(file_names[i + 1])
                    if not next_path.is_absolute():
                        next_path = base / next_path
                    if next_path.exists():
                        next_extract_future = ex.submit(
                            _extract_pdf_only, next_path, use_ocr, force_ocr
                        )
                if OCR_OUTPUT_DIR:
                    txt_path = save_extracted_text(path, plain, structured=structured)
                    if txt_path:
                        print(f"  Extracted text saved to {txt_path}")
                n_chars = len(plain)
                print(f"  Extracted {n_chars} characters")
                if not plain.strip():
                    plain, review = plain, {"answer": "No text could be extracted from the PDF.", "json": {"answer": "No text.", "has_issues": None, "issues": []}, "llm_provider": "", "llm_model": "", "processing_time_seconds": None}
                elif not ask_review:
                    print("  --- Extracted text (first 500 chars) ---")
                    print((plain[:500] + ("..." if n_chars > 500 else "")))
                    results.append({"file": path.name, "plain_preview": plain[:500], "review": None})
                    continue
                else:
                    t0 = time.perf_counter()
                    review = ask_invoice_review(plain, system_prompt=system_prompt, llm=llm, retry=retry, retry_interval=retry_interval)
                    review["processing_time_seconds"] = round(time.perf_counter() - t0, 2)
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


def _parse_positive_int(val: str, default: int, name: str) -> int:
    try:
        n = int(val)
        if n >= 0:
            return n
    except (TypeError, ValueError):
        pass
    return default


if __name__ == "__main__":
    import sys

    def _main() -> None:
        argv = sys.argv[1:]
        ask_review = "--no-llama" not in argv
        prompt_path = None
        llm_provider = "auto"
        retry = DEFAULT_RETRY
        retry_interval = DEFAULT_RETRY_INTERVAL
        force_ocr = False
        args = []
        i = 0
        while i < len(argv):
            if argv[i] == "--prompt" and i + 1 < len(argv):
                prompt_path = argv[i + 1]
                i += 2
                continue
            if argv[i].startswith("--llm="):
                llm_provider = argv[i].split("=", 1)[1].strip().lower()
                if llm_provider not in LLM_PROVIDERS:
                    print(f"Unknown --llm= value: {llm_provider}. Use: {', '.join(LLM_PROVIDERS)}")
                    sys.exit(1)
                i += 1
                continue
            if argv[i].startswith("--retry="):
                retry = _parse_positive_int(argv[i].split("=", 1)[1].strip(), DEFAULT_RETRY, "retry")
                i += 1
                continue
            if argv[i].startswith("--retry_interval="):
                retry_interval = _parse_positive_int(argv[i].split("=", 1)[1].strip(), DEFAULT_RETRY_INTERVAL, "retry_interval")
                i += 1
                continue
            if argv[i] == "--force-ocr":
                force_ocr = True
                i += 1
                continue
            if argv[i] != "--no-llama":
                args.append(argv[i])
            i += 1
        if not args:
            run_example(ask_review=ask_review, prompt_file=prompt_path, llm=llm_provider, retry=retry, retry_interval=retry_interval, force_ocr=force_ocr)
        else:
            # Expand directories to their PDF files so multiple files are processed
            base = Path(__file__).resolve().parent
            expanded = []
            for a in args:
                p = Path(a) if Path(a).is_absolute() else base / a
                if p.is_dir():
                    expanded.extend(sorted(p.glob("*.pdf")))
                else:
                    expanded.append(p)
            if expanded:
                run_files(expanded, ask_review=ask_review, prompt_file=prompt_path, llm=llm_provider, retry=retry, retry_interval=retry_interval, force_ocr=force_ocr)
            else:
                print("No PDF files found.")

    try:
        _main()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting cleanly.")
        sys.exit(0)
