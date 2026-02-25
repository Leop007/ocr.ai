# OCR & Invoice Review

Extract text from PDF invoices and optionally send it to a llama.cpp server for structured review. Supports both digital text and OCR for scanned pages.

## Requirements

```bash
pip install -r requirements.txt
```

For scanned PDFs you also need **poppler** (e.g. `sudo apt install poppler-utils` on Ubuntu).

## Usage

### Run the built-in example (one default PDF)

```bash
python main.py
```

Uses a default PDF in the `ocr` directory and sends the extracted text to the llama server for review.

### Process one or more PDFs

```bash
python main.py invoice1.pdf invoice2.pdf
python main.py path/to/invoice.pdf
```

Paths are relative to the `ocr` directory (or use absolute paths). Each file is extracted and reviewed; summaries are saved under `invoices_summary/`.

### Options

| Option | Description |
|--------|-------------|
| `--ocr-only` | OCR only: extract text with Tesseract (and digital extraction); no LLM review. Same as `--no-llama`. |
| `--no-llama` | Same as `--ocr-only`: extraction only, no LLM. |
| `--force-ocr` | Always use OCR on every page (for scanned PDFs). Uses 300 DPI. |
| `--ocr=<method>` | Extraction method: `tesseract` (default) or `llm`. If `llm`, PDF pages are sent to a vision LLM instead of Tesseract (no Tesseract required). |
| `--llm=<provider>` | When using LLM: `auto` (default), `local` (llama.cpp / LM Studio), `ollama` (Ollama), or `gemini` (Google AI). |
| `--retry=<n>` | Number of retries on timeout/server error/empty response (default: 2). |
| `--retry_interval=<n>` | Seconds to wait between retries (default: 2). |
| `--prompt <path>` | Use a custom system prompt from a text file (default: `prompt.txt` in this directory). |

**Examples:**

```bash
# OCR only (extract text, no LLM review)
python main.py --ocr-only my_invoice.pdf
# or: python main.py --no-llama my_invoice.pdf

# OCR only with LLM (vision): extract text via LLM instead of Tesseract; no review
python main.py --ocr-only --ocr=llm my_invoice.pdf
# Pick LLM: python main.py --ocr-only --ocr=llm --llm=gemini my_invoice.pdf

# Use Google Gemini (requires GEMINI_API_KEY in .env)
python main.py --llm=gemini invoice.pdf

# Use Ollama (native API, default localhost:11434)
python main.py --llm=ollama invoice.pdf

# Use local llama.cpp (OpenAI-compatible server)
python main.py --llm=local invoice.pdf

# Retry up to 4 times with 5s delay between attempts (useful for flaky local LLM)
python main.py --retry=4 --retry_interval=5 invoice.pdf

# Use a custom prompt file
python main.py --prompt my_prompt.txt invoice.pdf
```

### Local vs remote LLM

- **OCR** is always done by **Tesseract** (local).
- **Review** (and optional amount extraction) uses either:
  - **Local:** `--llm=ollama` (Ollama) or `--llm=local` (llama.cpp / LM Studio).
  - **Remote:** `--llm=gemini` (Google Gemini; requires `GEMINI_API_KEY`).

Use `python app.py` to pick local (Ollama / llama.cpp) or remote (Gemini) interactively.

### OCR only with LLM (no Tesseract)

You can use a **vision-capable LLM** to extract text from PDF pages instead of Tesseract. Use `--ocr=llm` (and optionally `--ocr-only` for extraction only, no review). The same LLM providers are supported (Gemini, Ollama, LM Studio / llama.cpp with a vision model). Requires `pdf2image` and poppler; Tesseract is not required when using `--ocr=llm`. In `python app.py`, choose option **2. OCR only with LLM**.

### Number comparison (Tesseract vs LLM)

Amounts are extracted from the Tesseract OCR text and, when the LLM returns a `numbers` array (see `prompt.txt`), compared with the LLM’s amounts. The result is in each run summary as `number_comparison`: `ocr_numbers`, `llm_numbers`, `match`, and `mismatches`. Summaries are written to `invoices_summary/<stem>_summary.json`.

## Output

- **Console:** Extracted text preview, review answer + JSON, and (when available) number comparison (OCR vs LLM).

### Re-runs with cached OCR

When `OCR_OUTPUT_DIR` is set and an extracted text file already exists (`ocr_output/<pdf_stem>_extracted.txt`), the script skips PDF extraction and runs only the LLM on the cached text. The summary is saved as `<stem>_summary_llm_rerun.json` to distinguish it from the original run.

### Run log (XLSX)

Each script run creates a **new** timestamped log file in the `log_files/` directory (e.g. `log_files/invoices_run_log_2025-02-12_143052.xlsx`). Columns: Source file, Start timestamp, OCR time (s), Summary time (s), Who processed, OCR output, Invoice summary. All files processed in that run are written to that single log file.

### Gemini fallback

When the local LLM (Ollama, llama.cpp, LM Studio) fails, and `GEMINI_API_KEY` is set, the script automatically retries with Google Gemini using the same OCR text.

- **`invoices_summary/`:** For each reviewed PDF, a JSON file is written:  
  `invoices_summary/<pdf_stem>_summary.json`  
  Contents: `source_file`, `answer`, review fields (`has_issues`, `issues`, `flags`), optional `numbers`, and `number_comparison` (OCR vs LLM amounts).  
  The directory is created automatically.

## Configuration (.env)

Copy `.env.example` to `.env` in the `ocr` directory and adjust as needed. The script loads `.env` on startup. Variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key (get at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)) | — |
| `GEMINI_MODEL` | Gemini model name | `gemini-2.5-flash` |
| `LLAMA_SERVER_URL` | Base URL for llama.cpp (OpenAI-compatible) | `http://localhost:8080` |
| `OLLAMA_URL` | Ollama server URL (for `--llm=ollama`) | `http://localhost:11434` |
| `OLLAMA_MODEL` | Model name for Ollama / local (e.g. `llama3.2`, `qwen2.5`) | `llama` |
| `LLM_TIMEOUT` | Request timeout (seconds) | `120` |
| `LLM_MAX_TOKENS` | Max tokens for local model (Ollama/llama.cpp) response | `2048` |
| `GEMINI_MAX_TOKENS` | Max tokens for Gemini response | `4096` |
| `LLM_TEMPERATURE` | Sampling temperature (local) | `0.2` |
| `INVOICES_SUMMARY_DIR` | Where to save run summaries (relative or absolute) | `invoices_summary` |
| `PROMPT_FILE` | Path to system prompt file (relative or absolute) | `prompt.txt` |
| `INVOICES_RUN_LOG` | Base name for run logs; each run creates `log_files/{stem}_{YYYY-MM-DD_HHMMSS}.xlsx` | `invoices_run_log.xlsx` |
| `OCR_OUTPUT_DIR` | Directory to save extracted text for inspection (set empty to disable) | `ocr_output` |
| `INVOICES_OCR_DIR` | Directory for scanned-only PDFs; files here always use OCR (300 DPI) | — |
| `OCR_DPI` | DPI for OCR fallback mode | `200` |
| `OCR_DPI_FORCED` | DPI when forcing OCR (`--force-ocr` or `INVOICES_OCR_DIR`) | `300` |
| `OCR_CONFIG` | Tesseract config (e.g. `--psm 6` for block text) | `--psm 6` |
| `OCR_LANG` | Tesseract language code (e.g. `eng`, `fra`, `deu`). Use `+` for multiple: `eng+fra` | `eng` |
| `MIN_TEXT_PER_PAGE` | Min characters per page below which OCR fallback is used | `50` |

For **Ollama**, use `--llm=ollama` and set `OLLAMA_URL=http://localhost:11434` (default), `OLLAMA_MODEL` to your model (e.g. `llama3.2`, `qwen2.5`). For **llama.cpp**, use `--llm=local` and set `LLAMA_SERVER_URL` (e.g. `http://localhost:1234`).

## Prompt

The default system prompt is read from `prompt.txt` in this directory. It instructs the model to return a JSON object with `answer`, `has_issues`, and `issues`. Override with `--prompt <path>`.
