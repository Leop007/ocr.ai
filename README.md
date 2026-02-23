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
| `--no-llama` | Only extract text; do not call the LLM for review. |
| `--force-ocr` | Always use OCR on every page (for scanned PDFs). Uses 300 DPI. |
| `--llm=<provider>` | LLM provider: `auto` (default), `local` (llama.cpp), `ollama` (Ollama native), or `gemini` (Google AI). |
| `--retry=<n>` | Number of retries on timeout/server error/empty response (default: 2). |
| `--retry_interval=<n>` | Seconds to wait between retries (default: 2). |
| `--prompt <path>` | Use a custom system prompt from a text file (default: `prompt.txt` in this directory). |

**Examples:**

```bash
# Extract text only, no LLM review
python main.py --no-llama my_invoice.pdf

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

## Output

- **Console:** Extracted text preview and the review answer + JSON for each file.

### Re-runs with cached OCR

When `OCR_OUTPUT_DIR` is set and an extracted text file already exists (`ocr_output/<pdf_stem>_extracted.txt`), the script skips PDF extraction and runs only the LLM on the cached text. The summary is saved as `<stem>_summary_llm_rerun.json` to distinguish it from the original run.

### Run log (XLSX)

Each processed file is appended to `invoices_run_log.xlsx` (or `INVOICES_RUN_LOG`), with columns: Source file, Start timestamp, OCR time (s), Summary time (s), Who processed (Ollama/LLM Studio/Gemini), OCR output, Invoice summary. Every run adds new rows.

### Gemini fallback

When the local LLM (Ollama, llama.cpp, LM Studio) fails, and `GEMINI_API_KEY` is set, the script automatically retries with Google Gemini using the same OCR text.
- **`invoices_summary/`:** For each reviewed PDF, a JSON file is written:  
  `invoices_summary/<pdf_stem>_summary.json`  
  Contents: `source_file`, `answer`, and the review fields (`has_issues`, `issues`, etc.).  
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
| `INVOICES_RUN_LOG` | XLSX file to append run logs (per-file: timestamp, processing time, OCR output, summary) | `invoices_run_log.xlsx` |
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
