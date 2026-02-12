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
| `--no-llama` | Only extract text; do not call the llama server for review. |
| `--prompt <path>` | Use a custom system prompt from a text file (default: `prompt.txt` in this directory). |

**Examples:**

```bash
# Extract text only, no LLM review
python main.py --no-llama my_invoice.pdf

# Use a custom prompt file
python main.py --prompt my_prompt.txt invoice.pdf
```

## Output

- **Console:** Extracted text preview and the review answer + JSON for each file.
- **`invoices_summary/`:** For each reviewed PDF, a JSON file is written:  
  `invoices_summary/<pdf_stem>_summary.json`  
  Contents: `source_file`, `answer`, and the review fields (`has_issues`, `issues`, etc.).  
  The directory is created automatically.

## Configuration (.env)

Copy `.env.example` to `.env` in the `ocr` directory and adjust as needed. The script loads `.env` on startup. Variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMA_SERVER_URL` | Base URL of the LLM server (Ollama, llama.cpp, etc.) | `http://localhost:8080` |
| `OLLAMA_MODEL` | Model name (must exist on your server) | `llama` |
| `LLM_TIMEOUT` | Request timeout (seconds) | `120` |
| `LLM_MAX_TOKENS` | Max tokens in the model response | `1024` |
| `LLM_TEMPERATURE` | Sampling temperature | `0.2` |
| `INVOICES_SUMMARY_DIR` | Where to save run summaries (relative or absolute) | `invoices_summary` |
| `PROMPT_FILE` | Path to system prompt file (relative or absolute) | `prompt.txt` |
| `OCR_DPI` | DPI for rendering PDF pages when using OCR | `200` |
| `MIN_TEXT_PER_PAGE` | Min characters per page below which OCR is used | `50` |

For **Ollama**, set `LLAMA_SERVER_URL=http://localhost:11434` and `OLLAMA_MODEL` to your model (e.g. `llama3.2`, `qwen2.5`).

## Prompt

The default system prompt is read from `prompt.txt` in this directory. It instructs the model to return a JSON object with `answer`, `has_issues`, and `issues`. Override with `--prompt <path>`.
