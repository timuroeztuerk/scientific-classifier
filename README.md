# scientific-classifier

A script that uses local/cloud LLM to classify bi and trigrams as "scientific" or "not scientific". It reads terms from a txt file, deduplicates them, and appends classification results to a CSV file. 

## Install
You will need a normal python environment with ollama installed from pip. I used mistral, but you can opt with any.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull mistral   # or another model; ensure it's available locally
```

## Run
```bash
python classifier.py
```
Progress bar shows classification status. You can stop and re-run; existing IDs are skipped (--no-cache, see below). 

## Output (`classification.csv` columns)
- `id`       : SHA1 of original term
- `sentence` : Original term exactly as in file (underscores kept)
- `is_scientific` : true/false
- `confidence`    : 0-1 float (formatted to 3 decimals)
- `justification` : ≤2 concise sentences from model
- `model`         : Model name used

## CLI options

`classifier.py` accepts the following command-line options:

- `--backend` : choose the backend to call. Allowed values: `ollama` (default), `gemini`.
- `--model`   : model name to use. Defaults to `mistral` for Ollama and `gemini-2.5-flash-lite` for Gemini if not provided.
- `--input`, `-i` : path to the input TSV file (default: `data/terms.txt`). The input should contain lines like `index<TAB>term` or just the term.
- `--output`, `-o` : path to the output CSV file (default: `classification.csv`). The CSV will contain columns: `id, sentence, is_scientific, confidence, justification, model`.
- `--no-cache` : when present, ignore previous results and force re-generation (the existing output file will be deleted before running).

Examples:

```bash
# Use the local Ollama backend with the mistral model (default)
python classifier.py --backend ollama --model mistral

# Use Gemini backend and write results to a custom output file
python classifier.py --backend gemini --model gemini-2.5-flash-lite --output results.csv

# Re-run all terms (ignore cached results)
python classifier.py --no-cache
```

Notes:

- If you choose `--backend gemini`, make sure the `google-genai` package is installed and an API key is set in your environment (GEMINI_API_KEY, GOOGLE_API_KEY, or API_KEY).
- If you choose `--backend ollama`, make sure the `ollama` Python package is installed and the model is available locally (see Install section).
