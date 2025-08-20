# scientific-classifier

A script that uses a local LLM to classify bi and trigrams as "scientific" or "not scientific". It reads terms from a TSV file, deduplicates them, and appends classification results to a CSV file. 

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
python classification.py
```
Progress bar shows classification status. You can stop and re-run; existing IDs are skipped.

## Output (`classification.csv` columns)
- `id`       : SHA1 of original term
- `sentence` : Original term exactly as in file (underscores kept)
- `is_scientific` : true/false
- `confidence`    : 0-1 float (formatted to 3 decimals)
- `justification` : ≤2 concise sentences from model
- `model`         : Model name used