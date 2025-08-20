# classify.py
import csv, hashlib, json, time
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import ollama

MODEL = 'mistral'  # e.g., 'mistral' | 'llama3' | 'qwen2:7b'
IN_PATH = Path('data/test.txt')      # TSV: index<TAB>term
OUT_PATH = Path('classification.csv')  # term,is_scientific,confidence,justification
TEMP = 0.01
MAX_RETRIES = 1

SYSTEM_PROMPT = (
  "You are a strict scientific trigram classifier. All trigrams are separated by '_' instead of a space."
  "Decide if a single sentence is 'scientific' or 'not scientific'."
  "Return JSON with keys: is_scientific (true/false), confidence (0-1 float), justification (<=2 sentences, concise). "
  "Be consistent and do not include any extra keys."
)

def classify(sentence: str) -> Dict:
    # sentence here can be a short term/phrase; keep prompt wording for consistency
    user = f"Classify the sentence (can be a term/phrase):\n\n\"{sentence.strip()}\""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = ollama.chat(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user}],
                options={"temperature": TEMP},
                format="json",  # enforce JSON
            )
            # Ollama chat returns a dict-like (ChatResponse) with message.content holding JSON string
            content = resp["message"]["content"] if isinstance(resp, dict) else getattr(resp, "message").get("content", "{}")
            data = json.loads(content)
            # validate types explicitly and raise on mismatch to trigger a retry
            if not isinstance(data.get("is_scientific"), bool):
                raise ValueError("invalid 'is_scientific' type")
            if not isinstance(data.get("confidence"), (int, float)):
                raise ValueError("invalid 'confidence' type")
            if not isinstance(data.get("justification"), str):
                raise ValueError("invalid 'justification' type")
            return data
        except Exception as e:
            last_exc = e
            # backoff before retrying
            time.sleep(1.5 * (attempt + 1))
            if attempt == MAX_RETRIES - 1:
                # final fallback after exhausting retries
                return {"is_scientific": False, "confidence": 0.0,
                        "justification": f"LLM error: {e}"}

    # Ensure a fallback return value in case loop exits unexpectedly
    return {"is_scientific": False, "confidence": 0.0, "justification": f"Unexpected error: {last_exc}"}

def sha(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()

def extract_term(line: str) -> str:
    """Extract the term from a TSV line of the form 'index<TAB>term'.
    Falls back gracefully if the separator is missing. Also strips surrounding whitespace.
    """
    # Split only on first tab to allow tabs inside term (unlikely here)
    if "\t" in line:
        _idx, term = line.split("\t", 1)
    else:
        # Accept also if separated by multiple spaces
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            term = " ".join(parts[1:])
        else:
            term = line
    return term.strip()

def load_done():
    done = set()
    if OUT_PATH.exists():
        with OUT_PATH.open(newline='', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                done.add(row["id"])
    return done

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUT_PATH.exists()
    done = load_done()

    with IN_PATH.open(encoding='utf-8') as f_in, \
         OUT_PATH.open('a', newline='', encoding='utf-8') as f_out:
        w = csv.DictWriter(f_out, fieldnames=["id","sentence","is_scientific","confidence","justification","model"])
        if write_header: w.writeheader()

        raw_lines = [line.rstrip("\n") for line in f_in if line.strip()]
        # Extract only the term column
        terms = [extract_term(l) for l in raw_lines]
        for term in tqdm(terms, desc="Classifying"):
            if not term:
                continue
            _id = sha(term)
            if _id in done: continue
            # Optional: convert underscores to spaces for better LLM understanding
            display_term = term.replace('_', ' ')
            res = classify(display_term)
            w.writerow({
                "id": _id,
                "sentence": term,  # keep original raw term form for reproducibility
                "is_scientific": res["is_scientific"],
                "confidence": f'{float(res["confidence"]):.3f}',
                "justification": res["justification"].strip(),
                "model": MODEL
            })
            f_out.flush()

if __name__ == "__main__":
    main()