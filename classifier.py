"""
Unified scientific term classifier supporting both Ollama and Gemini backends.
Reads TSV input (index<TAB>term) and outputs classification results to CSV.
"""

import csv, hashlib, json, os, time, argparse
from pathlib import Path
from typing import Dict, Optional, Any
from tqdm import tqdm

# Optional imports with graceful fallback
try:
    import ollama
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False

try:
    from google import genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

try:
    from dotenv import load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False

# Load .env if available
if _DOTENV_AVAILABLE:
    try:
        from dotenv import load_dotenv as _load_dotenv
        _load_dotenv()
    except ImportError:
        pass

# Default configuration
DEFAULT_OLLAMA_MODEL = 'mistral'
DEFAULT_GEMINI_MODEL = 'gemini-2.5-flash-lite'
IN_PATH = Path('data/terms.txt')      # TSV: index<TAB>term
OUT_PATH = Path('classification.csv')  # term,is_scientific,confidence,justification
TEMP = 0.01
MAX_RETRIES = 4

SYSTEM_PROMPT = (
    "You are a strict scientific trigram classifier. All trigrams are separated by '_' instead of a space. "
    "Decide if a single term (trigram/phrase) is 'scientific' or 'not scientific'. "
    "Return JSON with keys: is_scientific (true/false), confidence (0-1 float), justification (<=2 sentences, concise). "
    "Be consistent and do not include any extra keys."
)

# Global clients (lazy initialization)
_gemini_client: Optional[Any] = None

def get_gemini_client():
    """Get or create Gemini client with API key from environment."""
    global _gemini_client
    if _gemini_client is None:
        if not _GENAI_AVAILABLE:
            raise RuntimeError("google-genai package not installed. Run: pip install google-genai")
        
        from google import genai
        api_key = (os.getenv('API_KEY')
        )
        if not api_key:
            raise RuntimeError(
                "Missing API key. Set one of GEMINI_API_KEY, GOOGLE_API_KEY, or API_KEY "
                "(e.g., in a .env file or exported in your shell)."
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client

def classify_ollama(sentence: str, model: str) -> Dict:
    """Classify using Ollama backend."""
    if not _OLLAMA_AVAILABLE:
        raise RuntimeError("ollama package not installed. Run: pip install ollama")
    
    import ollama
    
    user = f"Classify the sentence (can be a term/phrase):\n\n\"{sentence.strip()}\""
    last_exc = None
    
    for attempt in range(MAX_RETRIES):
        try:
            resp = ollama.chat(
                model=model,
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
                        "justification": f"Ollama error: {e}"}

    # Ensure a fallback return value in case loop exits unexpectedly
    return {"is_scientific": False, "confidence": 0.0, "justification": f"Unexpected error: {last_exc}"}

def classify_gemini(sentence: str, model: str) -> Dict:
    """Classify using Gemini backend."""
    user_text = f"Classify the term (underscores separate tokens): \n\n{sentence.strip()}"
    last_exc = None
    client = get_gemini_client()
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=SYSTEM_PROMPT + " " + user_text,
                config={
                    'response_mime_type': 'application/json',
                    'temperature': TEMP,
                    'top_p': 0.95
                })
            
            # Try to extract JSON text robustly
            content_text = None
            if hasattr(response, 'text') and response.text:
                content_text = response.text
            else:
                # Fallback: dig into candidates/parts
                candidates = getattr(response, 'candidates', []) or []
                for cand in candidates:
                    for part in getattr(cand, 'content', {}).get('parts', []):  # newer structure
                        if isinstance(part, dict) and 'text' in part:
                            content_text = part['text']
                            break
                    if content_text:
                        break
            
            if not content_text:
                raise ValueError('No text content in Gemini response')
            
            # Some models may wrap JSON in code fences
            content_text = content_text.strip()
            if content_text.startswith('```'):
                # remove possible ```json fences
                lines = [ln for ln in content_text.splitlines() if not ln.strip().startswith('```')]
                content_text = "\n".join(lines).strip()
            
            data = json.loads(content_text)
            
            # validate
            if not isinstance(data.get('is_scientific'), bool):
                raise ValueError("invalid 'is_scientific' type")
            if not isinstance(data.get('confidence'), (int, float)):
                raise ValueError("invalid 'confidence' type")
            if not isinstance(data.get('justification'), str):
                raise ValueError("invalid 'justification' type")
            return data
        except Exception as e:
            last_exc = e
            time.sleep(1.2 * (attempt + 1))
            if attempt == MAX_RETRIES - 1:
                return {"is_scientific": False, "confidence": 0.0,
                        "justification": f"Gemini error: {e}"}
    
    return {"is_scientific": False, "confidence": 0.0, "justification": f"Unexpected error: {last_exc}"}

def classify(sentence: str, backend: str, model: str) -> Dict:
    """Classify using the specified backend."""
    if backend == 'ollama':
        return classify_ollama(sentence, model)
    elif backend == 'gemini':
        return classify_gemini(sentence, model)
    else:
        raise ValueError(f"Unknown backend: {backend}")

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

def load_done(out_path: Path):
    """Load set of already-processed term IDs from existing output file."""
    done = set()
    if out_path.exists():
        with out_path.open(newline='', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                done.add(row["id"])
    return done

def main():
    parser = argparse.ArgumentParser(
        description="Classify terms as scientific vs non-scientific using Ollama or Gemini.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python classifier.py --backend ollama --model mistral
  python classifier.py --backend gemini --model gemini-2.5-flash-lite --no-cache
  python classifier.py --backend ollama --input data/terms.txt --output results.csv
        """
    )
    
    # Backend selection
    parser.add_argument('--backend', choices=['ollama', 'gemini'], default='ollama',
                        help='Backend to use for classification (default: ollama)')
    
    # Model selection
    parser.add_argument('--model', type=str,
                        help=f'Model to use (default: {DEFAULT_OLLAMA_MODEL} for ollama, {DEFAULT_GEMINI_MODEL} for gemini)')
    
    # File paths
    parser.add_argument('--input', '-i', type=Path, default=IN_PATH,
                        help=f'Input TSV file (default: {IN_PATH})')
    parser.add_argument('--output', '-o', type=Path, default=OUT_PATH,
                        help=f'Output CSV file (default: {OUT_PATH})')
    
    # Cache control
    parser.add_argument('--no-cache', action='store_true',
                        help='Ignore previous results and force re-generation (deletes existing output file)')
    
    args = parser.parse_args()
    
    # Set default model based on backend
    if args.model is None:
        args.model = DEFAULT_GEMINI_MODEL if args.backend == 'gemini' else DEFAULT_OLLAMA_MODEL
    
    # Validate backend availability
    if args.backend == 'ollama' and not _OLLAMA_AVAILABLE:
        print("Error: Ollama backend requested but 'ollama' package not installed.")
        print("Install with: pip install ollama")
        return 1
    elif args.backend == 'gemini' and not _GENAI_AVAILABLE:
        print("Error: Gemini backend requested but 'google-genai' package not installed.")
        print("Install with: pip install google-genai")
        return 1
    
    print(f"Using {args.backend} backend with model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    
    # Prepare output directory and handle cache
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.no_cache and args.output.exists():
        args.output.unlink()
        print("Deleted existing output file for fresh start")
    
    write_header = not args.output.exists()
    done = set() if args.no_cache else load_done(args.output)
    
    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist")
        return 1
    
    with args.input.open(encoding='utf-8') as f_in, \
         args.output.open('a', newline='', encoding='utf-8') as f_out:
        
        w = csv.DictWriter(f_out, fieldnames=["id","sentence","is_scientific","confidence","justification","model"])
        if write_header: 
            w.writeheader()

        raw_lines = [line.rstrip("\n") for line in f_in if line.strip()]
        # Extract only the term column
        terms = [extract_term(l) for l in raw_lines]
        
        skipped = 0
        for term in tqdm(terms, desc="Classifying"):
            if not term:
                continue
            _id = sha(term)
            if _id in done: 
                skipped += 1
                continue
            
            # Optional: convert underscores to spaces for better LLM understanding
            display_term = term.replace('_', ' ')
            res = classify(display_term, args.backend, args.model)
            
            w.writerow({
                "id": _id,
                "sentence": term,  # keep original raw term form for reproducibility
                "is_scientific": res["is_scientific"],
                "confidence": f'{float(res["confidence"]):.3f}',
                "justification": res["justification"].strip(),
                "model": args.model
            })
            f_out.flush()
    
    if skipped > 0:
        print(f"Skipped {skipped} already-processed terms (use --no-cache to re-process)")
    
    print(f"Classification complete. Results written to {args.output}")
    return 0

if __name__ == "__main__":
    exit(main())
