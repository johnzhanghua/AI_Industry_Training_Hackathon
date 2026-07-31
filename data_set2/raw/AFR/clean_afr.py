import os
import json
import glob
import unicodedata
from datetime import datetime

def normalize_and_clean_text(text: str) -> str:
    """
    Removes hidden control characters, standardizes spacing, and normalizes
    Unicode text structures to prevent indexing mismatch issues.
    """
    if not text:
        return ""
    # NFKD decomposes characters, aligning accents and symbols
    text = unicodedata.normalize("NFKD", text)
    # Remove soft hyphens (\xad) which break text search matchers
    text = text.replace("\xad", "")
    # Replace non-breaking spaces (\xa0) with standard spaces
    text = text.replace("\xa0", " ")
    return text.strip()

def standardize_date(date_str: str) -> str:
    """
    Converts 'YYYYMMDD' (e.g., '20150131') to 'YYYY-MM-DD' (e.g., '2015-01-31').
    """
    if not date_str:
        return ""
    try:
        parsed_dt = datetime.strptime(date_str.strip(), "%Y%m%d")
        return parsed_dt.strftime("%Y-%m-%d")
    except ValueError:
        return date_str  # Return original if parsing fails

def clean_afr_dataset(raw_dir: str, output_dir: str):
    """
    Iterates over all JSONL files in the raw directory, cleans text fields,
    normalizes dates, and saves clean JSONL outputs.
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_files = glob.glob(os.path.join(raw_dir, "*.jsonl"))

    if not raw_files:
        print(f"No .jsonl files found in {raw_dir}")
        return

    print(f"Found {len(raw_files)} raw AFR file(s). Cleaning...")

    for file_path in raw_files:
        file_name = os.path.basename(file_path)
        target_path = os.path.join(output_dir, file_name)
        
        print(f"Cleaning file: {file_name}")
        
        with open(file_path, "r", encoding="utf-8") as infile, \
             open(target_path, "w", encoding="utf-8") as outfile:
            
            for line in infile:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    
                    cleaned_record = {
                        "headline": normalize_and_clean_text(data.get("HEADLINE")),
                        "subhead": normalize_and_clean_text(data.get("SUBHEAD")),
                        "intro": normalize_and_clean_text(data.get("INTRO")),
                        "text": normalize_and_clean_text(data.get("TEXT")),
                        "newspaper": normalize_and_clean_text(data.get("NEWSPAPER", "Australian Financial Review")),
                        "publication_date": standardize_date(data.get("PUBLICATIONDATE"))
                    }
                    
                    outfile.write(json.dumps(cleaned_record) + "\n")
                except Exception as e:
                    print(f"Skipping malformed row in {file_name}: {e}")

    print(f"AFR cleaning complete. Output saved to {output_dir}")

# Execute cleaning
if __name__ == "__main__":
    cwd_path = os.getcwd()
    clean_afr_dataset(cwd_path + "/data/AFR", cwd_path + "/data/cleaned_afr")