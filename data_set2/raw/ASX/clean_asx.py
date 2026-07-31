import os
import json
import glob

def clean_and_partition_asx(raw_dir: str, output_dir: str):
    """
    Reads multiple raw ASX JSONL files, rounds high-precision floats,
    strips '.AX' from tickers, and writes clean rows into ticker-specific files.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Search for all .jsonl files in the raw directory
    raw_files = glob.glob(os.path.join(raw_dir, "*.jsonl"))
    
    if not raw_files:
        print(f"No .jsonl files found in {raw_dir}")
        return

    print(f"Found {len(raw_files)} raw JSONL file(s). Starting processing...")

    # We keep open file handles in a dict to avoid opening/closing files constantly
    open_files = {}

    try:
        for file_path in raw_files:
            print(f"Processing: {os.path.basename(file_path)}")
            
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Normalize ticker format (e.g., "IAG.AX" -> "IAG")
                        raw_ticker = data.get("ticker", "").strip()
                        clean_ticker = raw_ticker.replace(".AX", "").replace(".ax", "").upper()
                        
                        if not clean_ticker:
                            continue
                        
                        # Standardize and round numeric entries
                        cleaned_data = {
                            "ticker": clean_ticker,
                            "raw_ticker": raw_ticker,
                            "date": data.get("date", "").strip(),
                            "open": round(float(data.get("open", 0)), 2),
                            "high": round(float(data.get("high", 0)), 2),
                            "low": round(float(data.get("low", 0)), 2),
                            "close": round(float(data.get("close", 0)), 2),
                            "volume": int(data.get("volume", 0))
                        }
                        
                        # Get or create the file handle for this specific ticker
                        if clean_ticker not in open_files:
                            target_path = os.path.join(output_dir, f"{clean_ticker}.jsonl")
                            open_files[clean_ticker] = open(target_path, 'a', encoding='utf-8')
                        
                        # Write the clean row to the ticker's partitioned file
                        open_files[clean_ticker].write(json.dumps(cleaned_data) + "\n")
                        
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"Skipping malformed row in {os.path.basename(file_path)}: {e}")

        print(f"Cleanup complete. All files successfully partitioned in {output_dir}")
        
    finally:
        # Safely close all open file handles
        for handle in open_files.values():
            handle.close()

# Execute cleaning
if __name__ == "__main__":
    # Adjust directory paths to match your folder structures
    cwd_path = os.getcwd()
    clean_and_partition_asx(cwd_path + "/data/ASX", cwd_path + "/data/cleaned_asx")