import pandas as pd
from datetime import datetime
import os

def clean_rba_dataset(input_csv: str, output_csv: str):
    """
    Cleans raw RBA cash rate CSV data:
    1. Standardizes dates to YYYY-MM-DD.
    2. Standardizes keys to lowercase snake_case.
    3. Converts string numbers to floats.
    4. Computes basis points (change_bps) from change percentage points.
    """
    if not os.path.exists(input_csv):
        print(f"Error: Raw file not found at {input_csv}")
        return

    # 1. Load the raw data
    df = pd.read_csv(input_csv)

    # Strip any whitespace from headers
    df.columns = df.columns.str.strip()

    # 2. Rename columns to standard snake_case
    df = df.rename(columns={
        "Effective Date": "effective_date",
        "Change % points": "change_pct",
        "Cash rate target%": "cash_rate_target"
    })

    # 3. Standardize Date Column to YYYY-MM-DD
    def parse_to_iso_date(date_str):
        try:
            # Parses formats like '3 Feb 2010' or '20 Mar 2020'
            parsed_dt = datetime.strptime(date_str.strip(), "%d %b %Y")
            return parsed_dt.strftime("%Y-%m-%d")
        except ValueError:
            return date_str  # Return original if parsing fails

    df["effective_date"] = df["effective_date"].apply(parse_to_iso_date)

    # 4. Clean and convert numeric values
    # Strip optional '+' signs from numeric strings (e.g., '+0.25' -> '0.25')
    df["change_pct"] = df["change_pct"].astype(str).str.replace("+", "", regex=False)
    df["change_pct"] = pd.to_numeric(df["change_pct"], errors="coerce").fillna(0.0)
    df["cash_rate_target"] = pd.to_numeric(df["cash_rate_target"], errors="coerce")

    # 5. Derive change_bps (Change in basis points)
    # 0.25% points = 25 bps. Multiply by 100 and round to integer.
    df["change_bps"] = (df["change_pct"] * 100).round().astype(int)

    # Save the cleaned dataset
    df.to_csv(output_csv, index=False)
    print(f"Successfully cleaned and saved dataset to {output_csv}")

# Execute the cleaning task
if __name__ == "__main__":
    # Update these paths as needed for your project structure

    cwd_path = os.getcwd() + "/data/RBA_Rates/"
    clean_rba_dataset(cwd_path+ "RBA-rates.csv", cwd_path + "rba_cash_rate_cleaned.csv")


